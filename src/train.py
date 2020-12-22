import argparse
import os
import joblib
import numpy as np
from sklearn.model_selection import cross_val_score
import uuid
import logging
import sys

import config
import model_dispatcher
from utils import read_train_data

logger = logging.getLogger('')
logger.setLevel(logging.DEBUG)


def run(model, cv):
    experiment_id = f'{str(uuid.uuid4())}'
    
    train_data = read_train_data()
    cv = config.cv_schemes[cv]
    pipe  = model_dispatcher.models[model]

    # note we pass in all data as pandas df
    # it is up to the model pipeline
    # to select which cols it wants
    X = train_data
    
    groups = train_data['date'].values
    y = (train_data['resp']>0).astype(int).values

    cv_scores = cross_val_score(
        pipe,
        X,
        y,
        cv=cv,
        groups=groups,
        scoring='roc_auc',
        verbose=10
    )
    print(cv_scores)
    print(f'mean: {np.mean(cv_scores)}   std: {np.std(cv_scores)}')

    # save the model for reproducibility later
    # we haven't fit the model though, just run CV
    joblib.dump(
        pipe,
        f'../models/{experiment_id}_pipe_notfitted.bin'
    )
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--cv',
        type=str,
        default='pgts_baseline'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='baseline_log_reg'
    )

    args = parser.parse_args()

    run(cv=args.cv, model=args.model)
