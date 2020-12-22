import argparse
import os
import joblib
import numpy as np
from sklearn.model_selection import cross_val_score

import config
import model_dispatcher

from utils import read_train_data

def run(model, cv):
    train_data = read_train_data()
    cv = config.cv_schemes[cv]
    pipe  = model_dispatcher.models[model]

    X = train_data[train_data.columns[train_data.columns.str.contains('feature')]].values
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
