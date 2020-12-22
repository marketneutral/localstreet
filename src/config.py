from cross_val import PurgedGroupTimeSeriesSplit
from pathlib import Path

MODELS_PATH = Path('~/localstreet/models')

cv = PurgedGroupTimeSeriesSplit(
    n_splits=4,
    max_train_group_size=150,
    group_gap=20,
    max_test_group_size=60
)

cv_schemes = {
    'pgts_baseline': cv
}
