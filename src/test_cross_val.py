from cross_val import PurgedGroupTimeSeriesSplit
import numpy as np

def test_purged_cv():
    n_samples = 2000
    n_groups = 20
    assert n_samples % n_groups == 0

    idx = np.linspace(0, n_samples-1, num=n_samples)
    X_train = np.random.random(size=(n_samples, 5))
    y_train = np.random.choice([0, 1], n_samples)
    groups = np.repeat(np.linspace(0, n_groups-1, num=n_groups), n_samples/n_groups)

    cv = PurgedGroupTimeSeriesSplit(
        n_splits=5,
        max_train_group_size=7,
        group_gap=2,
        max_test_group_size=3
    )

    for ii, (tr, tt) in enumerate(cv.split(X=X_train, y=y_train, groups=groups)):
        print(f'tr: {tr}')
        print(f'tt: {tt}')

    return True
