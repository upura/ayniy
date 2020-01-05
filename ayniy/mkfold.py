import numpy as np
from sklearn.model_selection import StratifiedKFold


def mkStratifiedKFold(train, use_col, n_splits=5, shuffle=True, name='fold_id'):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=7)
    train['fold_id'] = np.nan
    for i, (train_index, valid_index) in enumerate(cv.split(train, train[use_col])):
        train.loc[valid_index, 'fold_id'] = i
    train['fold_id'].to_csv(f'../input/{name}.csv', index=False)
