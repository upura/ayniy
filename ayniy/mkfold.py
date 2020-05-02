from collections import defaultdict, Counter

import random
import numpy as np
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.utils import check_random_state


def mkStratifiedKFold(train, use_col, n_splits=5, shuffle=True, name='fold_id'):
    """
    mkStratifiedKFold(train, configs['cols_definition']['target_col'], n_splits=5, shuffle=True, name='sk_fold_id')
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=7)
    train['fold_id'] = np.nan
    for i, (train_index, valid_index) in enumerate(cv.split(train, train[use_col])):
        train.loc[valid_index, 'fold_id'] = i
    train['fold_id'].to_csv(f'../input/{name}.csv', index=False)


def mkGroupKFold(train, use_col, n_splits=5, shuffle=True, name='fold_id'):
    """
    mkGroupKFold(train, 'session_id', n_splits=5, shuffle=True, name='group_fold_id')
    """
    cv = GroupKFold(n_splits=n_splits)
    groups = train[use_col]
    train['fold_id'] = np.nan
    for i, (train_index, valid_index) in enumerate(cv.split(train, train[use_col], groups)):
        train.loc[valid_index, 'fold_id'] = i
    train['fold_id'].to_csv(f'../input/{name}.csv', index=False)


def stratified_group_k_fold(X, y, groups, k, seed=None):
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std(
                [y_counts_per_fold[i][label] / y_distr[label]
                 for i in range(k)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)

    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(
            groups_and_y_counts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for i in range(k):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(k):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices


def mkStratifiedGroupKFold(train, target_col, group_col, n_splits=5, shuffle=True, name='fold_id'):
    cv = stratified_group_k_fold(train, train[target_col], train[group_col], k=n_splits, seed=7)
    train['fold_id'] = np.nan
    for i, (train_index, valid_index) in enumerate(cv):
        train.loc[valid_index, 'fold_id'] = i
    train['fold_id'].to_csv(f'../input/{name}.csv', index=False)


class RepeatedStratifiedGroupKFold():

    def __init__(self, n_splits=5, n_repeats=1, random_state=None):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    # Implementation based on this kaggle kernel:
    #    https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation
    def split(self, X, y=None, groups=None):
        k = self.n_splits

        def eval_y_counts_per_fold(y_counts, fold):
            y_counts_per_fold[fold] += y_counts
            std_per_label = []
            for label in range(labels_num):
                label_std = np.std(
                    [y_counts_per_fold[i][label] / y_distr[label] for i in range(k)]
                )
                std_per_label.append(label_std)
            y_counts_per_fold[fold] -= y_counts
            return np.mean(std_per_label)

        rnd = check_random_state(self.random_state)
        for repeat in range(self.n_repeats):
            labels_num = np.max(y) + 1
            y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
            y_distr = Counter()
            for label, g in zip(y, groups):
                y_counts_per_group[g][label] += 1
                y_distr[label] += 1

            y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
            groups_per_fold = defaultdict(set)

            groups_and_y_counts = list(y_counts_per_group.items())
            rnd.shuffle(groups_and_y_counts)

            for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
                best_fold = None
                min_eval = None
                for i in range(k):
                    fold_eval = eval_y_counts_per_fold(y_counts, i)
                    if min_eval is None or fold_eval < min_eval:
                        min_eval = fold_eval
                        best_fold = i
                y_counts_per_fold[best_fold] += y_counts
                groups_per_fold[best_fold].add(g)

            all_groups = set(groups)
            for i in range(k):
                train_groups = all_groups - groups_per_fold[i]
                test_groups = groups_per_fold[i]

                train_indices = [i for i, g in enumerate(groups) if g in train_groups]
                test_indices = [i for i, g in enumerate(groups) if g in test_groups]

                yield train_indices, test_indices
