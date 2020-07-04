import yaml

from sklearn.model_selection import StratifiedKFold

from ayniy.preprocessing.runner import Tabular


def test_preprocessing_runner():
    f = open('../experiments/configs/fe000.yml', 'r+')
    fe_configs = yaml.load(f, Loader=yaml.SafeLoader)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)

    tabular = Tabular(fe_configs, cv)
    tabular.create()
