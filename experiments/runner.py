"""
cd experiments
python runner.py --fe configs/fe000.yml --run configs/run000.yml
"""
import yaml
import argparse

from sklearn.model_selection import StratifiedKFold

from ayniy.preprocessing.runner import Tabular
from ayniy.model.runner import Runner


parser = argparse.ArgumentParser()
parser.add_argument('--fe')
parser.add_argument('--run')
args = parser.parse_args()

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)

if args.fe:
    f = open(args.fe, 'r+')
    fe_configs = yaml.load(f)

    tabular = Tabular(fe_configs, cv)
    tabular.create()

if args.run:
    g = open(args.run, 'r+')
    run_configs = yaml.load(g)

    runner = Runner(run_configs, cv)
    runner.run_train_cv()
    runner.run_predict_cv()
    runner.submission()
