# Ayniy, All You Need is YAML

Ayniy is a supporting tool for machine learning competitions.

[**Documentation**](https://upura.github.io/ayniy-docs/) | [**Slide (Japanese)**](https://speakerdeck.com/upura/ayniy-with-mlflow)

```python
# Import packages
from sklearn.model_selection import StratifiedKFold
import yaml

from ayniy.model.runner import Runner

# Load configs
f = open('configs/run000.yml', 'r+')
configs = yaml.load(f, Loader=yaml.SafeLoader)

# Difine CV strategy as you like
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)

# Modeling
runner = Runner(configs, cv)
runner.run_train_cv()
runner.run_predict_cv()
runner.submission()
```

## Examples

| Competition Name | Score (Rank) | Repository |
| --- | --- | --- | 
| [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic/) | 0.76555 (-) | [Link](https://github.com/upura/ayniy-titanic) |
| [Basketball Behavior Challenge BBC2020](https://competitions.codalab.org/competitions/23905) | 0.871700 (1) | [Link](https://github.com/upura/basketball-behavior-challenge) |
| [ひろしまQuest2020#stayhome【アイデア部門】](https://signate.jp/competitions/277) | - | [Link](https://github.com/upura/signate-hiroshima-quest-idea) |

## Environment

```
docker-compose -d --build
docker exec -it ayniy-test bash
```

## MLflow

```
cd experiments
mlflow ui -h 0.0.0.0
```

## Test

```bash
# pytest
pytest tests/ --cov=. --cov-report=html
# black
black .
# flake8
flake8 .
# mypy
mypy .
```

## Docs
In container,
```
cd docs
make html
```

Out of container,
```
sh deploy.sh
```
https://github.com/upura/ayniy-docs
