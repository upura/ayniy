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

| Platform | Competition Name | Rank | Repository |
| --- | --- | --- | --- | 
| CodaLab | [Basketball Behavior Challenge BBC2020](https://competitions.codalab.org/competitions/23905) | 1 | [Link](https://github.com/upura/basketball-behavior-challenge) |
| Nishika| [財務・非財務情報を活用した株主価値予測](https://www.nishika.com/competitions/4/summary) | 2 | [Link](https://github.com/upura/nishika-yuho) |
| SIGIR2021| [SIGIR eCOM 2021 Data Challenge](https://github.com/upura/sigir-ecom-2021) | 3 | [Link](https://sigir-ecom.github.io/data-task.html) |
| SIGNATE | [ひろしまQuest2020#stayhome【アイデア部門】](https://signate.jp/competitions/277) | - | [Link](https://github.com/upura/signate-hiroshima-quest-idea) |
| ProbSpace | [YouTube動画視聴回数予測](https://prob.space/competitions/youtube-view-count) | 6 | [Link](https://github.com/upura/probspace-youtube) |
| Solafune | [夜間光データから土地価格を予測](https://solafune.com/#/competitions/f03f39cc-597b-4819-b1a5-41479d4b73d6) | 6 | [Link](https://github.com/upura/solafune-light) |
| atmaCup | [#8 [初心者向] atmaCup](https://www.guruguru.science/competitions/13/) | - | [Link](https://github.com/upura/atmaCup8) |
| Kaggle | [WiDS Datathon 2021](https://www.kaggle.com/c/widsdatathon2021) | 64 | [Link](https://github.com/upura/widsdatathon2021) |
| Kaggle | [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic/) | - | [Link](https://github.com/upura/ayniy-titanic) |

## Starter Kit

### Scripts

```bash
mkdir project_dir
cd project_dir
sh start.sh
```

[kaggle_utils](https://github.com/upura/kaggle_utils/tree/update-refactoring) is used for feature engineering.

#### Environment

```bash
docker-compose -d --build
docker exec -it ayniy-test bash
```

#### MLflow

```bash
cd experiments
mlflow ui -h 0.0.0.0
```

### Kaggle Notebook

```bash
!git clone https://github.com/upura/ayniy
import sys
sys.path.append("/kaggle/working/ayniy")
!pip install -r /kaggle/working/ayniy/requirements.txt
!mkdir '../output/'
!mkdir '../output/logs'
from sklearn.model_selection import StratifiedKFold
from ayniy.model.runner import Runner
```

## For Developers

### Test

```bash
pysen run lint
pysen run format
```

### Docs
In container,
```bash
cd docs
make html
```

Out of container,
```bash
sh deploy.sh
```
https://github.com/upura/ayniy-docs
