Quick Start Guide
===============================================================

.. code-block:: python

   # Notebook settings
   from ayniy.utils import init_notebook
   init_notebook()
   %load_ext autoreload
   %autoreload 2

   # Import packages
   import yaml
   from sklearn.model_selection import StratifiedKFold
   from ayniy.preprocessing.runner import Tabular
   from ayniy.model.runner import Runner

   # Load configs
   f = open('configs/fe000.yml', 'r+')
   feConfigs = yaml.load(f)
   g = open('configs/exp000.yml', 'r+')
   expConfigs = yaml.load(g)

   # CV strategy
   cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)

   # Feature engineering
   tabular = Tabular(feConfigs, cv)
   tabular.create()
   X_train, X_test, y_train = tabular.load()

   # Modeling
   runner = Runner(expConfigs, cv)
   runner.run_train_cv()
   runner.run_predict_cv()
   runner.submission()
