Quick Start Guide
===============================================================

.. code-block:: python

   # Import packages
   import yaml
   from sklearn.model_selection import StratifiedKFold
   from ayniy.preprocessing.runner import Tabular
   from ayniy.model.runner import Runner

   # Load configs
   f = open('configs/fe000.yml', 'r+')
   fe_configs = yaml.load(f)
   g = open('configs/run000.yml', 'r+')
   run_configs = yaml.load(g)

   # CV strategy
   cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)

   # Feature engineering
   tabular = Tabular(fe_configs, cv)
   tabular.create()
   # X_train, X_test, y_train = tabular.load()

   # Modeling
   runner = Runner(run_configs, cv)
   runner.run_train_cv()
   runner.run_predict_cv()
   runner.submission()
