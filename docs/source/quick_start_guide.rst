Quick Start Guide
===============================================================

.. code-block:: python

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
