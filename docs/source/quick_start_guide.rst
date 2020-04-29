Quick Start Guide
===============================================================

.. code-block:: python

   from ayniy.preprocessing.runner import Tabular
   import yaml
   from sklearn.model_selection import StratifiedKFold


   f = open("configs/fe_00.yml", "r+")
   configs = yaml.load(f)
   cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)

   tabular = Tabular(configs, cv)
   tabular.create()
   X_train, X_test, y_train = tabular.load()
