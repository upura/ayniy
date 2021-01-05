Setup
===================================

When you use local editor like VSCode, you should create a project folder as follows.

.. code-block:: bash

   mkdir project_dir
   cd project_dir
   sh start.sh

You can use docker environment as follows.

.. code-block:: bash

   docker-compose -d --build
   docker exec -it {container_name} bash

When you use notebook editor like Kaggle Notebook, you can install packages as follows.

.. code-block:: bash

   !git clone https://github.com/upura/ayniy
   import sys
   sys.path.append("/kaggle/working/ayniy")
   !pip install -r /kaggle/working/ayniy/requirements.txt
   !mkdir '../output/'
   !mkdir '../output/logs'
   from sklearn.model_selection import StratifiedKFold
   from ayniy.model.runner import Runner
