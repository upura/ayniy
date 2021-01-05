# directory
mkdir input
mkdir input/pickle
mkdir input/feather
mkdir input
mkdir experiments
mkdir output
mkdir output/importance
mkdir output/logs
mkdir output/model
mkdir output/pred
mkdir output/submissions

# template
cp -r ../ayniy/ayniy/ ayniy/
cp ../ayniy/.dockerignore .dockerignore
cp ../ayniy/setup.cfg setup.cfg
cp ../ayniy/.gitignore .gitignore
cp ../ayniy/Dockerfile Dockerfile
cp ../ayniy/docker-compose.yml docker-compose.yml
cp ../ayniy/requirements.txt requirements.txt
cp ../ayniy/experiments/runner.py experiments/runner.py

# kaggle_utils
cp -r ../kaggle_utils/kaggle_utils/ kaggle_utils/

# README.md and .gitkeep
touch input/pickle/.gitkeep
touch input/feather/.gitkeep
touch output/importance/.gitkeep
touch output/logs/.gitkeep
touch output/model/.gitkeep
touch output/pred/.gitkeep
touch output/submissions/.gitkeep
touch README.md
