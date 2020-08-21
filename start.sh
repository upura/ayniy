# directory
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

# README.md and .gitkeep
touch input/.gitkeep
touch output/importance/.gitkeep
touch output/logs/.gitkeep
touch output/model/.gitkeep
touch output/pred/.gitkeep
touch output/submissions/.gitkeep
touch README.md
