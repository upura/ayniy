# Ayniy, All You Need is YAML

Ayniy is a supporting tool for machine learning competitions.

# Documentation

https://upura.github.io/ayniy-docs/

## Environment

```
docker-compose build
docker-compose up
```

- Change url `6caefe97b41d` to `localhost`
- ref: [Dockerでデータ分析環境を手軽に作る方法](https://amalog.hateblo.jp/entry/data-analysis-docker)

## Test

```
docker-compose build
docker-compose up -d
docker exec -it ayniy-test bash
```

## Docs

```
docker-compose build
docker-compose up -d
docker exec -it ayniy-test bash
cd docs
make html
```

```
cd docs/build/html
git a .
git c "update"
git push origin master
```
