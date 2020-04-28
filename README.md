# Ayniy, All You Need is YAML

Ayniy is a supporting tool for machine learning competitions.

# Documentation

- [ayiny.preprocessing](docs/preprocessing.md)

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
