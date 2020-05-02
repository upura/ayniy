# Ayniy, All You Need is YAML

Ayniy is a supporting tool for machine learning competitions.

[**Documentation**](https://upura.github.io/ayniy-docs/) | [**Slide (Japanese)**](https://speakerdeck.com/upura/introduction-ayniy)

## Environment

```
docker-compose build
docker-compose up
```

## MLflow

```
cd experiments
mlflow ui
```

## Test

```
docker-compose build
docker-compose up -d
docker exec -it ayniy-test bash
```
```
cd tests
pytest --cov=. --cov-report=html
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
