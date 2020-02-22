# ayiny.preprocessing

## ayiny.preprocessing.tabular

### Count null

```python
train, test = count_null(train, test, {'encode_col': train.columns})
```

### Label encoding

```python
train, test = label_encoding(train, test, {'encode_col': categorical_cols})
```

### Count encoding

```python
train, test = count_encoding(train, test, {'encode_col': categorical_cols})
```

### Count encoding interaction

```python
train, test = count_encoding_interact(train, test, {'encode_col': categorical_cols})
```

### Matrix factorization

```python
train, test = matrix_factorization(
    train, test,
    col_definition={'encode_col': configs['preprocessing']['matrix_factorization']},
    option={'n_components_lda': 5, 'n_components_svd': 3})
```

### Aggregation

```python
train, test = aggregation(
    train, test,
    col_definition={'groupby_dict': configs['preprocessing']['aggregation']['groupby_dict'],
                    'nunique_dict': configs['preprocessing']['aggregation']['nunique_dict']})
```

### Numeric interaction

```python
train, test = numeric_interact(train, test, {'encode_col': numerical_col})
```

### Target encoding

```python
train, test = target_encoding(
    train, test,
    col_definition={'encode_col': configs['preprocessing']['target_encoding'],
                    'target_col': configs['cols_definition']['target_col']},
    option={'cv': cv})
```

### Use col

```python
train, test = delete_cols(train, test, {'encode_col': use_col})
```

### Delete col

```python
unique_cols, duplicated_cols, high_corr_cols = detect_delete_cols(
    train, test,
    col_definition={'escape_col': categorical_cols},
    option={'threshold': 0.995})
train, test = delete_cols(train, test, unique_cols + duplicated_cols + high_corr_cols)
```

### Save as pickle

```python
save_as_pickle(
    train, test,
    col_definition={'target_col': configs['cols_definition']['target_col']},
    option={'exp_id': ''})
```

## ayiny.preprocessing.text

### Text normalization

```python
train, test = text_normalize(train, test,
                             col_definition={'text_col': 'headline'})
```

### [Simple Word-Embedding-based Methods](https://arxiv.org/abs/1805.09843)

- 'agg'
    - {'max', 'mean'}
    - SWEM-max or SWEM-mean

```python
train, test = get_swem(train, test,
                       col_definition={'text_col': 'headline',
                                       'target_col': 'bookmark'},
                       option={'agg': 'max', 'lang': 'ja'})
```

### Bag of Words

```python
train, test = get_count(train, test,
                        col_definition={'text_col': 'headline',
                                        'target_col': 'bookmark'},
                        option={'n_components': 5, 'lang': 'ja'})
```

### TF-IDF

```python
train, test = get_tfidf(train, test,
                        col_definition={'text_col': 'headline',
                                        'target_col': 'bookmark'},
                        option={'n_components': 5, 'lang': 'ja'})
```

### [SCDV: Sparse Composite Document Vectors using soft clustering over distributional representations](https://dheeraj7596.github.io/SDV/)

```python
train, test = get_scdv(train, test,
                       col_definition={'text_col': 'headline',
                                       'target_col': 'bookmark'},
                       option={'n_components': 5, 'lang': 'ja'})
```
