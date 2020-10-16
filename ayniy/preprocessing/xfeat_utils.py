import os
from typing import List

import pandas as pd
from sklearn.model_selection import KFold
from xfeat import Pipeline, TargetEncoder


def xfeat_runner(pipelines: List, input_df: pd.DataFrame, output_filename: str) -> None:
    """Handling xfeat pipelines

    Args:
        pipelines (List): The definition of pipeline
        input_df (pd.DataFrame): Input pd.DataFrame
        output_filename (str): Output filename
    """
    if not os.path.exists(output_filename):
        print("Processing ...", [p.__class__.__name__ for p in pipelines])
        Pipeline(pipelines).fit_transform(input_df).reset_index(drop=True).to_feather(
            output_filename
        )
    else:
        print("Skip ...", [p.__class__.__name__ for p in pipelines])


def xfeat_target_encoding(
    target_col: str, input_df: pd.DataFrame, output_filename: str
) -> None:

    _train = input_df.dropna(subset=[target_col]).copy()
    _test = input_df.loc[input_df[target_col].isnull()].copy()

    fold = KFold(n_splits=5, shuffle=True, random_state=111)
    encoder = TargetEncoder(fold=fold, target_col=target_col, output_suffix="")
    _train = encoder.fit_transform(_train)
    _test = encoder.transform(_test)

    pd.concat([_train, _test], sort=False).drop(target_col, axis=1).reset_index(
        drop=True
    ).to_feather(output_filename)
