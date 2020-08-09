import os
from typing import List

import pandas as pd
from xfeat import Pipeline


def xfeat_runner(pipelines: List,
                 input_df: pd.DataFrame,
                 output_filename: str) -> None:
    """Handling xfeat pipelines

    Args:
        pipelines (List): The definition of pipeline
        input_df (pd.DataFrame): Input pd.DataFrame
        output_filename (str): Output filename
    """
    if not os.path.exists(output_filename):
        print('Processing ...', [p.__class__.__name__ for p in pipelines])
        Pipeline(pipelines).fit_transform(input_df).reset_index(
            drop=True
        ).to_feather(output_filename)
    else:
        print('Skip ...', [p.__class__.__name__ for p in pipelines])
