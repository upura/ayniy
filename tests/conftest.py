from typing import Tuple

import pandas as pd
import pytest
import seaborn as sns


@pytest.fixture
def load_titanic() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = sns.load_dataset("titanic")
    test = train.copy()
    return train, test
