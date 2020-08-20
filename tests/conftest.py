import seaborn as sns
import pytest


@pytest.fixture
def load_titanic():
    train = sns.load_dataset("titanic")
    test = train.copy()
    return train, test
