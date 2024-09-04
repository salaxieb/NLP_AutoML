import pandas as pd
import pytest


@pytest.fixture(scope="session")
def dataset():
    dataset = pd.read_csv("./tests/fixtures/avito1k_train.csv")
    return dataset.dropna(subset=["description"])
