import mlrun
import pandas as pd
from sklearn.datasets import load_wine


@mlrun.handler(outputs=["dataset", "label_column"])
def wine_data_generator():
    """
    A function which generates the wine dataset
    """
    wine = load_wine()
    wine_dataset = pd.DataFrame(
        data=wine.data, columns=wine.feature_names
    )
    wine_labels = pd.DataFrame(data=wine.target, columns=["label"])
    wine_dataset = pd.concat(
        [wine_dataset, wine_labels], axis=1
    )
    return wine_dataset, "label"



