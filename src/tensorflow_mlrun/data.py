import mlrun
import pandas as pd
from sklearn.datasets import load_diabetes


@mlrun.handler(outputs=["dataset", "label_column"])
def diabetes_data_generator():
    """
    A function which generates the diabetes dataset
    """
    diabetes = load_diabetes()
    diabetes_dataset = pd.DataFrame(
        data=diabetes.data, columns=diabetes.feature_names
    )
    diabetes_labels = pd.DataFrame(data=diabetes.target, columns=["label"])
    diabetes_dataset = pd.concat(
        [diabetes_dataset, diabetes_labels], axis=1
    )
    return diabetes_dataset, "label"



