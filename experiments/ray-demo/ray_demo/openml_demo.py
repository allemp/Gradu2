# %%
import openml
import pandas as pd

# %%
dataset_df = openml.datasets.get_dataset(61)
# %%
type(dataset_df)

# %%
X, y, *other = dataset_df.get_data()
# %%
