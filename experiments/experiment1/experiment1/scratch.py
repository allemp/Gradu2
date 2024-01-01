# %%
from datetime import datetime
from time import time

import openml
import pandas as pd
import psutil
import ray
from ray import tune
from ray.train.trainer import BaseTrainer
from ray.tune.logger.aim import AimLoggerCallback
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, hinge_loss

# %%
openml_dataset = openml.datasets.get_dataset(554)
# %%
data = ray.data.from_pandas(openml_dataset.get_data()[0]).repartition(7)
# %%
train, test = data.train_test_split(test_size=0.3, shuffle=True, seed=2023)

# %%
k = 10
ds = train.split_proportionately([1 / k] * (k - 1))
# %%
