# %%
from datetime import datetime
from itertools import cycle
from pathlib import Path

import openml
import pandas as pd
import ray
from experiment1.experiment1_utils import RegressorTrainer
from ray import tune
from ray.tune.logger.aim import AimLoggerCallback

# %%
time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(time_now)
tune_config = tune.TuneConfig(num_samples=3, max_concurrent_trials=7)
run_config = ray.train.RunConfig(
    callbacks=[
        AimLoggerCallback(
            repo=Path.cwd().joinpath("ray_results"),
        )
    ],
    storage_path=Path.cwd().joinpath("ray_results"),
    name=f"red_wine {time_now}",
)
# %%

openml_dataset = openml.datasets.get_dataset(44972)
data = ray.data.from_pandas(openml_dataset.get_data()[0]).repartition(7)
train, test = data.train_test_split(test_size=0.3, shuffle=True, seed=2023)
# %%

my_trainer = RegressorTrainer(
    datasets={"train": train, "test": test},
    metadata={
        "params": {
            "target_variable": "quality",
            "max_steps": 25000,
            "batch_size": 32,
            "tol": 0.01,
            "loss_fn": "squared_error",
        }
    },
)

# %%

my_tuner = ray.tune.Tuner(
    my_trainer,
    tune_config=tune_config,
    run_config=run_config,
    param_space={
        "metadata": {
            "params": {
                "batch_size": tune.grid_search([2**p for p in range(5, 6)]),
                "tol": tune.grid_search([10 ** (-p) for p in range(4, 5)]),
                "loss_fn": tune.grid_search(["squared_error"]),
            }
        }
    },
)
# %%
my_tuner.fit()

# %%
