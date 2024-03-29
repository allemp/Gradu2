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
class CustomTrainer(BaseTrainer):
    def setup(self):
        self.model = SGDClassifier(max_iter=1000, tol=self.metadata["params"]["tol"])
        self.batch_size = self.metadata["params"]["batch_size"]
        self.tol = self.metadata["params"]["tol"]

    def training_loop(self):
        process = psutil.Process()
        classes = self.datasets["train"].unique("class")
        X_test = self.datasets["test"].to_pandas().drop("class", axis=1)
        y_test = self.datasets["test"].to_pandas()["class"]
        max_steps = 10000
        step = 0
        step_time_sum = 0
        while step <= max_steps:
            train_loss_sum = 0
            num_batches = 0
            train_batches = self.datasets["train"].iter_batches(
                batch_size=self.batch_size, batch_format="pandas"
            )

            for batch in train_batches:
                start_time = time()
                X = batch.drop("class", axis=1)
                y = batch["class"]
                self.model.partial_fit(X, y, classes=classes)

                batch_loss = hinge_loss(
                    y, self.model.decision_function(X), labels=classes
                )
                train_loss_sum += batch_loss
                num_batches += 1
                step += 1
                step_time_sum += time() - start_time
                if step >= max_steps:
                    break
                if step % 100 == 0:
                    pred = self.model.predict(X_test)
                    accuracy = accuracy_score(pred, y_test)
                    test_loss = hinge_loss(
                        y_test, self.model.decision_function(X_test), labels=classes
                    )
                    cpu = process.cpu_percent(interval=0.0)
                    mem_info = process.memory_info()
                    ram = (mem_info.rss - mem_info.shared) / 1024 / 1024
                    ray.train.report(
                        {
                            "step": step,
                            "train_loss": train_loss_sum / num_batches,
                            "test_loss": test_loss,
                            "accuracy": accuracy,
                            "cpu": cpu,
                            "ram": ram,
                            "avg_step_time": step_time_sum / step,
                            "batch_size": self.batch_size,
                            "tol": self.tol,
                        }
                    )


# %%
tune_config = tune.TuneConfig(num_samples=3, max_concurrent_trials=7)
run_config = ray.train.RunConfig(
    callbacks=[
        AimLoggerCallback(
            repo="/mnt/c/Users/master/Documents/University/Gradu2/experiments/experiment1/experiment1/ray_results",
        )
    ],
    storage_path="/mnt/c/Users/master/Documents/University/Gradu2/experiments/experiment1/experiment1/ray_results",
    name=datetime.now().strftime("%Y-%m-%d-%H%M%S"),
)
# %%

my_trainer = CustomTrainer(
    datasets={"train": train, "test": test},
    metadata={"params": {"batch_size": 1024, "tol": 0.01}},
)

# %%

my_tuner = ray.tune.Tuner(
    my_trainer,
    param_space={
        "metadata": {
            "params": {
                "batch_size": tune.grid_search([2**p for p in range(5, 6)]),
                "tol": tune.grid_search([10 ** (-p) for p in range(1, 5)]),
            }
        }
    },
    tune_config=tune_config,
    run_config=run_config,
)
# %%
my_tuner.fit()

# %%
