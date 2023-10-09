# %%
import pandas as pd
import psutil
import ray
from pmlb import fetch_data
from ray import tune
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.train.trainer import BaseTrainer
from ray.tune import Tuner
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, hinge_loss

# %%
raw_df = fetch_data("mnist")
# %%
data = ray.data.from_pandas(raw_df).repartition(7).random_shuffle()
# %%
train, test = data.train_test_split(0.3)


# %%
class CustomTrainer(BaseTrainer):
    def setup(self):
        self.model = SGDClassifier(max_iter=1000, tol=1e-3)
        self.batch_size = self.metadata["params"]["batch_size"]

    def training_loop(self):
        process = psutil.Process()
        classes = self.datasets["train"].unique("target")
        X_test = self.datasets["test"].to_pandas().drop("target", axis=1)
        y_test = self.datasets["test"].to_pandas()["target"]
        max_steps = 10000
        step = 0
        while step <= max_steps:
            train_loss_sum = 0
            num_batches = 0
            train_batches = self.datasets["train"].iter_batches(
                batch_size=self.batch_size, batch_format="pandas"
            )

            for batch in train_batches:
                print("Step: ", step)
                X = batch.drop("target", axis=1)
                y = batch["target"]
                self.model.partial_fit(X, y, classes=classes)
                batch_loss = hinge_loss(y, self.model.decision_function(X))
                train_loss_sum += batch_loss
                num_batches += 1
                step += 1
                if step >= max_steps:
                    break
                if step % 10 == 0:
                    pred = self.model.predict(X_test)
                    accuracy = accuracy_score(pred, y_test)
                    test_loss = hinge_loss(y_test, self.model.decision_function(X_test))
                    cpu = process.cpu_percent(interval=0.0)
                    ram = process.memory_info().rss
                    ray.train.report(
                        {
                            "step": step,
                            "train_loss": train_loss_sum / num_batches,
                            "test_loss": test_loss,
                            "accuracy": accuracy,
                            "cpu": cpu,
                            "ram": ram,
                        }
                    )


# %%
tune_config = tune.TuneConfig(num_samples=1, max_concurrent_trials=4)
run_config = ray.train.RunConfig(
    name="MLFlow", callbacks=[MLflowLoggerCallback(experiment_name="test1")]
)
# %%

my_trainer = CustomTrainer(
    datasets={"train": train, "test": test}, metadata={"params": {"batch_size": 1024}}
)

# %%

my_tuner = Tuner(
    my_trainer,
    param_space={
        "metadata": {
            "params": {"batch_size": tune.grid_search([2**i for i in range(6, 15)])}
        }
    },
    tune_config=tune_config,
    run_config=run_config,
)
# %%
my_tuner.fit()

# %%
# result_df.plot(x="step", y=["ram"])
# %%
