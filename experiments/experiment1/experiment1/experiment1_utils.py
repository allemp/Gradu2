from time import time

import pandas as pd
import psutil
import ray
from ray.train.trainer import BaseTrainer
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.metrics import accuracy_score, hinge_loss, log_loss, mean_squared_error


def loss_fn_wrapper(loss_fn, *args, **kwargs):
    if loss_fn == "hinge":
        return hinge_loss(*args, **kwargs)
    if loss_fn == "perceptron":
        return hinge_loss(*args, **kwargs)
    if loss_fn == "log_loss":
        return log_loss(*args, **kwargs)


class RegressorTrainer(BaseTrainer):
    def setup(self):
        self.batch_size = self.metadata["params"]["batch_size"]
        self.tol = self.metadata["params"]["tol"]
        self.loss_fn = self.metadata["params"]["loss_fn"]
        self.max_steps = self.metadata["params"]["max_steps"]
        self.model = SGDRegressor(
            loss=self.loss_fn, max_iter=self.max_steps, tol=self.tol
        )

    def training_loop(self):
        process = psutil.Process()
        X_test = self.datasets["test"].to_pandas().drop("quality", axis=1)
        y_test = self.datasets["test"].to_pandas()["quality"]
        max_steps = self.max_steps
        step = 0
        step_time_sum = 0
        train_loss_sum = 0
        num_batches = 0
        epoch = 0

        while step < max_steps:
            epoch += 1
            train_batches = self.datasets["train"].iter_batches(
                batch_size=self.batch_size, batch_format="pandas"
            )

            for batch in train_batches:
                start_time = time()
                X = batch.drop("quality", axis=1)
                y = batch["quality"]
                self.model.partial_fit(X, y)

                batch_loss = mean_squared_error(
                    y,
                    self.model.predict(X),
                )
                train_loss_sum += batch_loss
                num_batches += 1
                step += 1
                step_time_sum += time() - start_time
                if step >= max_steps:
                    break
                if step % 100 == 0:
                    test_loss = mean_squared_error(
                        y,
                        self.model.predict(X),
                    )
                    cpu = process.cpu_percent(interval=0.0)
                    mem_info = process.memory_info()
                    ram = (mem_info.rss - mem_info.shared) / 1024 / 1024
                    ray.train.report(
                        {
                            "epoch": epoch,
                            "step": step,
                            "train_loss": train_loss_sum / num_batches,
                            "test_loss": test_loss,
                            "cpu": cpu,
                            "ram": ram,
                            "avg_step_time": step_time_sum / step,
                            "batch_size": self.batch_size,
                        }
                    )


class ClassifierTrainer(BaseTrainer):
    def setup(self):
        self.batch_size = self.metadata["params"]["batch_size"]
        self.tol = self.metadata["params"]["tol"]
        self.loss_fn = self.metadata["params"]["loss_fn"]
        self.max_steps = self.metadata["params"]["max_steps"]
        self.model = SGDClassifier(
            loss=self.loss_fn, max_iter=self.max_steps, tol=self.tol
        )

    def training_loop(self):
        process = psutil.Process()
        classes = self.datasets["train"].unique("class")
        X_test = self.datasets["test"].to_pandas().drop("class", axis=1)
        y_test = self.datasets["test"].to_pandas()["class"]
        max_steps = self.max_steps
        step = 0
        step_time_sum = 0
        train_loss_sum = 0
        num_batches = 0
        epoch = 0

        while step < max_steps:
            epoch += 1
            train_batches = self.datasets["train"].iter_batches(
                batch_size=self.batch_size, batch_format="pandas"
            )

            for batch in train_batches:
                start_time = time()
                X = batch.drop("class", axis=1)
                y = batch["class"]
                self.model.partial_fit(X, y, classes=classes)

                batch_loss = loss_fn_wrapper(
                    self.loss_fn, y, self.model.decision_function(X), labels=classes
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
                    test_loss = loss_fn_wrapper(
                        self.loss_fn,
                        y_test,
                        self.model.decision_function(X_test),
                        labels=classes,
                    )
                    cpu = process.cpu_percent(interval=0.0)
                    mem_info = process.memory_info()
                    ram = (mem_info.rss - mem_info.shared) / 1024 / 1024
                    ray.train.report(
                        {
                            "epoch": epoch,
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
