# %%
import time
from typing import Dict, List

import psutil
import pynvml
import ray
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.train.sklearn import SklearnTrainer
from ray.tune.logger import TBXLoggerCallback
from sklearn.ensemble import RandomForestRegressor

# %%

# %%
train_dataset = ray.data.from_items([{"x": x, "y": x + 1} for x in range(32000)])
# %%
scaling_config = ray.air.config.ScalingConfig(trainer_resources={"CPU": 4})


# %%
class MyCallback(ray.tune.logger.LoggerCallback):
    def __init__(self) -> None:
        self.step_results = []
        self.results = []

    def log_trial_result(self, iteration: int, trial: "Trial", result: Dict):
        result["system_metrics"] = self.step_results
        self.step_results = []
        self.results.append(result)

    def on_step_end(self, iteration, trials, **info):
        if iteration % 1 == 0:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            cpu = psutil.cpu_percent()
            ram = psutil.virtual_memory().percent
            system_metrics = {
                "iteration": iteration,
                "cpu_util_percent": cpu,
                "memory_percent": ram,
                "gpu_util": util.gpu,
                "gpu_memory": util.memory,
            }
            self.step_results.append(system_metrics)
            # print(
            #    f"iteration {iteration},CPU {cpu}, Memory {ram}, gpu-util: {util.gpu} | gpu-mem: {util.memory} |"
            # )


# %%
my_callback = MyCallback()
run_config = ray.air.RunConfig(
    verbose=2,
    callbacks=[
        my_callback,
        MLflowLoggerCallback(experiment_name="train_experiment"),
        TBXLoggerCallback(),
    ],
)
# %%
trainer = SklearnTrainer(
    run_config=run_config,
    estimator=RandomForestRegressor(),
    label_column="y",
    scaling_config=scaling_config,
    datasets={"train": train_dataset},
)

# %%
result1 = trainer.fit()
# %%
# print(my_callback.results)

# %%
