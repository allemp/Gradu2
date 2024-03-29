# %%
print("Hello World")

# %%
import torch

# %%
torch.cuda.is_available()
# %%
torch.cuda.device_count()

# %%
torch.cuda.get_device_name(0)

# %%
import numpy as np

# %%
import ray
import torch
import torch.nn as nn
from ray import train
from ray.air import session
from ray.air.config import ScalingConfig
from ray.data.preprocessors import Chain, Concatenator, StandardScaler
from ray.train.torch import TorchCheckpoint, TorchTrainer

# %%
ray.init(num_gpus=1)
# %% Load data.
dataset = ray.data.read_csv("s3://anonymous@air-example-data/breast_cancer.csv")

# %% Split data into train and validation.
train_dataset, valid_dataset = dataset.train_test_split(test_size=0.3)

# %%Create a test dataset by dropping the target column.
test_dataset = valid_dataset.drop_columns(cols=["target"])

# %%
preprocessor = Chain(
    StandardScaler(columns=["mean radius", "mean texture"]),
    Concatenator(exclude=["target"], dtype=np.float32),
)


# %%
def create_model(input_features):
    return nn.Sequential(
        nn.Linear(in_features=input_features, out_features=16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
        nn.Sigmoid(),
    )


def train_loop_per_worker(config):
    batch_size = config["batch_size"]
    lr = config["lr"]
    epochs = config["num_epochs"]
    num_features = config["num_features"]

    # Get the Dataset shard for this data parallel worker,
    # and convert it to a PyTorch Dataset.
    train_data = session.get_dataset_shard("train")
    # Create model.
    model = create_model(num_features)
    model = train.torch.prepare_model(model)

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for cur_epoch in range(epochs):
        for batch in train_data.iter_torch_batches(
            batch_size=batch_size, dtypes=torch.float32
        ):
            # "concat_out" is the output column of the Concatenator.
            inputs, labels = batch["concat_out"], batch["target"]
            optimizer.zero_grad()
            predictions = model(inputs)
            train_loss = loss_fn(predictions, labels.unsqueeze(1))
            train_loss.backward()
            optimizer.step()
        loss = train_loss.item()
        session.report({"loss": loss}, checkpoint=TorchCheckpoint.from_model(model))


num_features = len(train_dataset.schema().names) - 1

trainer = TorchTrainer(
    train_loop_per_worker=train_loop_per_worker,
    train_loop_config={
        "batch_size": 128,
        "num_epochs": 20,
        "num_features": num_features,
        "lr": 0.001,
    },
    scaling_config=ScalingConfig(
        num_workers=1,  # Number of workers to use for data parallelism.
        use_gpu=True,  # so that the example works on Colab.
    ),
    datasets={"train": train_dataset},
    preprocessor=preprocessor,
)
# %%Execute training.
best_result = trainer.fit()
print(f"Last result: {best_result.metrics}")
# Last result: {'loss': 0.6559339960416158, ...}
# %%
