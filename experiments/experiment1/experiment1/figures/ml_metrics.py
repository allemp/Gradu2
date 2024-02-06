# %%
import matplotlib.pyplot as plt
import pandas as pd
from aim import Repo

# %%

repo = Repo("../ray_results")

# %%
query = "(run.experiment == '2024-01-23-033632') and (metric.name in ['ray/tune/train_loss', 'ray/tune/test_loss', 'ray/tune/step', 'ray/tune/tol', 'ray/tune/batch_size', 'ray/tune/ram', 'ray/tune/cpu', 'ray/tune/accuracy', 'ray/tune/avg_step_time'])"
# %%
dfs = []
for run_metrics_col in repo.query_metrics(query).iter_runs():
    dfs.append(run_metrics_col.dataframe())

# %%
df = pd.concat(dfs).reset_index()
# %%
df2 = df.pivot(
    index=["run.trial_id", "step"], columns="metric.name", values="value"
).reset_index(drop=False)

# %%
df3 = df2[df2["ray/tune/batch_size"] == 1024]
df3 = df3[df3["ray/tune/tol"] == 0.01]
df3 = df3[df3["ray/tune/cpu"] > 0]

# %%
df3 = df3.drop("run.trial_id", axis=1).groupby(["step"]).mean().reset_index(drop=False)
df3["ray/tune/avg_step_time"] = df3["ray/tune/avg_step_time"] * 1000
# %%
plt.rcParams["text.usetex"] = True  # TeX rendering
plt.rcParams.update({"font.size": 18, "font.family": "serif", "figure.figsize": (8, 5)})
# plt.figure(figsize=(8, 5))

# Training error
fig, axs = plt.subplots(2, 1, sharex=True)
x = df3["ray/tune/step"]
y = df3["ray/tune/train_loss"]
label = r"$Training$"
axs[0].plot(x, y, label=label)


# Test Error
x = df3["ray/tune/step"]
y = df3["ray/tune/test_loss"]
label = r"$Test$"
axs[0].plot(x, y, label=label)

axs[0].set_ylim([0, 4 * 10**4])
axs[0].ticklabel_format(scilimits=(0, 0))


# axs[0].set_xlabel(r"$Steps$")
axs[0].set_ylabel(r"$Loss$")
axs[0].set_title(r"MNIST")

# Accuracy
x = df3["ray/tune/step"]
y = df3["ray/tune/accuracy"]
axs[1].plot(x, y)

axs[1].set_ylim([0.7, 0.9])

axs[1].set_xlabel(r"$Steps$")
axs[1].set_ylabel(r"$Accuracy$")
# axs[1].set_title(r"MNIST")

# Rest
axs[0].legend()
fig.tight_layout()

plt.savefig("../../../../thesis/assets/ml_metrics.png", dpi=300)

# %%
