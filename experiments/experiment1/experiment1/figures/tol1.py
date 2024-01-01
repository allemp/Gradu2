# %%
import matplotlib.pyplot as plt
import pandas as pd
from aim import Repo

# %%

repo = Repo("../ray_results")

# %%
query = "(run.experiment == '2023-12-07-115243') and (metric.name in ['ray/tune/train_loss', 'ray/tune/test_loss', 'ray/tune/step', 'ray/tune/tol', 'ray/tune/batch_size', 'ray/tune/ram', 'ray/tune/cpu', 'ray/tune/accuracy', 'ray/tune/avg_step_time'])"
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
df3 = df2[df2["ray/tune/batch_size"] == 32]
df3 = df3[df3["ray/tune/cpu"] > 0]
df3 = (
    df3.drop("run.trial_id", axis=1)
    .groupby(["ray/tune/tol", "step"])
    .mean()
    .reset_index(drop=False)
)
df3["ray/tune/avg_step_time"] = df3["ray/tune/avg_step_time"] * 1000
# %%
plt.rcParams["text.usetex"] = True  # TeX rendering
plt.rcParams.update(
    {"font.size": 18, "font.family": "serif", "figure.figsize": (8, 10)}
)
# plt.figure(figsize=(8, 5))

# Tolerations

tols = df3["ray/tune/tol"].unique().tolist()
tols.sort(reverse=True)

# RAM
fig, axs = plt.subplots(3, 1, sharex=True)
for tol in tols:
    x = df3[df3["ray/tune/tol"] == tol]["ray/tune/step"]
    y = df3[df3["ray/tune/tol"] == tol]["ray/tune/ram"]
    label = r"$tol = " + str(tol) + r"$"
    axs[0].plot(x, y, label=label)

# axs[0].set_xlabel(r"$Steps$")
axs[0].set_ylabel(r"$RAM$" + "\n" + r"$(MB)$")
axs[0].set_title(r"MNIST")

# CPU
for tol in tols:
    x = df3[df3["ray/tune/tol"] == tol]["ray/tune/step"]
    y = df3[df3["ray/tune/tol"] == tol]["ray/tune/cpu"]
    label = r"$tol = " + str(tol) + r"$"
    axs[1].plot(x, y, label=label)

# axs[1].set_xlabel(r"$Steps$")
axs[1].set_ylabel(r"$CPU$" + "\n" + r"$(\%)$")
# axs[1].set_title(r"MNIST")


# Average Step Time
for tol in tols:
    x = df3[df3["ray/tune/tol"] == tol]["ray/tune/step"]
    y = df3[df3["ray/tune/tol"] == tol]["ray/tune/avg_step_time"]
    label = r"$tol = " + str(tol) + r"$"
    axs[2].plot(x, y, label=label)

axs[2].set_xlabel(r"$Steps$")
axs[2].set_ylabel(r"$Mean Step Time$" + "\n" + r"$(ms)$")
# axs[1].set_title(r"MNIST")

# Rest
axs[0].legend()
# fig.tight_layout()

plt.savefig("../../../../thesis/assets/tol.png", dpi=300)

# %%
