"""Bar chart of RMSE by model and split. Run after benchmark.py."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

results = pd.read_csv("results/results.csv")
summary = results.groupby(["model", "split"])["rmse"].agg(["mean", "std"]).reset_index()

models = ["predict_mean", "lightgbm_fingerprints"]
splits = ["random", "scaffold"]
colors = {"random": "#7f8c8d", "scaffold": "#c0392b"}

fig, ax = plt.subplots(figsize=(6, 4))
width = 0.35
for j, split in enumerate(splits):
    sub = summary[summary["split"] == split].set_index("model").loc[models]
    xs = [i + (j - 0.5) * width for i in range(len(models))]
    ax.bar(xs, sub["mean"], width, yerr=sub["std"], capsize=4,
           label=f"{split} split", color=colors[split])
ax.set_xticks(range(len(models)))
ax.set_xticklabels(["predict the mean\n(baseline)", "LightGBM on\nMorgan fingerprints"])
ax.set_ylabel("RMSE (log solubility), lower is better")
ax.set_title("ESOL solubility: random split flatters the model")
ax.legend()
fig.tight_layout()
fig.savefig("results/rmse_by_split.png", dpi=150)
print("Saved results/rmse_by_split.png")
