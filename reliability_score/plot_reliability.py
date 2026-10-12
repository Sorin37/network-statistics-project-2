
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# --------------------------------------------------
# 1. Load stability results
# --------------------------------------------------

base_dir = Path(__file__).resolve().parent
csv_path = base_dir / "results" / "stability" / "top20_frequency.csv"

df = pd.read_csv(csv_path)

# Candidates appearing in the top 20 in all five runs
stable_links = df[df["runs_in_top20"] == 5].copy()

# Sort by mean reliability
stable_links = stable_links.sort_values("mean", ascending=True)

# Create readable link labels
stable_links["link"] = (
    stable_links["taxon_1"].astype(str)
    + " — "
    + stable_links["taxon_2"].astype(str)
)

# --------------------------------------------------
# 2. Create plot
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 5.5))

bar_color = "#8BAFB5"
error_color = "#333333"

ax.barh(
    stable_links["link"],
    stable_links["mean"],
    xerr=stable_links["std"],
    height=0.65,
    capsize=4,
    color=bar_color,
    edgecolor="none",
    error_kw={
        "ecolor": error_color,
        "elinewidth": 1.5,
        "capthick": 1.5
    }
)


import numpy  as np
#Title
ax.set_title(
    "Mean reliability of consistently top-ranked links",
    fontsize=14,
    fontweight="bold",
    pad=16
)

# X-axis scale from 0 to 1
ax.set_xlim(0, 1)
ax.set_xticks(np.arange(0, 1.01, 0.2))
ax.set_xlabel("Mean reliability score", fontsize=11)

# Make tick labels visible
ax.tick_params(
    axis="x",
    bottom=True,
    labelbottom=True,
    labelsize=10
)

# Gridlines
ax.grid(axis="x", linestyle="--", alpha=0.25)
ax.set_axisbelow(True)
# Legend
legend_elements = [
    Patch(
        facecolor=bar_color,
        label="Mean reliability"
    ),
    Line2D(
        [0], [0],
        color=error_color,
        marker="|",
        markersize=10,
        linewidth=1.5,
        label="±1 standard deviation"
    )
]

# Legend below the plot, avoiding overlap
ax.legend(
    handles=legend_elements,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.13),
    ncol=2,
    frameon=False,
    fontsize=10
)

plt.tight_layout()

# --------------------------------------------------
# 3. Save figure
# --------------------------------------------------

plots_dir = base_dir / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

output_path = plots_dir / "stability_top7.png"

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()

print("Plot saved to:", output_path)
print("Consistent candidates:", len(stable_links))
