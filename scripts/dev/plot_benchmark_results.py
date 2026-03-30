from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

benchmark_res_basedir = Path("gpu_benchmark_results_all/")
devices = ["rtx3080ti", "l40s", "h100"]
simplify_geom = False

# Merge all dataframes
dfs = []
for suffix in devices:
    dir_ = benchmark_res_basedir / f"gpu_benchmark_results_{suffix}"
    gpu_info_df = pd.read_csv(dir_ / "gpu_info.csv")
    gpu_name = gpu_info_df["name"][0]
    df = pd.read_csv(dir_ / "gpu_simulation_benchmark_results.csv")
    df["device"] = suffix
    df["device_fullname"] = gpu_name
    dfs.append(df)
df_all = pd.concat(dfs, ignore_index=True)

# Plot
fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
for device in devices:
    df = df_all[df_all["device"] == device]
    df = df[df["simplify_geom"] == simplify_geom]
    device_fullname = df["device_fullname"].iloc[0]
    ax.plot(df["n_worlds"], df["realtime_factor"], marker="o", label=device_fullname)

ax.set_xscale("log", base=2)

# Show every x value as a plain number
x_values = sorted(df_all["n_worlds"].unique())
ax.set_xticks(x_values)
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

# Add gridlines
ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

ax.set_xlabel("Number of parallel worlds (log scale)")
ax.set_ylabel("Throughput (× realtime)")
ax.set_title("GPU simulation scaling test")
ax.legend()

output_path = benchmark_res_basedir / f"scaling_test_simplifygeom{simplify_geom}.png"
fig.savefig(output_path, dpi=100)
