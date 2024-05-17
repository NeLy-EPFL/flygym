import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from tqdm import tqdm


plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42

_metric_configs = {
    "r2_prop2heading": ("Δheading estimator", "tab:green"),
    "r2_prop2disp": ("Δforward displacement estimator", "tab:purple"),
}


def plot_example_trials(gaits, num_trials_per_gait, trial_data, output_path):
    fig, axs = plt.subplots(1, 3, figsize=(9, 3), tight_layout=True)
    for gait, ax in zip(gaits, axs):
        for seed in range(num_trials_per_gait):
            fly_pos = trial_data[(gait, seed)]["fly_pos"]
            ax.plot(fly_pos[:, 0], fly_pos[:, 1], alpha=1)
        ax.plot([0], [0], "o", color="black")
        ax.set_title(f"{gait.title()} gait")
        ax.set_xlim(-150, 150)
        ax.set_ylim(-150, 150)
        ax.set_aspect("equal")
    fig.savefig(output_path)


def plot_contact_forces(trial_data, force_thresholds, output_path):
    fig, axs = plt.subplots(3, 1, figsize=(6, 6), tight_layout=True, sharex=True)
    t_grid = np.arange(trial_data[("tripod", 0)]["contact_force"].shape[0]) * 1e-4
    for i, pos in enumerate(["fore", "mid", "hind"]):
        ax = axs[i]
        ax.plot(t_grid, trial_data[("tripod", 0)]["contact_force"][:, i], lw=1)
        ax.plot(t_grid, trial_data[("tripod", 0)]["contact_force"][:, 3 + i], lw=1)
        ax.axhline(force_thresholds[i], color="black", lw=1)
        ax.set_xlim(10, 10.5)
        ax.set_ylim(0, 20)
        ax.set_ylabel("Contact force [mN]")
        ax.set_title(f"{pos.title()} legs")
        if i == 2:
            ax.set_xlabel("Time [s]")
        sns.despine(ax=ax)
    fig.savefig(output_path)


def plot_contact_force_thr_sensitivity_analysis(model_info_df, time_scale, output_path):
    fig, axs = plt.subplots(2, 3, figsize=(10, 5), tight_layout=True)
    for i, target in enumerate(["heading", "disp"]):
        for j, gait in enumerate(["tripod", "tetrapod", "wave"]):
            ax = axs[i, j]
            for leg_code, leg in {"F": "fore", "M": "mid", "H": "hind"}.items():
                model_df1_sub = model_info_df.loc[
                    gait, :, time_scale, :, :, :, leg_code
                ]
                thr_to_r2 = model_df1_sub.groupby(f"contact_force_thr_{leg}")[
                    f"r2_prop2{target}"
                ].mean()
                ax.plot(thr_to_r2, label=f"{leg} legs", marker="o", markersize=3)
            ax.set_ylim(0, 1)
            target_str = {"heading": "heading", "disp": "displacement"}[target]
            ax.set_title(f"{gait.title()} gait, {target_str} prediction")
            if i == 1 and j == 2:
                ax.legend(frameon=False, loc="lower right")
            if i == 1:
                ax.set_xlabel("Contact force threshold [mN]")
            if j == 0:
                ax.set_ylabel("$R^2$")
            sns.despine(ax=ax)
    fig.savefig(output_path)


def plot_time_scale_sensitivity_analysis(
    model_info_df, gaits, time_scales, output_path
):
    fig, axs = plt.subplots(2, 3, figsize=(12, 6), tight_layout=True)
    for col, gait in enumerate(gaits):
        for row, (metric, (title, color)) in enumerate(_metric_configs.items()):
            ax = axs[row, col]
            mean_r2s = []
            for i, time_scale in enumerate(time_scales):
                sel = model_info_df.loc[gait, :, time_scale]
                ax.plot([i] * len(sel), sel[metric], ".", color=color)
                mean_r2s.append(sel[metric].mean())
            ax.plot(range(len(time_scales)), mean_r2s, label=title, color=color)
            ax.set_xticks(range(len(time_scales)))
            ax.set_xticklabels(time_scales)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Time scale [s]")
            ax.set_ylabel(f"$R^2$")
            ax.set_title(f"{gait.title()} gait\n{title}")
            sns.despine(ax=ax)
    fig.savefig(output_path)


def leg_combination_sensitivity_analysis(model_info_df, gaits, output_path):
    fig, axs = plt.subplots(2, 3, figsize=(12, 6), tight_layout=True)
    for col, gait in enumerate(gaits):
        for row, (metric, (title, color)) in enumerate(_metric_configs.items()):
            ax = axs[row, col]
            sns.boxplot(
                data=model_info_df.loc[gait].reset_index(),
                x="legs",
                y=metric,
                color=color,
                boxprops=dict(facecolor="none"),
                ax=ax,
            )
            sns.stripplot(
                data=model_info_df.loc[gait],
                x="legs",
                y=metric,
                color=color,
                ax=ax,
            )
            ax.set_ylim(0, 1)
            ax.set_xlabel("Legs used")
            ax.set_ylabel(f"$R^2$")
            ax.set_title(f"{gait.title()} gait\n{title}")
            sns.despine(ax=ax)
    fig.savefig(output_path)


def plot_all_path_integration_trials(
    path_integration_results, trial_data, gaits, seeds, output_path, sample_rate=0.001
):
    sample_interval = int(1 / sample_rate)
    fig, axs = plt.subplots(
        len(seeds), 3, figsize=(9, 3 * len(seeds)), tight_layout=True
    )
    for col, gait in enumerate(gaits):
        for row, seed in enumerate(tqdm(seeds, desc=f"Plotting {gait} gait")):
            trial_res = path_integration_results[(gait, seed)]
            pos_real = trial_data[(gait, seed)]["fly_pos"]
            pos_pred = trial_res["pos_pred"]

            ax = axs[row, col]
            ax.plot(
                pos_real[::sample_interval, 0],
                pos_real[::sample_interval, 1],
                color="black",
                label="Actual",
            )
            ax.plot(
                pos_pred[::sample_interval, 0],
                pos_pred[::sample_interval, 1],
                color="tab:red",
                label="Estimated",
            )
            ax.plot([0], [0], "o", color="black")
            ax.set_title(f"{gait.title()} gait, trial {seed}")
            # ax.set_xlim(-150, 150)
            # ax.set_ylim(-150, 150)
            ax.axis("square")
            ax.set_aspect("equal")
            ax.legend()
    fig.savefig(output_path)


def make_model_prediction_scatter_plot(
    path_integration_results, output_path, sample_rate=0.001
):
    sample_interval = int(1 / sample_rate)
    heading_diff_pred_all = []
    heading_diff_real_all = []
    disp_diff_pred_all = []
    disp_diff_real_all = []
    for res in path_integration_results.values():
        heading_diff_pred_all.append(res["heading_diff_pred"])
        heading_diff_real_all.append(res["heading_diff_actual"])
        disp_diff_pred_all.append(res["displacement_diff_pred"])
        disp_diff_real_all.append(res["displacement_diff_actual"])
    heading_diff_pred_all = np.concatenate(heading_diff_pred_all)
    heading_diff_real_all = np.concatenate(heading_diff_real_all)
    disp_diff_pred_all = np.concatenate(disp_diff_pred_all)
    disp_diff_real_all = np.concatenate(disp_diff_real_all)

    fig, axs = plt.subplots(1, 2, figsize=(5, 2.5), tight_layout=True)

    axs[0].axhline(0, color="black", lw=1)
    axs[0].axvline(0, color="black", lw=1)
    axs[0].scatter(
        np.rad2deg(heading_diff_real_all[::sample_interval]),
        np.rad2deg(heading_diff_pred_all[::sample_interval]),
        s=0.1,
        color=_metric_configs["r2_prop2heading"][1],
        alpha=0.5,
    )
    axs[0].plot([-180, 180], [-180, 180], "--", color="black", lw=1, zorder=1e9)
    r2 = r2_score(
        heading_diff_real_all[np.isfinite(heading_diff_real_all)],
        heading_diff_pred_all[np.isfinite(heading_diff_real_all)],
    )
    axs[0].text(0.1, 0.9, f"$R^2$={r2:.2f}", transform=axs[0].transAxes)
    axs[0].set_xlabel(r"Actual Δheading [$^\circ$]")
    axs[0].set_ylabel(r"Predicted Δheading [$^\circ$]")
    axs[0].set_xlim(-130, 130)
    axs[0].set_ylim(-130, 130)
    axs[0].set_aspect("equal")
    sns.despine(ax=axs[0], bottom=True, left=True)

    axs[1].scatter(
        disp_diff_real_all[::sample_interval],
        disp_diff_pred_all[::sample_interval],
        s=1,
        color=_metric_configs["r2_prop2disp"][1],
        alpha=0.5,
    )
    axs[1].plot([0, 20], [0, 20], "--", color="black", lw=1, zorder=1e9)
    r2 = r2_score(
        disp_diff_real_all[np.isfinite(disp_diff_real_all)],
        disp_diff_pred_all[np.isfinite(disp_diff_real_all)],
    )
    axs[1].text(0.1, 0.9, f"$R^2$={r2:.2f}", transform=axs[1].transAxes)
    axs[1].set_xlabel(r"Actual Δdisplacement [mm]")
    axs[1].set_ylabel(r"Predicted Δdisplacement [mm]")
    axs[1].set_xlim(2, 12)
    axs[1].set_ylim(2, 12)
    axs[1].set_aspect("equal")
    sns.despine(ax=axs[1])

    fig.savefig(output_path)
