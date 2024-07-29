import pickle
import numpy as np
import pandas as pd
import gc
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm, trange
from flygym.examples.path_integration import model, util, viz


plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42

base_dir = Path("./outputs/path_integration/")
models_dir = base_dir / "models"
figs_dir = base_dir / "figs"
models_dir.mkdir(exist_ok=True, parents=True)
figs_dir.mkdir(exist_ok=True, parents=True)

gaits = ["tripod", "tetrapod", "wave"]
num_trials_per_gait = 15
training_trials = list(range(10))
testing_trials = list(range(10, 15))


# Load random exploration data
trial_data = {}
for gait in gaits:
    for seed in trange(num_trials_per_gait, desc=f"Loading {gait} gait trials"):
        trial_dir = base_dir / f"random_exploration/seed={seed}_gait={gait}"
        trial_data[(gait, seed)] = util.load_trial_data(trial_dir)


# Plot example exploration trials
viz.plot_example_trials(
    gaits, num_trials_per_gait, trial_data, output_path=figs_dir / "example_trials.pdf"
)


# Visualize contact forces for each pair of legs
seed = 0
force_thresholds = (0.5, 1, 3)
viz.plot_contact_forces(
    trial_data, force_thresholds, output_path=figs_dir / "contact_forces.pdf"
)


def fit_1d_linear_model(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    model = LinearRegression()
    model.fit(x, y)
    r2 = model.score(x, y)
    return model.coef_, model.intercept_, r2


def model_info_to_dict(
    model_info: tuple[float, float, float], model_name: str, legs: str
) -> dict[str, float]:
    coefs, intercept, r2 = model_info
    leg_mask = util.get_leg_mask(legs)
    coefs_all = np.full(3, np.nan)
    coefs_all[leg_mask] = coefs
    return {
        f"k_fore_{model_name}": coefs_all[0],
        f"k_mid_{model_name}": coefs_all[1],
        f"k_hind_{model_name}": coefs_all[2],
        f"b_{model_name}": intercept,
        f"r2_{model_name}": r2,
    }


def fit_models(
    trial_data: dict[str, np.ndarray],
    time_scale: float,
    contact_force_thr: tuple[float, float, float],
    legs: str,
    dt: float = 1e-4,
):
    variables = util.extract_variables(
        trial_data, time_scale, contact_force_thr, legs, dt
    )
    model_info = {}

    model_prop2heading = fit_1d_linear_model(
        variables["stride_total_diff_lrdiff"], variables["heading_diff"]
    )
    model_info.update(model_info_to_dict(model_prop2heading, "prop2heading", legs))

    model_prop2disp = fit_1d_linear_model(
        variables["stride_total_diff_lrsum"], variables["forward_disp_total_diff"]
    )
    model_info.update(model_info_to_dict(model_prop2disp, "prop2disp", legs))

    model_dn2heading = fit_1d_linear_model(
        variables["diff_dn_drive"], variables["heading_diff"]
    )
    model_info.update(model_info_to_dict(model_dn2heading, "dn2heading", legs))

    model_dn2disp = fit_1d_linear_model(
        variables["sum_dn_drive"], variables["forward_disp_total_diff"]
    )
    model_info.update(model_info_to_dict(model_dn2disp, "dn2disp", legs))

    return model_info


# Let's inspect what it looks like roughly
variables = util.extract_variables(
    trial_data=trial_data[("tripod", 0)],
    time_scale=0.64,
    contact_force_thr=(0.5, 1, 3),
    legs="FMH",
)

# scale some variables just so they look visually aligned - this is a quick & dirty plot
fig, ax = plt.subplots()
t_grid = np.arange(variables["heading_diff"].shape[0]) * 1e-4 + 0.64
ax.axhline(0, color="black", lw=1)
ax.plot(
    t_grid, variables["stride_total_diff_lrdiff"][:, 0] * 0.4, label="Fore legs prop."
)
ax.plot(
    t_grid, variables["stride_total_diff_lrdiff"][:, 1] * 0.2, label="Mid legs prop."
)
ax.plot(
    t_grid, variables["stride_total_diff_lrdiff"][:, 2] * 0.6, label="Hind legs prop."
)
ax.plot(t_grid, variables["diff_dn_drive"] * -0.8, label="Descending")
ax.plot(t_grid, variables["heading_diff"] * 1, color="black", label="Heading change")
ax.set_xlim(6, 10)
ax.set_yticks([])
ax.set_ylim(-1.3, 1.3)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Signal [AU]")
ax.legend(ncol=2)
sns.despine(ax=ax)
fig.savefig(figs_dir / "input_variables.pdf")

fit_models(
    trial_data=trial_data[("tripod", 0)],
    time_scale=0.64,
    contact_force_thr=(0.5, 1, 3),
    legs="FMH",
)


# First sensitivity analysis: Find optimal contact force threshold
trial_variables = {}
time_scales = [0.64]
contact_force_thresholds = [0.1, 0.5, 1, 1.5, 2, 2.5, 3.5, 5, 7]
configs = []
for gait in gaits:
    for seed in training_trials:
        for time_scale in time_scales:
            for thr in contact_force_thresholds:
                for leg in ["F", "M", "H"]:
                    contact_force_thr_all = [np.nan, np.nan, np.nan]
                    contact_force_thr_all["FMH".index(leg)] = thr
                    config = (gait, seed, time_scale, contact_force_thr_all, leg)
                    configs.append(config)


def wrapper(config):
    gait, seed, time_scale, contact_force_thr, legs = config
    model_info = fit_models(
        trial_data[(gait, seed)], time_scale, contact_force_thr, legs
    )
    model_info["gait"] = gait
    model_info["seed"] = seed
    model_info["time_scale"] = time_scale
    model_info["contact_force_thr_fore"] = contact_force_thr[0]
    model_info["contact_force_thr_mid"] = contact_force_thr[1]
    model_info["contact_force_thr_hind"] = contact_force_thr[2]
    model_info["legs"] = legs
    return model_info


results = Parallel(n_jobs=-2)(delayed(wrapper)(config) for config in tqdm(configs))

model_df1 = pd.DataFrame(results)
model_df1 = model_df1.set_index(
    [
        "gait",
        "seed",
        "time_scale",
        "contact_force_thr_fore",
        "contact_force_thr_mid",
        "contact_force_thr_hind",
        "legs",
    ]
)
model_df1.to_pickle(models_dir / "model_df_sensitivity_analysis_1.pkl")


# Let's check the result of this rough sensitivity analysis
viz.plot_contact_force_thr_sensitivity_analysis(
    model_df1, time_scale=0.64, output_path=figs_dir / "sensitivity_analysis_1.pdf"
)


# Second sensitivity analysis: Find the optima ltime scale
trial_variables = {}
time_scales = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12]
# legs_combi = ["F", "M", "H", "FM", "FH", "MH", "FMH"]
configs = list(
    itertools.product(gaits, training_trials, time_scales, [(0.5, 1, 3)], ["FMH"])
)


def wrapper(config):
    gait, seed, time_scale, contact_force_thr, legs = config
    model_info = fit_models(
        trial_data[(gait, seed)], time_scale, contact_force_thr, legs
    )
    model_info["gait"] = gait
    model_info["seed"] = seed
    model_info["time_scale"] = time_scale
    model_info["contact_force_thr_fore"] = contact_force_thr[0]
    model_info["contact_force_thr_mid"] = contact_force_thr[1]
    model_info["contact_force_thr_hind"] = contact_force_thr[2]
    model_info["legs"] = legs
    return model_info


results = Parallel(n_jobs=-2)(delayed(wrapper)(config) for config in tqdm(configs))

model_df2 = pd.DataFrame(results)
model_df2 = model_df2.set_index(
    [
        "gait",
        "seed",
        "time_scale",
        "contact_force_thr_fore",
        "contact_force_thr_mid",
        "contact_force_thr_hind",
        "legs",
    ]
)
model_df2.to_pickle(models_dir / "model_df_sensitivity_analysis_2.pkl")
viz.plot_time_scale_sensitivity_analysis(
    model_df2, gaits, time_scales, output_path=figs_dir / "sensitivity_analysis_2.pdf"
)


# Third sensitivity analysis: Evaluate contribution of each leg
trial_variables = {}
legs_combi = ["F", "M", "H", "FM", "FH", "MH", "FMH"]
configs = list(
    itertools.product(gaits, training_trials, [0.64], [(0.5, 1, 3)], legs_combi)
)


def wrapper(config):
    gait, seed, time_scale, contact_force_thr, legs = config
    model_info = fit_models(
        trial_data[(gait, seed)], time_scale, contact_force_thr, legs
    )
    model_info["gait"] = gait
    model_info["seed"] = seed
    model_info["time_scale"] = time_scale
    model_info["contact_force_thr_fore"] = contact_force_thr[0]
    model_info["contact_force_thr_mid"] = contact_force_thr[1]
    model_info["contact_force_thr_hind"] = contact_force_thr[2]
    model_info["legs"] = legs
    return model_info


results = Parallel(n_jobs=-2)(delayed(wrapper)(config) for config in tqdm(configs))

model_df3 = pd.DataFrame(results)
model_df3 = model_df3.set_index(
    [
        "gait",
        "seed",
        "time_scale",
        "contact_force_thr_fore",
        "contact_force_thr_mid",
        "contact_force_thr_hind",
        "legs",
    ]
)
model_df3.to_pickle(models_dir / "model_df_sensitivity_analysis_3.pkl")
viz.leg_combination_sensitivity_analysis(
    model_df3, gaits, output_path=figs_dir / "sensitivity_analysis_3.pdf"
)


# Demonstrate final model
# It looks like the best model parameters are the following:
# - Contact force threshold: (0.5, 1, 3) mN
# - Time scale: 0.64s
# - Legs: forelegs and midlegs
#
# Let's use this model to perform path integration. We will do it on the 5
# trials that we did *not* use to train the models.
#
# First, let's build an ensemble model of the 5 trained models. Since the
# models are linear, we can simply average the weights and biases.

ensemble_models_df = model_df3.groupby(
    [
        "gait",
        "time_scale",
        "contact_force_thr_fore",
        "contact_force_thr_mid",
        "contact_force_thr_hind",
        "legs",
    ]
).mean()
ensemble_models_df.to_pickle(models_dir / "ensemble_models_df.pkl")

legs = "FMH"
contact_force_thr = (0.5, 1, 3)
time_scale = 0.64
path_integration_results = {}
for col, gait in enumerate(gaits):
    for row, seed in enumerate(tqdm(testing_trials, desc=f"Processing {gait} gait")):
        # Select model from ensembled model dataframe and construct intermediate models
        model_info = ensemble_models_df.loc[gait, time_scale, *contact_force_thr, legs]
        prop2heading_coefs = model_info[
            [f"k_{leg}_prop2heading" for leg in ["fore", "mid", "hind"]]
        ].values
        heading_model = model.LinearModel(
            prop2heading_coefs, model_info["b_prop2heading"], legs
        )
        prop2disp_coefs = model_info[
            [f"k_{leg}_prop2disp" for leg in ["fore", "mid", "hind"]]
        ].values
        disp_model = model.LinearModel(prop2disp_coefs, model_info["b_prop2disp"], legs)

        # Perform path integration
        path_integration_results[(gait, seed)] = model.path_integrate(
            trial_data[(gait, seed)],
            heading_model,
            disp_model,
            time_scale=time_scale,
            contact_force_thr=contact_force_thr,
            legs=legs,
            dt=1e-4,
        )

viz.plot_all_path_integration_trials(
    path_integration_results,
    trial_data,
    gaits,
    testing_trials,
    output_path=figs_dir / "all_path_integration_trials.pdf",
)

viz.make_model_prediction_scatter_plot(
    path_integration_results, figs_dir / "scatter.pdf"
)
