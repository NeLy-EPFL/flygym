import numpy as np
import pkg_resources
import pickle
import matplotlib.pyplot as plt

from pathlib import Path
from flygym.envs.nmf_mujoco import NeuroMechFlyMuJoCo
from flygym.util.config import all_leg_dofs, leg_dofs_3_per_leg

import numpy as np

# load PhysicsError
from dm_control.rl.control import PhysicsError

from hyperopt import hp
from hyperopt import STATUS_OK
from hyperopt import tpe
from hyperopt import fmin
from hyperopt import Trials
import argparse

from scipy.interpolate import griddata

import warnings

SIZE_PER_PARAM = 3


def run_simulation(nmf, num_steps, data_block):
    obs_list = []
    contact_forces = np.ones(
        (len(nmf.collision_tracked_geoms), num_steps)) * np.nan
    distance = 0

    try:
        for k in range(num_steps):
            joint_pos = data_block[:, k]
            action = {'joints': joint_pos}
            obs, info = nmf.step(action)
            obs_list.append(obs)

            contact_forces[:, k] = obs['contact_forces'].copy()
            # if not touching the floor for the last 20 steps, break
            if np.all(contact_forces[:, k - 20:k] <= 0) and k > 1000:
                return np.inf, np.inf, False

    except PhysicsError as pe:
        return np.inf, np.inf, False
    else:
        distance = -1 * (obs_list[0]["fly"][0][0] - obs_list[-1]["fly"][0][0])
        ang_diff = np.abs(obs_list[-1]["fly"][1] - obs_list[0]["fly"][1])
        return distance, ang_diff, True


def objective(all_params, ang_weight=0.5):
    # Keep track of evals
    global ITERATION

    ITERATION += 1

    # Initiatlize the nmf parameter with the right parameters
    if "friction" in all_params["physics_config"]:
        all_params["terrain_config"] = {}
        all_params["terrain_config"]["friction"] = all_params["physics_config"][
            "friction"]

    nmf = NeuroMechFlyMuJoCo(render_mode='headless',
                             timestep=1e-4,
                             render_config={'playspeed': 0.1, 'camera': 0},
                             init_pose='stretch',
                             actuated_joints=all_leg_dofs,
                             floor_collisions_geoms="all",
                             **all_params)

    # Interpolate 5x
    num_steps = int(run_time / nmf.timestep)
    data_block = np.zeros((len(nmf.actuated_joints), num_steps))
    measure_t = np.arange(len(data['joint_LFCoxa'])) * data['meta']['timestep']
    interp_t = np.arange(num_steps) * nmf.timestep
    for i, joint in enumerate(nmf.actuated_joints):
        data_block[i, :] = np.interp(interp_t, measure_t, data[joint])

    distance, ang_diff, finished_fine = run_simulation(nmf, num_steps, data_block)
    loss = 0
    if finished_fine:
        # distance is probably going to stay between 0 and 1, the angle should evolve between 0 and 1 but scaled to have a lower importance
        loss = -1 * (distance - np.mean(ang_diff / (np.pi / 2)) * ang_weight)
    else:
        loss = np.inf

    # Get the loss
    return {'loss': loss, 'params': all_params, 'iteration': ITERATION,
            'status': STATUS_OK, "distance": distance, "ang_diff": ang_diff}

# Global variable
global ITERATION

ITERATION = 0

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Script for optimizing nmf physics parameters.')

    # Define command line arguments
    parser.add_argument('--params', nargs='+', type=str,
                        help='List of parameters to optimize.',
                        default=['joint_stiffness', 'joint_damping',
                                 'actuator_kp'])
    parser.add_argument('--epochs', type=int,
                        help='Number of epochs to run the optimization.',
                        default=100)
    parser.add_argument('--run_time', type=float,
                        help='Duration of the simulation.', default=0.3)
    parser.add_argument('--output', type=str, help='Name of the output file.',
                        default='/Users/stimpfli/Desktop/flygym/scripts/paramsearch_results')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Print the results of the optimization.')

    # Parse the arguments
    args = parser.parse_args()

    # Get the values of the arguments
    params = args.params
    epochs = args.epochs
    run_time = args.run_time
    output_file = args.output
    verbose = args.verbose

    return params, epochs, run_time, output_file, verbose

def visualize_log(output_file, params_names, params_values, accs,
                  extr, size_per_param=SIZE_PER_PARAM, n_pts=500,
                  dpi=500):
    fig_log, axs_log = plt.subplots(len(params_names), len(params_names) + 1,
                                    figsize=(
                                        (
                                                    len(params_names) + 1) * size_per_param,
                                        len(params_names) * size_per_param),
                                    dpi=dpi,
                                    width_ratios=[1] * len(params_names) + [
                                        0.25])

    nan_ids = np.isnan(accs)
    order = np.arange(len(accs))

    for i, param1 in enumerate(params_names):
        for j, param2 in enumerate(params_names):
            if i == j:
                axs_log[i, j].scatter(params_values[i], accs,
                                      c=order, cmap="Greys",
                                      alpha=0.5, s=5)

                # show a axvline for nan values
                for x in params_values[i][nan_ids]:
                    axs_log[i, j].axvline(x, c='r', linestyle=':',
                                          alpha=0.5, lw=1)

                axs_log[i, j].set_box_aspect(1)
                axs_log[i, j].set_xscale('log')

            else:
                min1, max1 = extr[param1]
                min2, max2 = extr[param2]
                xi_log = np.geomspace(min2, max2, n_pts)
                yi_log = np.geomspace(min1, max1, n_pts)
                X_log, Y_log = np.meshgrid(xi_log, yi_log)

                zi_log = griddata((params_values[j],
                                   params_values[i]),
                                  accs,
                                  (X_log, Y_log),
                                  method='nearest',
                                  fill_value=np.nan)

                a_log = axs_log[i, j].pcolor(X_log, Y_log, zi_log,
                                             cmap="viridis")

                # plot colormap crosses for non nan and red cross for nan
                axs_log[i, j].scatter(params_values[j][~nan_ids],
                                      params_values[i][~nan_ids],
                                      marker='+', c=order[~nan_ids],
                                      cmap="Greys", alpha=0.5, s=5)
                axs_log[i, j].scatter(params_values[j][nan_ids],
                                      params_values[i][nan_ids],
                                      marker='+', c='r', alpha=0.5, s=5)

                axs_log[i, j].set_box_aspect(1)
                axs_log[i, j].set_xlim(min2, max2)
                axs_log[i, j].set_ylim(min1, max1)

                axs_log[i, j].set_xscale('log')
                axs_log[i, j].set_yscale('log')
                axs_log[i, j].set_aspect("auto")

            if i == len(params_names) - 1:
                axs_log[i, j].set_xlabel(param2)
            if j == 0:
                axs_log[i, j].set_ylabel(param1)

    cbar_ax_log = fig_log.add_axes([0.85, 0.15, 0.05, 0.7])
    cb_log = fig_log.colorbar(a_log, cax=cbar_ax_log, extend='max')
    cb_log.set_label('loss')

    fig_log.suptitle(f'Optimization landscape for {params_names}')

    for i in range(len(params_names)):
        fig_log.delaxes(axs_log[i, len(params_names)])

    fig_log.savefig(output_file)

    return 0


def visualize_lin(output_file, params_names, params_values, accs,
                  extr, size_per_param=SIZE_PER_PARAM, n_pts=500,
                  dpi=500):

    fig_lin, axs_lin = plt.subplots(len(params_names), len(params_names) + 1,
                                    figsize=((len(params_names) + 1) * size_per_param,
                                             len(params_names) * size_per_param),
                                    dpi=dpi,
                                    width_ratios=[1] * len(params_names) + [0.25])

    nan_ids = np.isnan(accs)
    order = np.arange(len(accs))

    for i, param1 in enumerate(params_names):
        for j, param2 in enumerate(params_names):
            if i == j:
                axs_lin[i, j].scatter(params_values[i], accs,
                                      c=order, cmap="Greys",
                                      alpha=0.5, s=5)
                # show a axvline for nan values
                for x in params_values[i][nan_ids]:
                    axs_lin[i, j].axvline(x, c='r', linestyle=':',
                                          alpha=0.5, lw=1)
                axs_lin[i, j].set_box_aspect(1)

            else:
                min1, max1 = extr[param1]
                min2, max2 = extr[param2]
                xi = np.linspace(min2, max2, n_pts)
                yi = np.linspace(min1, max1, n_pts)
                X, Y = np.meshgrid(xi, yi)

                zi = griddata((params_values[j],
                               params_values[i]),
                              accs,
                              (X, Y),
                              method='nearest',
                              fill_value=np.nan)

                a_lin = axs_lin[i, j].pcolor(X, Y, zi, cmap="viridis")

                # plot colormap crosses for non nan and red cross for nan
                axs_lin[i, j].scatter(params_values[j][~nan_ids],
                                      params_values[i][~nan_ids],
                                      marker='+', c=order[~nan_ids],
                                      cmap="Greys", alpha=0.5, s=5)
                axs_lin[i, j].scatter(params_values[j][nan_ids],
                                      params_values[i][nan_ids],
                                      marker='+', c='r', alpha=0.5, s=5)

                axs_lin[i, j].set_box_aspect(1)

                axs_lin[i, j].set_xlim(min2, max2)
                axs_lin[i, j].set_ylim(min1, max1)

                # axs[i, j].set_aspect(np.log10(max2 - min2)/np.log10(max1 - min1))
                axs_lin[i, j].set_aspect("auto")

            # axs[i, j].set_box_aspect(1)®

            if i == len(params_names) - 1:
                axs_lin[i, j].set_xlabel(param2)
            if j == 0:
                axs_lin[i, j].set_ylabel(param1)

    cbar_ax_lin = fig_lin.add_axes([0.85, 0.15, 0.05, 0.7])
    cb_lin = fig_lin.colorbar(a_lin, cax=cbar_ax_lin, extend='max')
    cb_lin.set_label('loss')

    fig_lin.suptitle(f'Optimization landscape for {params_names}')

    for i in range(len(params_names)):
        fig_lin.delaxes(axs_lin[i, len(params_names)])

    fig_lin.savefig(output_file)

    return 0

def save_best_run(vid_dir, best_params, run_time):

    nmf = NeuroMechFlyMuJoCo(render_mode='saved',
                             timestep=1e-4,
                             render_config={'playspeed': 0.1, 'camera': 0},
                             init_pose='stretch',
                             actuated_joints=all_leg_dofs,
                             floor_collisions_geoms="all",
                             **best_params)
    # Interpolate 5x
    num_steps = int(run_time / nmf.timestep)
    data_block = np.zeros((len(nmf.actuated_joints), num_steps))
    measure_t = np.arange(len(data['joint_LFCoxa'])) * data['meta']['timestep']
    interp_t = np.arange(num_steps) * nmf.timestep
    for i, joint in enumerate(nmf.actuated_joints):
        data_block[i, :] = np.interp(interp_t, measure_t, data[joint])

    num_steps = int(run_time / nmf.timestep)
    obs_list = []

    for k in range(num_steps):
        joint_pos = data_block[:, k]
        action = {'joints': joint_pos}
        obs, info = nmf.step(action)
        obs_list.append(obs)
        nmf.render()

    nmf.save_video(vid_dir)


# use uniform for uniform distriution
space = {'physics_config': {
    "joint_stiffness": hp.loguniform("joint_stiffness", np.log(1e-3),
                                     np.log(10)),
    "joint_damping": hp.loguniform("joint_damping", np.log(1e-3), np.log(10)),
    "actuator_kp": hp.loguniform("actuator_kp", np.log(1e-3), np.log(70)),

    "friction": (hp.loguniform("sliding_friction", np.log(1e-4), np.log(10)),
                 hp.loguniform("torsional_friction", np.log(1e-4), np.log(10)),
                 hp.loguniform("rolling_friction_", np.log(1e-4), np.log(10)))},

    "actuated_joints": hp.choice("actuated_joints",
                                 [all_leg_dofs, leg_dofs_3_per_leg])}

if __name__ == '__main__':
    params, n_epochs, run_time, out_dir, verbose = parse_arguments()
    if not verbose:
        warnings.filterwarnings("ignore")

    out_dir = Path(out_dir)

    # Remove parameters that are not to be optimized for the space
    params_to_remove = []
    for subspace, subitems in space.items():
        if subspace not in params:
            if isinstance(subitems, dict):
                for subsubspace in subitems.keys():
                    if subsubspace not in params:
                        params_to_remove.append([subspace, subsubspace])
            else:
                params_to_remove.append([subspace])

    for param in params_to_remove:
        if len(param) == 2:
            del space[param[0]][param[1]]
        else:
            del space[param[0]]

    # Load recorded data
    data_path = Path(pkg_resources.resource_filename('flygym', 'data'))
    with open(data_path / 'behavior' / '210902_pr_fly1.pkl', 'rb') as f:
        data = pickle.load(f)

    # Keep track of results
    nmf_trials = Trials()
    # Run optimization
    best_nmf = fmin(fn=objective, space=space, algo=tpe.suggest,
                    max_evals=n_epochs, trials=nmf_trials)

    print("Best parameters found:", best_nmf)

    # Save the results
    res_dir = out_dir / f'nmf_trials_{"_".join(params)}_{n_epochs}points.pkl'
    res_dir.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(nmf_trials, open(res_dir, 'wb'))

    params_values = np.array(
        [[t["misc"]["vals"][p][0] for t in nmf_trials.trials] for p
         in params]
    )
    accs = np.array([t['result']['loss'] for t in nmf_trials.trials])

    # replace inf by nans (inf is better for the algorithm but not for the
    # visualization)
    accs[accs == np.inf] = np.nan

    # Visualize the results
    # for each pair of param interpolate the 2D space losses
    # indicate with a cross all real values
    extremums = {}
    for i, param in enumerate(params):
        extremums[param] = [np.min(params_values[i]), np.max(params_values[i])]

    visualize_log(
        out_dir / f'nmf_trials_{"_".join(params)}_{n_epochs}points_log.png',
        params, params_values, accs, extremums)

    visualize_lin(
        out_dir / f'nmf_trials_{"_".join(params)}_{n_epochs}points_lin.png',
        params, params_values, accs, extremums)

    # Get video of the best trial

    save_best_run(out_dir /
                  f'nmf_trials_{"_".join(params)}_{n_epochs}points_best.mp4',
                  nmf_trials.trials[np.nanargmin(accs)]["result"]["params"],
                  run_time)
