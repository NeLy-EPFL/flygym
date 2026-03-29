import textwrap
from typing import Literal

from tabulate import tabulate

__all__ = ["print_perf_report"]


def print_perf_report(
    total_physics_time_ns: int,
    total_render_time_ns: int,
    n_steps: int,
    n_frames_rendered: int,
    timestep: float,
    show_in_notebook: bool | Literal["auto"] = "auto",
) -> None:
    """Print a single-world performance report.

    Args:
        total_physics_time_ns: Total wall-clock time spent in physics steps, in ns.
        total_render_time_ns: Total wall-clock time spent rendering, in ns.
        n_steps: Number of physics steps taken.
        n_frames_rendered: Number of frames rendered.
        timestep: Simulation timestep in seconds.
        show_in_notebook: Whether the report should be displayed as an HTML element
            for better display in a Jupyter notebook. If set to "auto", the function
            will attempt to detect if it's running in a notebook environment.
    """
    if show_in_notebook == "auto":
        show_in_notebook = check_environment() == "notebook"

    if n_steps == 0:
        raise ValueError("n_steps must be > 0 to print performance report.")

    total_walltime_ns = total_physics_time_ns + total_render_time_ns
    total_per_iter_us = 1e-3 * total_walltime_ns / n_steps
    physics_time_per_iter_us = 1e-3 * total_physics_time_ns / n_steps
    physics_percent = 100 * total_physics_time_ns / total_walltime_ns
    physics_throughput = 1e9 * n_steps / total_physics_time_ns
    total_throughput = 1e9 * n_steps / total_walltime_ns
    physics_realtime_x = physics_throughput * timestep
    total_realtime_x = total_throughput * timestep

    if n_frames_rendered == 0:
        render_time_per_iter_us = float("nan")
        render_time_per_frame_us = float("nan")
        render_percent = float("nan")
        render_throughput = float("nan")
        render_realtime_x = float("nan")
    else:
        render_time_per_iter_us = 1e-3 * total_render_time_ns / n_steps
        render_time_per_frame_us = 1e-3 * total_render_time_ns / n_frames_rendered
        render_percent = 100 * total_render_time_ns / total_walltime_ns
        render_throughput = 1e9 * n_steps / total_render_time_ns
        render_realtime_x = render_throughput * timestep

    table = [
        [
            "Physics simulation advancement",
            physics_time_per_iter_us,
            physics_percent,
            physics_throughput,
            physics_realtime_x,
        ],
        [
            "Rendering*",
            render_time_per_iter_us,
            render_percent,
            render_throughput,
            render_realtime_x,
        ],
        [
            "TOTAL",
            total_per_iter_us,
            100,
            total_throughput,
            total_realtime_x,
        ],
    ]

    if n_frames_rendered == 0:
        rendering_note = "* Note: No frames were rendered."
    else:
        rendering_note = (
            f"* Note: {n_frames_rendered} frames were rendered out of {n_steps} steps. "
            f"Therefore, rendering time per image is {render_time_per_frame_us:.0f} us."
        )

    tab_str = tabulate(
        table,
        headers=[
            "\nStage",
            "Time/step\n(us)",
            "Percent\n(%)",
            "Throughput\n(iters/s)",
            "Throughput\nx realtime",
        ],
        floatfmt=("s", ".0f", ".0f", ".0f", ".2f"),
        tablefmt="html" if show_in_notebook else "simple_grid",
    )

    if show_in_notebook:
        from IPython.display import HTML, display

        print("PERFORMANCE PROFILE")
        display(HTML(tab_str))
        print(rendering_note)
    else:
        tab_width = max(len(line) for line in tab_str.splitlines())
        print()
        print("PERFORMANCE PROFILE".center(tab_width))
        print(tab_str)
        # Print the rendering note wrapped to the table width for better display
        print(textwrap.fill(rendering_note, width=tab_width))
        print()


def print_perf_report_parallel(
    total_physics_time_ns: int,
    total_render_time_ns: int,
    n_steps: int,
    n_frames_rendered: int,
    timestep: float,
    n_worlds: int,
    n_worlds_rendered: int,
    show_in_notebook: bool | Literal["auto"] = "auto",
) -> None:
    """Print a multi-world performance report including parallelized throughput.

    Args:
        total_physics_time_ns: Total wall-clock time spent in physics steps, in ns.
        total_render_time_ns: Total wall-clock time spent rendering, in ns.
        n_steps: Number of physics steps taken.
        n_frames_rendered: Number of frames rendered.
        timestep: Simulation timestep in seconds.
        n_worlds: Total number of parallel worlds.
        n_worlds_rendered: Number of worlds that were rendered.
        show_in_notebook: Whether the report should be displayed as an HTML element
            for better display in a Jupyter notebook. If set to "auto", the function
            will attempt to detect if it's running in a notebook environment.
    """
    if show_in_notebook == "auto":
        show_in_notebook = check_environment() == "notebook"

    if n_steps == 0:
        raise ValueError(
            "n_steps must be > 0 to print performance report. "
            "Hint: Did you place `sim.step()` inside a captured graph? If so, "
            "If so, profiling cannot be meaningfully done due to GPU-CPU synch "
            "constraints."
        )

    total_walltime_ns = total_physics_time_ns + total_render_time_ns
    total_per_iter_us = 1e-3 * total_walltime_ns / n_steps
    physics_time_per_iter_us = 1e-3 * total_physics_time_ns / n_steps
    physics_percent = 100 * total_physics_time_ns / total_walltime_ns
    physics_throughput = 1e9 * n_steps / total_physics_time_ns
    total_throughput = 1e9 * n_steps / total_walltime_ns
    physics_realtime_x = physics_throughput * timestep
    total_realtime_x = total_throughput * timestep

    if n_frames_rendered == 0:
        render_time_per_iter_us = float("nan")
        render_time_per_frame_us = float("nan")
        render_percent = float("nan")
        render_throughput = float("nan")
        render_realtime_x = float("nan")
    else:
        render_time_per_iter_us = 1e-3 * total_render_time_ns / n_steps
        render_time_per_frame_us = 1e-3 * total_render_time_ns / n_frames_rendered
        render_percent = 100 * total_render_time_ns / total_walltime_ns
        render_throughput = 1e9 * n_steps / total_render_time_ns
        render_realtime_x = render_throughput * timestep

    table = [
        [
            "Physics simulation advancement",
            physics_time_per_iter_us,
            physics_percent,
            physics_throughput,
            physics_realtime_x,
            physics_throughput * n_worlds,
            physics_realtime_x * n_worlds,
        ],
        [
            "Rendering*",
            render_time_per_iter_us,
            render_percent,
            render_throughput,
            render_realtime_x,
            render_throughput * n_worlds_rendered,
            render_realtime_x * n_worlds_rendered,
        ],
        [
            "TOTAL",
            total_per_iter_us,
            100,
            total_throughput,
            total_realtime_x,
            total_throughput * n_worlds,
            total_realtime_x * n_worlds,
        ],
    ]

    if n_frames_rendered == 0:
        rendering_note = "* Note: No frames were rendered."
    else:
        rendering_note = (
            f"* Note: {n_frames_rendered} frames were rendered out of {n_steps} steps. "
            f"Therefore, rendering time per image is {render_time_per_frame_us:.0f} us."
        )

    tab_str = tabulate(
        table,
        headers=[
            "\nStage",
            "Time/step\n(us)",
            "Percent\n(%)",
            "Throughput\n(iters/s)",
            "Throughput\nx realtime",
            "Throughput\n(iters/s)\n(parallelized)",
            "Throughput\nx realtime\n(parallelized)",
        ],
        floatfmt=("s", ".0f", ".0f", ".0f", ".2f", ".0f", ".2f"),
        tablefmt="html" if show_in_notebook else "simple_grid",
    )

    if show_in_notebook:
        from IPython.display import HTML, display

        print("PERFORMANCE PROFILE")
        display(HTML(tab_str))
        print(rendering_note)
    else:
        tab_width = max(len(line) for line in tab_str.splitlines())
        print()
        print("PERFORMANCE PROFILE".center(tab_width))
        print(tab_str)
        # Print the rendering note wrapped to the table width for better display
        print(textwrap.fill(rendering_note, width=tab_width))
        print()


def check_environment():
    """Detect the current execution environment. Possible return values are:
    "notebook", "terminal", "other", "standard_python".
    """
    try:
        from IPython import get_ipython

        shell = get_ipython().__class__.__name__

        if shell == "ZMQInteractiveShell":
            return "notebook"  # Jupyter Notebook or JupyterLab
        elif shell == "TerminalInteractiveShell":
            return "terminal"  # IPython terminal
        else:
            return "other"  # Other IPython shells
    except (NameError, ImportError):
        return "standard_python"  # Standard Python interpreter (script)
