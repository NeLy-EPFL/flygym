import textwrap
from tabulate import tabulate

__all__ = ["print_perf_report"]


def print_perf_report(
    total_physics_time_ns: int,
    total_render_time_ns: int,
    n_steps: int,
    n_frames_rendered: int,
    timestep: float,
):
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
        tablefmt="simple_grid",
    )
    tab_width = max(len(line) for line in tab_str.splitlines())
    title = "PERFORMANCE PROFILE"
    print()
    print(title.center(tab_width))
    print(tab_str)

    if n_frames_rendered == 0:
        rendering_note = "* Note: No frames were rendered."
    else:
        rendering_note = (
            f"* Note: {n_frames_rendered} frames were rendered out of {n_steps} steps. "
            f"Therefore, rendering time per image is {render_time_per_frame_us:.0f} us."
        )
    print(textwrap.fill(rendering_note, width=tab_width))
    print()
