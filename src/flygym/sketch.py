import textwrap
from tabulate import tabulate

# total_prestep_ns = 1.2e9
# total_physics_ns = 5.5e9
# total_poststep_ns = 0.8e9
# total_render_ns = 2.5e9
# curr_step = 10000
# frames_rendered = 1000


def print_perf_report(
    total_prestep_ns: int,
    total_physics_ns: int,
    total_poststep_ns: int,
    total_render_ns: int,
    curr_step: int,
    frames_rendered: int,
):
    total_walltime_ns = (
        total_prestep_ns + total_physics_ns + total_poststep_ns + total_render_ns
    )
    total_per_iter_us = 1e-3 * total_walltime_ns / curr_step
    prestep_time_per_iter_us = 1e-3 * total_prestep_ns / curr_step
    physics_time_per_iter_us = 1e-3 * total_physics_ns / curr_step
    poststep_time_per_iter_us = 1e-3 * total_poststep_ns / curr_step
    render_time_per_iter_us = 1e-3 * total_render_ns / curr_step
    render_time_per_frame_us = 1e-3 * total_render_ns / frames_rendered

    prestep_percent = 100 * total_prestep_ns / total_walltime_ns
    physics_percent = 100 * total_physics_ns / total_walltime_ns
    poststep_percent = 100 * total_poststep_ns / total_walltime_ns
    render_percent = 100 * total_render_ns / total_walltime_ns

    prestep_throughput = 1e9 * curr_step / total_prestep_ns
    physics_throughput = 1e9 * curr_step / total_physics_ns
    poststep_throughput = 1e9 * curr_step / total_poststep_ns
    render_throughput = 1e9 * curr_step / total_render_ns
    total_throughput = 1e9 * curr_step / total_walltime_ns

    table = [
        [
            "Pre-step control input computation",
            prestep_time_per_iter_us,
            prestep_percent,
            prestep_throughput,
        ],
        [
            "Physics simulation advancement",
            physics_time_per_iter_us,
            physics_percent,
            physics_throughput,
        ],
        [
            "Post-step updated state processing",
            poststep_time_per_iter_us,
            poststep_percent,
            poststep_throughput,
        ],
        [
            "Rendering*",
            render_time_per_iter_us,
            render_percent,
            render_throughput,
        ],
        [
            "TOTAL",
            total_per_iter_us,
            100,
            total_throughput,
        ],
    ]
    tab_str = tabulate(
        table,
        headers=["\nStage", "Time/step\n(us)", "Percent\n(%)", "Throughput\n(iters/s)"],
        floatfmt=("s", ".0f", ".0f", ".0f"),
        tablefmt="simple_grid",
    )
    tab_width = max(len(line) for line in tab_str.splitlines())
    title = "PERFORMANCE PROFILE"
    print()
    print(title.center(tab_width))
    print(tab_str)
    rendering_note = (
        f"* Note: {frames_rendered} frames were rendered out of {curr_step} steps. "
        f"Therefore, rendering time per image is {render_time_per_frame_us:.0f} us."
    )
    print(textwrap.fill(rendering_note, width=tab_width))
    print()


print_perf_report()
