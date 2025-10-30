import sys
import argparse
from typing import Tuple, Optional, Any


from src.models import (
    DisplayConfig,
    Mode,
    Parameter,
    Payoffs,
    DEFAULT_STEPS,
)

from src.sandbox import Sandbox


def parse_args() -> (
    Tuple[DisplayConfig, int, int, Mode, Optional[Parameter], Optional[float], bool]
):
    parser = argparse.ArgumentParser(description="qdrift plotting")

    parser.add_argument(
        "--run", action="store_true", help="do a single run from your TOML file"
    )
    parser.add_argument(
        "--n",
        type=int,
        help="number of agents",
    )
    parser.add_argument(
        "--steps",
        type=int,
        help="number of timesteps",
        default=DEFAULT_STEPS,
    )

    parser.add_argument(
        "--gaussian",
        action="store_true",
        help="apply gaussian smoothing",
    )
    parser.add_argument(
        "--box",
        action="store_true",
        help="apply box (mean) filter",
    )
    parser.add_argument(
        "--median",
        action="store_true",
        help="apply median filter",
    )

    parser.add_argument(
        "--sigma",
        type=float,
        help="sigma value for gaussian filter",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        help="window size for box or median filters",
    )

    parser.add_argument(
        "--paramtest",
        type=lambda s: Parameter[s.upper()],
        choices=list(Parameter),
        help="analyze a parameter's trends",
    )
    parser.add_argument(
        "--param_increment",
        type=float,
        help="define the step size for testing.",
    )
    parser.add_argument(
        "--save_runs",
        action="store_true",
        help="save all the runs played out in param testing.",
        default=False,
    )

    args = parser.parse_args()

    if not args.n:
        sys.exit("ERROR: --n is required.")

    if args.box or args.median:
        if args.window_size is None:
            sys.exit("ERROR: --window-size is required when using --box or --median.")

    if args.gaussian:
        if args.sigma is None:
            sys.exit("ERROR: --sigma is required when using --gaussian.")

    if args.window_size and not (args.box or args.median):
        sys.exit("ERROR: --window-size should only be used with --box or --median.")

    if args.sigma and not args.gaussian:
        sys.exit("ERROR: --sigma should only be used with --gaussian.")

    if args.run:
        if args.paramtest:
            sys.exit("ERROR: --run and --paramtest are mutually exclusive.")
        if args.save_runs:
            sys.exit("ERROR: --save_runs can only be used with --paramtest.")

    if args.param_increment:
        if not args.paramtest:
            sys.exit("ERROR: --param_increment must be used with --paramtest.")

        if args.param_increment >= 1:
            sys.exit("ERROR: --param_increment must be < 1.")
    else:
        if args.paramtest:
            sys.exit("ERROR: --param_increment must be used with --paramtest.")

    if not args.run and not args.paramtest:
        sys.exit("ERROR: either --run or --paramtest must be specified.")

    return (
        DisplayConfig(
            box=args.box,
            median=args.median,
            gaussian=args.gaussian,
            window=args.window_size,
            sigma=args.sigma,
        ),
        args.n,
        args.steps,
        Mode.RUN if args.run else Mode.PARAMTEST,
        args.paramtest if args.paramtest else None,
        args.param_increment if args.param_increment else None,
        args.save_runs,
    )


def main():
    (
        display_args,
        n_agents,
        n_steps,
        mode,
        param_type,
        param_increment,
        save_runs,
    ) = parse_args()
    payoffs = Payoffs()

    if mode == Mode.RUN:
        sandbox = Sandbox(
            n_agents=n_agents,
            display_args=display_args,
            payoffs=payoffs,
        )

        sandbox.run(n_steps=n_steps)
    elif (
        mode == Mode.PARAMTEST
        and param_type is not None
        and param_increment is not None
    ):
        sandbox = Sandbox(
            n_agents=n_agents,
            display_args=display_args,
            payoffs=payoffs,
        )

        sandbox.param_test(param_type, n_steps, param_increment, save_runs)


if __name__ == "__main__":
    main()
