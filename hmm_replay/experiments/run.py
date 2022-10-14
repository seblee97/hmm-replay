import argparse
import os

from hmm_replay import constants, runners
from hmm_replay.experiments import hmm_config
from hmm_replay.runners import core_runner
from run_modes import cluster_run, parallel_run, serial_run, single_run, utils

MAIN_FILE_PATH = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()

parser.add_argument("--mode", metavar="-M", required=True, help="run experiment.")
parser.add_argument(
    "--config_path",
    metavar="-C",
    default="config.yaml",
    help="path to base configuration file.",
)
parser.add_argument("--seeds", metavar="-S", default=None, help="list of seeds to run.")
parser.add_argument("--config_changes", metavar="-CC", default="config_changes.py")

# cluster config
parser.add_argument("--scheduler", type=str, help="univa or slurm", default="univa")
parser.add_argument("--num_cpus", type=int, default=4)
parser.add_argument("--num_gpus", type=int, default=0)
parser.add_argument("--mem", type=int, default=16)
parser.add_argument("--timeout", type=str, default="")
parser.add_argument("--cluster_debug", action="store_true")

STOCHASTIC_PACKAGES = ["numpy", "random"]
RUN_METHODS = ["run"]


if __name__ == "__main__":

    args = parser.parse_args()

    results_folder = os.path.join(MAIN_FILE_PATH, constants.RESULTS)

    config_class_name = "HMMConfig"
    config_module_name = "hmm_config"
    config_module_path = os.path.join(MAIN_FILE_PATH, "hmm_config.py")
    config_class = hmm_config.HMMConfig
    config = config_class(args.config_path)

    runner_class_name = "CoreRunner"
    runner_module_name = "core_runner"
    runners_module_path = os.path.dirname(os.path.abspath(runners.__file__))
    runner_class = core_runner.CoreRunner

    if args.mode == constants.SINGLE:

        _, single_checkpoint_path = utils.setup_experiment(
            mode="single",
            results_folder=results_folder,
            config_path=args.config_path,
        )

        single_run.single_run(
            runner_class=runner_class,
            config_class=config_class,
            config_path=args.config_path,
            checkpoint_path=single_checkpoint_path,
            run_methods=RUN_METHODS,
            stochastic_packages=STOCHASTIC_PACKAGES,
        )

    elif args.mode in [
        constants.SINGLE_PARALLEL,
        constants.SINGLE_SERIAL,
        constants.SINGLE_CLUSTER,
        constants.PARALLEL,
        constants.SERIAL,
        constants.CLUSTER,
    ]:

        if constants.SINGLE in args.mode:
            config_changes_path = None
        else:
            config_changes_path = args.config_changes

        try:
            if args.seeds is not None:
                seeds_arg = int(args.seeds)
            else:
                seeds_arg = args.seeds
        except ValueError:
            seeds_arg = args.seeds
        seeds = utils.process_seed_arguments(seeds_arg)

        experiment_path, checkpoint_paths = utils.setup_experiment(
            mode="multi",
            results_folder=results_folder,
            config_path=args.config_path,
            config_changes_path=config_changes_path,
            seeds=seeds,
        )

        if constants.PARALLEL in args.mode:

            parallel_run.parallel_run(
                runner_class=runner_class,
                config_class=config_class,
                config_path=os.path.join(experiment_path, "config.yaml"),
                checkpoint_paths=checkpoint_paths,
                run_methods=RUN_METHODS,
                stochastic_packages=STOCHASTIC_PACKAGES,
            )

        elif constants.SERIAL in args.mode:

            serial_run.serial_run(
                runner_class=runner_class,
                config_class=config_class,
                config_path=os.path.join(experiment_path, "config.yaml"),
                checkpoint_paths=checkpoint_paths,
                run_methods=RUN_METHODS,
                stochastic_packages=STOCHASTIC_PACKAGES,
            )

        elif constants.CLUSTER in args.mode:

            cluster_run.cluster_run(
                runner_class_name=runner_class_name,
                runner_module_name=runner_module_name,
                runner_module_path=runner_module_path,
                config_class_name=config_class_name,
                config_module_name=config_module_name,
                config_module_path=config_module_path,
                config_path=os.path.join(experiment_path, "config.yaml"),
                checkpoint_paths=checkpoint_paths,
                run_methods=RUN_METHODS,
                stochastic_packages=STOCHASTIC_PACKAGES,
                env_name="rl_nav",
                scheduler=args.scheduler,
                num_cpus=args.num_cpus,
                num_gpus=args.num_gpus,
                memory=args.mem,
                walltime=args.timeout,
                cluster_debug=args.cluster_debug,
            )

    else:
        raise ValueError(f"run mode {args.mode} not recognised.")
