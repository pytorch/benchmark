"""MLCube handler file"""
import os
import shutil
import subprocess
from pathlib import Path

import typer
import yaml

app = typer.Typer()


class RunTestTask(object):
    """Run test task Class"""

    @staticmethod
    def run(parameters_file: str, output_dir: str, mode: str) -> None:
        """Execute test.py script using python"""

        if mode == "list":
            command_args = "--ignore_machine_config --collect-only"

        elif mode == "run":
            with open(parameters_file, "r") as stream:
                parameters = yaml.safe_load(stream)

            command_args = f"-k test_{parameters['model_name']}_{parameters['mode']}_{parameters['platform']}"

        else:
            command_args = ""

        env = os.environ.copy()
        env.update(
            {
                "MODE": mode,
                "COMMAND_ARGS": command_args,
                "OUTPUT_DIR": output_dir,
            }
        )

        process = subprocess.Popen("./run_test.sh", cwd=".", env=env)
        process.wait()


class RunTestBenchTask(object):
    """Run test bench task Class"""

    @staticmethod
    def run(parameters_file: str, output_dir: str, mode: str) -> None:
        """Execute test_bench.py script using pytest"""

        if mode == "list":
            command_args = "--ignore_machine_config --collect-only"

        else:
            with open(parameters_file, "r") as stream:
                parameters = yaml.safe_load(stream)

            command_args = "--ignore_machine_config --benchmark-autosave"
            if parameters["platform"] == "cpu":
                command_args += " --cpu_only"

            if mode == "run":
                command_args += f" -k {parameters['test_bench_name']}"

            elif mode == "run_all":
                command_args += ""

        env = os.environ.copy()
        env.update(
            {
                "MODE": mode,
                "COMMAND_ARGS": command_args,
                "OUTPUT_DIR": output_dir,
            }
        )

        process = subprocess.Popen("./run_test_bench.sh", cwd=".", env=env)
        process.wait()


@app.command("list_test")
def list_test(
    output_dir: str = typer.Option(..., "--output_dir"),
):
    """List test linked to test.py script"""
    RunTestTask.run("", output_dir, mode="list")


@app.command("run_test")
def run_test(
    parameters_file: str = typer.Option(..., "--parameters_file"),
    output_dir: str = typer.Option(..., "--output_dir"),
):
    """Run specific test using test.py script"""
    RunTestTask.run(parameters_file, output_dir, mode="run")


@app.command("run_test_all")
def run_test_all(
    output_dir: str = typer.Option(..., "--output_dir"),
):
    """Run all tests under test.py script"""
    RunTestTask.run("", output_dir, mode="run_all")


@app.command("list_test_bench")
def list_test_bench(
    output_dir: str = typer.Option(..., "--output_dir"),
):
    """List test linked to test_bench.py script"""
    RunTestBenchTask.run("", output_dir, mode="list")


@app.command("run_test_bench")
def run_test_bench(
    parameters_file: str = typer.Option(..., "--parameters_file"),
    output_dir: str = typer.Option(..., "--output_dir"),
):
    """Run specific test using test_bench.py script"""
    RunTestBenchTask.run(parameters_file, output_dir, mode="run")


@app.command("run_test_bench_all")
def run_test_bench_all(
    parameters_file: str = typer.Option(..., "--parameters_file"),
    output_dir: str = typer.Option(..., "--output_dir"),
):
    """Run all tests under test_bench.py script"""
    RunTestBenchTask.run(parameters_file, output_dir, mode="run_all")


if __name__ == "__main__":
    app()
