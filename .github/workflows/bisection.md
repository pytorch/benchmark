# How to manually run a bisection job

To run a bisection job, please follow the instructions below:

1. Create a cool, unique name of your bisect job, call it `${BISECT_ISSUE}`. To avoid name conflict, we suggest using the github issue number.

2. On the github self-hosted runner node, create a bisection work directory at `${HOME}/.torchbench/bisection/${BISECT_ISSUE}`.

3. In the bisection work directory, create only one file: config.yaml. Example of config.yaml can be found [here](https://github.com/pytorch/benchmark/blob/0.1/.github/scripts/bisection-config.sample.yaml).

4. In the GitHub workflow [page](https://github.com/pytorch/benchmark/actions/workflows/bisection.yml) Specify the bisection job name as the "Bisection Issue Name" field in the UI.

5. Click the "Run workflow" button on master branch and wait for the job to finish. The result will be stored at `${HOME}/.torchbench/bisection/${BISECT_ISSUE}/result.json`.
