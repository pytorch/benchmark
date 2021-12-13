#!/bin/sh

set -xueo pipefail

conda env remove --name ${CONDA_ENV}
