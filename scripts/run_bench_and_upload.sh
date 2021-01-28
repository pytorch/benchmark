#!/bin/bash
set -e
. ~/miniconda3/etc/profile.d/conda.sh
conda activate base

# set credentials for git https pushing
cat > ~/.netrc <<DONE
machine github.com
login pytorchbot
password ${GITHUB_PYTORCHBOT_TOKEN}
DONE

if [ "$CIRCLE_BRANCH" = "master" ]
then
    PYTEST_FILTER=""
else
    PYTEST_FILTER="(not cyclegan) and (not (stargan and train and cpu))"
fi

BENCHMARK_DATA="`pwd`/.data"
mkdir -p ${BENCHMARK_DATA}
BENCHMARK_FILENAME=${CIRCLE_SHA1}_$(date +"%Y%m%d_%H%M%S").json
BENCHMARK_ABS_FILENAME=${BENCHMARK_DATA}/${BENCHMARK_FILENAME}
pytest test_bench.py --ignore_machine_config --setup-show --benchmark-sort=Name --benchmark-json=${BENCHMARK_ABS_FILENAME} -k "$PYTEST_FILTER"

# Compute benchmark score
TORCHBENCH_SCORE=$(python compute_score.py --configuration torchbenchmark/score/configs/v0/config-v0.yaml --benchmark_data_file ${BENCHMARK_ABS_FILENAME})
# Token is only present for certain jobs, only upload if present
if [ -z "$SCRIBE_GRAPHQL_ACCESS_TOKEN" ]
then
    echo "Skipping benchmark upload, token is missing."
else
    python scripts/upload_scribe.py --pytest_bench_json ${BENCHMARK_ABS_FILENAME} --torchbench_score ${TORCHBENCH_SCORE}
fi


git clone https://github.com/wconstab/pytorch_benchmark_data
pushd pytorch_benchmark_data
git config --global user.name "pytorchbot"
git config --global --unset url.ssh://git@github.com.insteadof || true

# Store the files with respect to the repo that their hash refers to
# separate master and pull_requests just for conveneince in the UI
if [ "$CIRCLE_BRANCH" = "master" ]
then
    SUBDIR="${CIRCLE_PROJECT_REPONAME}/master"
else
    SUBDIR="${CIRCLE_PROJECT_REPONAME}/pull_requests"
fi
mkdir -p $SUBDIR/
cp ${BENCHMARK_ABS_FILENAME} $SUBDIR/
git add ${SUBDIR}/${BENCHMARK_FILENAME}
if [ "$CIRCLE_BRANCH" = "master" ]
then
    ln -s -f ${SUBDIR}/${BENCHMARK_FILENAME} ${SUBDIR}/latest
    echo ${BENCHMARK_FILENAME} >> ${SUBDIR}/history
    git add ${SUBDIR}/latest ${SUBDIR}/history
fi
git commit -m"Add ${BENCHMARK_FILENAME} - to reproduce results, see pytorch and machine versions in .json"
git push origin master
popd
