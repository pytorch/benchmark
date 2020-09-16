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
    PYTEST_FILTER="(not cyclegan)"
fi

BENCHMARK_DATA="`pwd`/.data"
mkdir -p ${BENCHMARK_DATA}
BENCHMARK_FILENAME=data_${CIRCLE_SHA1}.json
BENCHMARK_ABS_FILENAME=${BENCHMARK_DATA}/${BENCHMARK_FILENAME}
pytest test_bench.py --setup-show --benchmark-sort=Name --benchmark-json=${BENCHMARK_ABS_FILENAME} -k "$PYTEST_FILTER"


# Token is only present for certain jobs, only upload if present
if [ -z "$SCRIBE_GRAPHQL_ACCESS_TOKEN" ]
then
    echo "Skipping benchmark upload, token is missing."
else
    python scripts/upload_scribe.py --pytest_bench_json ${BENCHMARK_ABS_FILENAME}
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
    git add ${SUBDIR}/latest
fi
git commit -m"Add ${BENCHMARK_FILENAME} - to reproduce results, see pytorch and machine versions in .json"
git push origin master
popd

if [ "$CIRCLE_BRANCH" != "master" ]
then
    echo "Running benchmark comparison step"
    python scripts/compare_benchmark.py \
        --old pytorch_benchmark_data/benchmark/master/latest \
        --new ${BENCHMARK_ABS_FILENAME} \
        2>&1 | tee compare.log
fi

# Post a comment to the PR
if [ "$CIRCLE_BRANCH" != "master" ]
then
    sudo apt-get install jq
    GH_USER='pytorchbot'
    GH_API=$GITHUB_PYTORCHBOT_TOKEN
    pr_response=$(curl --location --request GET "https://api.github.com/repos/$CIRCLE_PROJECT_USERNAME/$CIRCLE_PROJECT_REPONAME/pulls?head=$CIRCLE_PROJECT_USERNAME:$CIRCLE_BRANCH&state=open" \
    -u $GH_USER:$GH_API)

    if [ $(echo $pr_response | jq length) -eq 0 ]; then
    echo "No PR found to update"
    else
    pr_comment_url=$(echo $pr_response | jq -r ".[]._links.comments.href")
    fi

    curl --location --request POST "$pr_comment_url" \
    -u $GH_USER:$GH_API \
    --header 'Content-Type: application/json' \
    --data-raw "{
    \"body\": \"`cat compare.log`\"
    }"
fi