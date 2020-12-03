#1/bin/sh

set -eo pipefail

cat > ~/.netrc <<EOF
machine github.com
login pytorchbot
password ${GITHUB_TORCHBENCH_TOKEN}
EOF

BENCHMARK_DATA_DIR=${HOME}/pytorch_benchmark_data
BENCHMARK_DATA_REPO=https://github.com/wconstab/pytorch_benchmark_data

if [ "${GITHUB_REPOSITORY}" != "pytorch/benchmark" ]; then
    echo "Please run this action only on the master branch of the pytorch/benchmark repo"
    exit 1;
fi

# Upload the first json in the same run to the wconstab/pytorch_benchmark_data repository
SUBDIR="nightly-v0"
DATA_DIR=${HOME}/benchmark-results/gh${GITHUB_RUN_ID}
BENCHMARK_ABS_FILE=$(find ${DATA_DIR} -name *.json | head -n 1)
# Make sure the result json exists
if [ -z "${BENCHMARK_ABS_FILE}" ]; then
    exit 1
fi

if [ -e $BENCHMARK_DATA_DIR ]; then
    pushd ${BENCHMARK_DATA_DIR}
    git pull origin master
else
    git clone ${BENCHMARK_DATA_REPO} ${BENCHMARK_DATA_DIR}
fi

git config --global user.name "pytorchbot"

BENCHMARK_FILENAME=${GITHUB_SHA}_$(basename ${BENCHMARK_ABS_FILE})
pushd ${BENCHMARK_DATA_DIR}
mkdir -p ${SUBDIR}
cp ${BENCHMARK_ABS_FILE} ${SUBDIR}/${BENCHMARK_FILENAME}
ln -s -f ${SUBDIR}/${BENCHMARK_FILENAME} ${SUBDIR}/latest
echo ${BENCHMARK_FILENAME} >> ${SUBDIR}/history

git add ${SUBDIR}
git commit -m"Add ${BENCHMARK_FILENAME} - to reproduce results, see pytorch and machine versions in .json."
git push origin master

popd
