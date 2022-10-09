set -x

mode_all=$1
if [ ${mode_all} == "all" ]; then
    mode_all="latency,multi_instance"
fi

mode_list=($(echo "${mode_all}" |sed 's/,/ /g'))

for mode in ${mode_list[@]}
do
    model="all"
    bash launch_benchmark.sh ${model} ${mode} "-m eager"
    bash launch_benchmark.sh ${model} ${mode} "-m jit "
    bash launch_benchmark.sh ${model} ${mode} "-m eager --channels-last"
    bash launch_benchmark.sh ${model} ${mode} "-m jit --channels-last"

    model="time_long"
    bash launch_benchmark.sh ${model} ${mode} "--num-iter 10 -m eager"
    bash launch_benchmark.sh ${model} ${mode} "--num-iter 10 -m jit "
    bash launch_benchmark.sh ${model} ${mode} "--num-iter 10 -m eager --channels-last"
    bash launch_benchmark.sh ${model} ${mode} "--num-iter 10 -m jit --channels-last"
done
