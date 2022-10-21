#set -x

mkdir debug_logs
export OUTPUT_DIR="$(pwd)/debug_logs"

model_all=$1
numa_mode=$2
additional_options=$3

vision_models="alexnet,densenet121,mnasnet1_0,mobilenet_v2,mobilenet_v3_large,resnet18,resnet50,resnext50_32x4d,shufflenet_v2_x1_0,squeezenet1_1,vgg16"
detectron_models="detectron2_fasterrcnn_r_101_c4,detectron2_fasterrcnn_r_101_dc5,detectron2_fasterrcnn_r_101_fpn,detectron2_fasterrcnn_r_50_c4,detectron2_fasterrcnn_r_50_dc5,detectron2_fasterrcnn_r_50_fpn,detectron2_fcos_r_50_fpn,detectron2_maskrcnn,detectron2_maskrcnn_r_101_c4,detectron2_maskrcnn_r_101_fpn,detectron2_maskrcnn_r_50_c4,detectron2_maskrcnn_r_50_fpn"
hf_models="hf_Albert,hf_Bart,hf_Bert,hf_BigBird,hf_DistilBert,hf_GPT2,hf_GPT2_large,hf_Longformer,hf_Reformer,hf_T5,hf_T5_base,hf_T5_large"
timm_models="timm_efficientnet,timm_nfnet,timm_regnet,timm_resnest,timm_vision_transformer,timm_vision_transformer_large,timm_vovnet"
channels_last_models="dcgan,mobilenet_v2_quantized_qat,pytorch_unet,resnet50_quantized_qat,yolov3"
time_long_models="LearningToPaint,maml,speech_transformer"
other_models="attention_is_all_you_need_pytorch,BERT_pytorch,demucs,dlrm,drq,fastNLP_Bert,maml_omniglot,nvidia_deeprecommender,opacus_cifar10, \
                pyhpc_equation_of_state,pyhpc_isoneutral_mixing,pyhpc_turbulent_kinetic_energy,pytorch_CycleGAN_and_pix2pix,pytorch_stargan, \
                soft_actor_critic,Super_SloMo,tacotron2,tts_angular,vision_maskrcnn"
functorch_models="functorch_dp_cifar10,functorch_maml_omniglot,lennard_jones"
not_implemented_models="Background_Matting,DALLE2_pytorch,fambench_xlmr,moco,pytorch_struct,timm_efficientdet"

if [ "${model_all}" == "all" ];then
    #model_all="${vision_models},${detectron_models},${hf_models},${timm_models},${channels_last_models},${time_long_models},${other_models}"
    model_all="${vision_models},${hf_models},${timm_models},${channels_last_models},${other_models}"
elif [ "${model_all}" == "time_long" ]; then
    model_all="${detectron_models},${time_long_models}"
fi

model_list=($(echo "${model_all}" |sed 's/,/ /g'))

sockets_num=$(lscpu |grep 'Socket(s):' |sed 's/[^0-9]//g')
cores_per_socket=$(lscpu |grep 'Core(s) per socket:' |sed 's/[^0-9]//g')
phsical_cores_num=`expr ${sockets_num} \* ${cores_per_socket}`
numa_nodes_num=$(lscpu |grep 'NUMA node(s):' |sed 's/[^0-9]//g')
cores_per_node=`expr ${phsical_cores_num} / ${numa_nodes_num}`
cpu_model="$(lscpu |grep 'Model name:' |sed 's/.*: *//')"

if [ ${numa_mode} == "throughput" ]; then
    ncpi=${cores_per_node}
    num_instances=1
    batch_size=`expr ${cores_per_node} \* 2`
elif [ ${numa_mode} == "latency" ]; then
    ncpi=1
    num_instances=${cores_per_node}
    batch_size=1
elif [ ${numa_mode} == "multi_instance" ]; then
    ncpi=4
    num_instances=`expr ${cores_per_node} / ${ncpi}`
    batch_size=1
elif [ ${numa_mode} == "throughput_bs1" ]; then
    ncpi=${cores_per_node}
    num_instances=1
    batch_size=1
fi
numa_launch_header=" python -m launch --node_id 0 --ninstances ${num_instances} --ncore_per_instance ${ncpi} --log_path=${OUTPUT_DIR}"

export LD_PRELOAD=${CONDA_PREFIX}/lib/libjemalloc.so
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
export KMP_AFFINITY="granularity=fine,compact,1,0"
export KMP_BLOCKTIME=1
export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
export KMP_SETTINGS=1

for model in ${model_list[@]}
do
    log_file_prefix="${model}_${numa_mode}"
    numa_launch_header+=" --log_file_prefix=${model}_${numa_mode}"
    ${numa_launch_header} debug_run.py ${model} --bs ${batch_size} ${additional_options}
    cpu_total_time=$(grep "CPU Total Wall Time:" ./debug_logs/${log_file_prefix}*.log | sed -e 's/.*CPU Total Wall Time://;s/[^0-9.]//g' | awk 'BEGIN {sum = 0;}{sum = sum + $1;} END {printf("%.3f", sum);}')
    num_cpu_total_time=$(grep "CPU Total Wall Time:" ./debug_logs/${log_file_prefix}*.log | wc -l)
    cpu_avg_time=`awk 'BEGIN {printf "%.2f",'${cpu_total_time}'/'${num_cpu_total_time}'}'`
    throughput=$(grep "Throughput:" ./debug_logs/${log_file_prefix}*.log | sed -e 's/.*Throughput//;s/[^0-9.]//g' | awk 'BEGIN {sum = 0;}{sum = sum + $1;} END {printf("%.3f", sum);}')
    echo torchbench ${model} ${precision} ${numa_mode} ${cpu_avg_time} ${throughput} | tee -a ./debug_logs/summary.log
done

cat ./debug_logs/summary.log