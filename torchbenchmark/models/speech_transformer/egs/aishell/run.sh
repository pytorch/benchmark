#!/bin/bash

# -- IMPORTANT
data=/home/work_nfs/common/data # Modify to your aishell data path
stage=-1  # Modify to control start from witch stage
# --

ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
nj=40

dumpdir=dump   # directory to dump full features

# Feature configuration
do_delta=false
LFR_m=4  # Low Frame Rate: number of frames to stack
LFR_n=3  # Low Frame Rate: number of frames to skip

# Network architecture
# Encoder
d_input=80
n_layers_enc=6
n_head=8
d_k=64
d_v=64
d_model=512
d_inner=2048
dropout=0.1
pe_maxlen=5000
# Decoder
d_word_vec=512
n_layers_dec=6
tgt_emb_prj_weight_sharing=1
# Loss
label_smoothing=0.1

# Training config
epochs=150
# minibatch
shuffle=1
batch_size=16
batch_frames=15000
maxlen_in=800
maxlen_out=150
# optimizer
k=0.2
warmup_steps=4000
# save & logging
checkpoint=0
continue_from=""
print_freq=10
visdom=0
visdom_lr=0
visdom_epoch=0
visdom_id="Transformer Training"

# Decode config
beam_size=5
nbest=1
decode_max_len=100

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;
. ./cmd.sh
. ./path.sh

if [ $stage -le 0 ]; then
    echo "stage 0: Data Preparation"
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    # Generate wav.scp, text, utt2spk, spk2utt (segments)
    local/aishell_data_prep.sh $data/data_aishell/wav $data/data_aishell/transcript || exit 1;
    # remove space in text
    for x in train test dev; do
        cp data/${x}/text data/${x}/text.org
        paste -d " " <(cut -f 1 -d" " data/${x}/text.org) <(cut -f 2- -d" " data/${x}/text.org | tr -d " ") \
            > data/${x}/text
    done
fi

feat_train_dir=${dumpdir}/train/delta${do_delta}; mkdir -p ${feat_train_dir}
feat_test_dir=${dumpdir}/test/delta${do_delta}; mkdir -p ${feat_test_dir}
feat_dev_dir=${dumpdir}/dev/delta${do_delta}; mkdir -p ${feat_dev_dir}
if [ $stage -le 1 ]; then
    echo "stage 1: Feature Generation"
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    fbankdir=fbank
    for data in train test dev; do
        steps/make_fbank.sh --cmd "$train_cmd" --nj $nj --write_utt2num_frames true \
            data/$data exp/make_fbank/$data $fbankdir/$data || exit 1;
    done
    # compute global CMVN
    compute-cmvn-stats scp:data/train/feats.scp data/train/cmvn.ark
    # dump features for training
    for data in train test dev; do
        feat_dir=`eval echo '$feat_'${data}'_dir'`
        dump.sh --cmd "$train_cmd" --nj $nj --do_delta $do_delta \
            data/$data/feats.scp data/train/cmvn.ark exp/dump_feats/$data $feat_dir
    done
fi

dict=data/lang_1char/train_chars.txt
echo "dictionary: ${dict}"
nlsyms=data/lang_1char/non_lang_syms.txt
if [ $stage -le 2 ]; then
    echo "stage 2: Dictionary and Json Data Preparation"
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    mkdir -p data/lang_1char/

    echo "make a non-linguistic symbol list"
    # It's empty in AISHELL-1
    cut -f 2- data/train/text | grep -o -P '\[.*?\]' | sort | uniq > ${nlsyms}
    cat ${nlsyms}

    echo "make a dictionary"
    echo "<unk> 0" >  ${dict}
    echo "<sos> 1" >> ${dict}
    echo "<eos> 2" >> ${dict}
    text2token.py -s 1 -n 1 -l ${nlsyms} data/train/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+2}' >> ${dict}
    wc -l ${dict}

    echo "make json files"
    for data in train test dev; do
        feat_dir=`eval echo '$feat_'${data}'_dir'`
        data2json.sh --feat ${feat_dir}/feats.scp --nlsyms ${nlsyms} \
             data/$data ${dict} > ${feat_dir}/data.json
    done
fi

if [ -z ${tag} ]; then
    expdir=exp/train_m${LFR_m}_n${LFR_n}_in${d_input}_elayer${n_layers_enc}_head${n_head}_k${d_k}_v${d_v}_model${d_model}_inner${d_inner}_drop${dropout}_pe${pe_maxlen}_emb${d_word_vec}_dlayer${n_layers_dec}_share${tgt_emb_prj_weight_sharing}_ls${label_smoothing}_epoch${epochs}_shuffle${shuffle}_bs${batch_size}_bf${batch_frames}_mli${maxlen_in}_mlo${maxlen_out}_k${k}_warm${warmup_steps}
    if ${do_delta}; then
        expdir=${expdir}_delta
    fi
else
    expdir=exp/train_${tag}
fi
mkdir -p ${expdir}

if [ ${stage} -le 3 ]; then
    echo "stage 3: Network Training"
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        train.py \
        --train-json ${feat_train_dir}/data.json \
        --valid-json ${feat_dev_dir}/data.json \
        --dict ${dict} \
        --LFR_m ${LFR_m} \
        --LFR_n ${LFR_n} \
        --d_input $d_input \
        --n_layers_enc $n_layers_enc \
        --n_head $n_head \
        --d_k $d_k \
        --d_v $d_v \
        --d_model $d_model \
        --d_inner $d_inner \
        --dropout $dropout \
        --pe_maxlen $pe_maxlen \
        --d_word_vec $d_word_vec \
        --n_layers_dec $n_layers_dec \
        --tgt_emb_prj_weight_sharing $tgt_emb_prj_weight_sharing \
        --label_smoothing ${label_smoothing} \
        --epochs $epochs \
        --shuffle $shuffle \
        --batch-size $batch_size \
        --batch_frames $batch_frames \
        --maxlen-in $maxlen_in \
        --maxlen-out $maxlen_out \
        --k $k \
        --warmup_steps $warmup_steps \
        --save-folder ${expdir} \
        --checkpoint $checkpoint \
        --continue-from "$continue_from" \
        --print-freq ${print_freq} \
        --visdom $visdom \
        --visdom_lr $visdom_lr \
        --visdom_epoch $visdom_epoch \
        --visdom-id "$visdom_id"
fi

if [ ${stage} -le 4 ]; then
    echo "stage 4: Decoding"
    decode_dir=${expdir}/decode_test_beam${beam_size}_nbest${nbest}_ml${decode_max_len}
    mkdir -p ${decode_dir}
    ${cuda_cmd} --gpu ${ngpu} ${decode_dir}/decode.log \
        recognize.py \
        --recog-json ${feat_test_dir}/data.json \
        --dict $dict \
        --result-label ${decode_dir}/data.json \
        --model-path ${expdir}/final.pth.tar \
        --beam-size $beam_size \
        --nbest $nbest \
        --decode-max-len $decode_max_len

    # Compute CER
    local/score.sh --nlsyms ${nlsyms} ${decode_dir} ${dict}
fi
