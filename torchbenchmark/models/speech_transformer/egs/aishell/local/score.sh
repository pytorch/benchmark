#!/bin/bash

[ -f path.sh ] && . ./path.sh

nlsyms=""

. utils/parse_options.sh

if [ $# != 2 ]; then
    echo "Usage: $0 <data-dir> <dict>"
    exit 1
fi

dir=$1
dic=$2

json2trn.py ${dir}/data.json ${dic} ${dir}/ref.trn ${dir}/hyp.trn

if [ ! -z ${nlsyms} ]; then
  cp ${dir}/ref.trn ${dir}/ref.trn.org
  cp ${dir}/hyp.trn ${dir}/hyp.trn.org
  filt.py -v $nlsyms ${dir}/ref.trn.org > ${dir}/ref.trn
  filt.py -v $nlsyms ${dir}/hyp.trn.org > ${dir}/hyp.trn
fi

sclite -r ${dir}/ref.trn trn -h ${dir}/hyp.trn trn -i rm -o all stdout > ${dir}/result.txt

echo "write a CER (or TER) result in ${dir}/result.txt"
grep -e Avg -e SPKR -m 2 ${dir}/result.txt
