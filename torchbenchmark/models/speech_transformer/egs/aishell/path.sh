MAIN_ROOT=$PWD/../..
KALDI_ROOT=$MAIN_ROOT/tools/kaldi
SRC_ROOT=$MAIN_ROOT/src

# BEGIN from kaldi path.sh
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sctk/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
# END

export PATH=/home/work_nfs/ktxu/tools/anaconda3/bin:$PATH
export PATH=$SRC_ROOT/bin/:$SRC_ROOT/utils/:$PATH
export PYTHONPATH=$SRC_ROOT/data/:$SRC_ROOT/transformer/:$SRC_ROOT/solver/:$SRC_ROOT/utils/:$PYTHONPATH
