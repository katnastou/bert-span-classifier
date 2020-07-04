#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=8G
#SBATCH -p gputest
#SBATCH -t 00:15:00
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks-per-node=1
#SBATCH --account=Project_2001426
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

OUTPUT_DIR="output/$SLURM_JOBID"

# function on_exit {
#     rm -rf "$OUTPUT_DIR"
#     rm -f jobs/$SLURM_JOBID
# }
# trap on_exit EXIT

if [[ "$#" -lt 7 ]]; then
    echo "Usage: $0 model data_dir seq_len batch_size learning_rate epochs task_name [model_dir] [--other-args]"
    exit 1
fi

MODEL="$1"
DATA_DIR="$2"
MAX_SEQ_LENGTH="$3"
BATCH_SIZE="$4"
LEARNING_RATE="$5"
EPOCHS="$6"
TASK_NAME="$7"

if [[ "$#" -gt 7 ]] && [[ "$8" != --* ]]; then
    modelparam="--model_dir $8"
    shift 8
else
    modelparam=""
    shift 7
fi
otherparams="$@"

VOCAB="$(dirname "$MODEL")/vocab.txt"
CONFIG="$(dirname "$MODEL")/bert_config.json"

if [[ $MODEL =~ "uncased" ]]; then
    caseparam="--do_lower_case"
elif [[ $MODEL =~ "multilingual" ]]; then
    caseparam="--do_lower_case"
else
    caseparam=""
fi

rm -rf "OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

rm -f latest.out latest.err
ln -s logs/$SLURM_JOBID.out latest.out
ln -s logs/$SLURM_JOBID.err latest.err

module purge
module load gcc/8.3.0
module load cuda
module load cudnn

source venv/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "START $SLURM_JOBID: $(date)"

# https://stackoverflow.com/a/34195247
if compgen -G "$DATA_DIR/train*.tfrecord" >/dev/null; then
    train_data=$(ls "$DATA_DIR"/train*.tfrecord | tr '\n' ',' | perl -pe 's/,$//')
else
    train_data=$(ls "$DATA_DIR"/train*.tsv | tr '\n' ',' | perl -pe 's/,$//')
fi
echo "Using $train_data as training data" >&2

$label_field = "-1"

$text_fields = "3"

srun python3 train.py \
    --replace_span_A "[unused1]" \
    --replace_span_B "[unused3]" \
    --task_name "$TASK_NAME" \
    --bert_config_file "$CONFIG" \
    --init_checkpoint "$MODEL" \
    --vocab_file "$VOCAB" \
    --learning_rate $LEARNING_RATE \
    --max_seq_length $MAX_SEQ_LENGTH \
    --batch_size $BATCH_SIZE \
    --num_train_epochs $EPOCHS \
    --train_data "$train_data" \
    --dev_data "$DATA_DIR/dev.tsv" \
    --labels "$DATA_DIR/labels.txt" \
    --label_field "$label_field" \
    --text_fields "$text_fields" \
    $caseparam \
    $modelparam \
    $otherparams

result=$(egrep '^Final dev accuracy:' logs/${SLURM_JOB_ID}.out | perl -pe 's/.*accuracy: (\S+)\%.*/$1/')

echo -n 'TEST-RESULT'$'\t'
echo -n 'init_checkpoint'$'\t'"$MODEL"$'\t'
echo -n 'data_dir'$'\t'"$DATA_DIR"$'\t'
echo -n 'max_seq_length'$'\t'"$MAX_SEQ_LENGTH"$'\t'
echo -n 'train_batch_size'$'\t'"$BATCH_SIZE"$'\t'
echo -n 'learning_rate'$'\t'"$LEARNING_RATE"$'\t'
echo -n 'num_train_epochs'$'\t'"$EPOCHS"$'\t'
echo -n 'other_parameters'$'\t'"$otherparams"$'\t'
echo 'accuracy'$'\t'"$result"

gpuseff $SLURM_JOBID

echo "END $SLURM_JOBID: $(date)"
