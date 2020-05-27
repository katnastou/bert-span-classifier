# bert-span-classifier

Text span classifier using BERT

## Quickstart

```
git clone https://github.com/spyysalo/bert-span-classifier.git
cd bert-span-classifier/

./scripts/get-models.sh

module load tensorflow/2.0.0    # for Slurm, replace with equivalent

python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt

python train.py \
    --init_checkpoint models/cased_L-12_H-768_A-12/bert_model.ckpt \
    --vocab_file models/cased_L-12_H-768_A-12/vocab.txt \
    --bert_config_file models/cased_L-12_H-768_A-12/bert_config.json \
    --train_data example-data/train.tsv --dev_data example-data/dev.tsv
```

## On slurm

First edit `slurm/slurm-run-test.sh` to match your setup (partition etc.)

```
sbatch slurm/slurm-run-test.sh models/cased_L-12_H-768_A-12/bert_model.ckpt example-data 64 16 3e-5 2
```
