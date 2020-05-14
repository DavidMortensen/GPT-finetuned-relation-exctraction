# GPT-finetuned-relation-exctraction
Improving Relationship Extraction with pre-trained Transformer Language Models (Open AI GPT). Most code are directly taken from the work of Huggingface's reimplementation of the OpenAI GPT.

## Installation

Clone repo to local and then download the weights of the OpenAI pre-trained Transformer with the following command: 

```
./download-model.sh
```

## Preperation of the data

The tacred dataset can be downloaded from LDC (https://catalog.ldc.upenn.edu/LDC2018T24) and will need to be formated using: 
```bash
python data_converter.py <DATASET DIR> <CONVERTED DATASET DIR>
```
## Run training

```bash
python re_model.py train \
  --write-model True \
  --masking-mode grammar_and_ner \
  --batch-size 8 \
  --max-epochs 3 \
  --lm-coef 0.5 \
  --learning-rate 5.25e-5 \
  --learning-rate-warmup 0.002 \
  --clf-pdrop 0.1 \
  --attn-pdrop 0.1 \
  --word-pdrop 0.0 \
  --dataset tacred \
  --data-dir tacred_data \
  --seed=0 \
  --log-dir ./logs/
```

## Model Evaluation
The best model found in the research is supplied in the hand-in and can be run using the below script

## Model Evaluation

Edit the paths accordingly to the directories of the files. 
```bash
python re_model.py evaluate \
  --dataset tacred \
  --masking_mode grammar_and_ner \
  --test_file /GPT-finetuned-relation-exctraction/tacred_data/test.jsonl \
  --save_dir /GPT-finetuned-relation-exctraction/logs/2020-05-10__14-47__764220/models/ \
  --model_file /GPT-finetuned-relation-exctraction/logs/2020-05-10__14-47__764220/models/model_epoch-3_dev-macro-f1-0.5660020089794017_dev-loss-4.859414281470467_2020-05-10__14-47__764220.pt \
  --batch_size 8 \
  --log_dir ./logs/
```
