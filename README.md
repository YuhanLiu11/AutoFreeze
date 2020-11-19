# AutoFreeze: Automatically Freezing Model Blocks to Accelerate Fine-tuning

This is the code for AutoFreeze, where we develop a system for adaptively freezing transformer blocks of BERT encoder without harming model accuracy, achieving fine-tuning speedup. The code is developed upon the repo: [BERT4doc-Classification](https://github.com/xuyige/BERT4doc-Classification).

## Requirements

+ pandas
+ numpy
+ torch==1.0.1
+ tqdm

## Run the code

### 1) Prepare the dataset & models:

Please follow the original repo to prepare dataset and BERT models.

Data available at: [here](https://drive.google.com/drive/folders/1Rbi0tnvsQrsHvT_353pMdIbRwDlLhfwM).

Some additional datasets available at: [here](https://course.fast.ai/datasets).

Models available at: 

[BERT-Base, Uncased](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip)

[BERT-Base, Chinese](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)

### 2) Fine-tuning with freezing:

Run AutoFreeze with stepped learning rate

```shell
python freeze_intermediate_e2e_lr.py \
--task_name imdb \
--do_train \
--do_eval \
--do_lower_case \
--vocab_file /mnt/uncased_L-12_H-768_A-12/vocab.txt  \
--bert_config_file /mnt/uncased_L-12_H-768_A-12/bert_config.json \
--init_checkpoint /mnt/uncased_L-12_H-768_A-12/pytorch_model.bin  \
--max_seq_length 512  \
--train_batch_size 6  \
--learning_rate 1e-5  \
--num_train_epochs 4.0  \
--output_dir /mnt/output \
--seed 42   \
--gradient_accumulation_steps 1 \
--num_intervals 20 \
--random_seeds 0,1,2,3
```

Run AutoFreeze with caching enabled (if whole dataset fit in CPU memory)

```shell
python freeze_e2e_cache_mem.py \
--task_name imdb \
--do_train \
--do_eval \
--do_lower_case \
--vocab_file /mnt/uncased_L-12_H-768_A-12/vocab.txt  \
--bert_config_file /mnt/uncased_L-12_H-768_A-12/bert_config.json \
--init_checkpoint /mnt/uncased_L-12_H-768_A-12/pytorch_model.bin  \
--max_seq_length 512  \
--train_batch_size 6  \
--learning_rate 1e-5  \
--num_train_epochs 4.0  \
--output_dir /mnt/output \
--seed 42   \
--gradient_accumulation_steps 1 \
--num_intervals 20 \
--random_seeds 0,1,2,3
```


Run AutoFreeze with caching enabled (if whole dataset doesn't fit in CPU memory)

```shell
python freeze_e2e_cache.py \
--task_name imdb \
--do_train \
--do_eval \
--do_lower_case \
--vocab_file /mnt/uncased_L-12_H-768_A-12/vocab.txt  \
--bert_config_file /mnt/uncased_L-12_H-768_A-12/bert_config.json \
--init_checkpoint /mnt/uncased_L-12_H-768_A-12/pytorch_model.bin  \
--max_seq_length 512  \
--train_batch_size 6  \
--learning_rate 1e-5  \
--num_train_epochs 4.0  \
--output_dir /mnt/output \
--seed 42   \
--gradient_accumulation_steps 1 \
--num_intervals 20 \
--random_seeds 0,1,2,3
```
