#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ../src/train_bash.py \
    --stage sft \
    --model_name_or_path ../chatglm2-6b \
    --do_train \
    --dataset lima \
    --dataset_dir ../data \
    --finetuning_type lora \
    --output_dir ../sft-ckp \
    --overwrite_cache \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --save_steps 20000 \
    --warmup_steps 0 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --fp16 \
    # --accelerate config 10 \
