export CUDA_VISIBLE_DEVICES=1,2

n_gpus=2
pretrained_dir='/data01/lizehan/news/himae/ckpt/news-mae-base-100k1024bs3e-4-ds/checkpoint-100000'
save_dir='ckpt/debug_himae_small_100k32bs2e-4'

# torchrun --nproc_per_node $n_gpus

deepspeed run_himae_pretraining.py --deepspeed ds_config.json \
    --model_name_or_path $pretrained_dir \
    --output_dir $save_dir \
    --do_train \
    --do_eval \
    --train_data_dir ../mind/MINDsmall_train \
    --eval_data_dir ../mind/MINDlarge_dev \
    --filter_null_behavior \
    --news_fields category subcategory title abstract \
    --logging_steps 1000 \
    --save_steps 10000 \
    --evaluation_strategy steps \
    --eval_steps 10000 \
    --learning_rate 2e-4 \
    --max_steps 100000 \
    --user_model mean \
    --news_max_len 128 \
    --max_history_len 8 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --encoder_mlm_probability 0.3 \
    --decoder_mlm_probability 0.5 \
    --n_head_layers 2 \
    --overwrite_output_dir \
    --fp16 \
    --negatives_x_device \
    --gradient_checkpointing False
    # --normalize_news \
    # --normalize_user 
