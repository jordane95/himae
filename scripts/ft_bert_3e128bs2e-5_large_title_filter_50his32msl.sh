export CUDA_VISIBLE_DEVICES=0

pretrained_dir='bert-base-uncased'

save_dir='ckpt/ft_bert_3e128bs2e-5_large_title_filter_50his32msl'

python finetune.py \
    --model_name_or_path $pretrained_dir \
    --output_dir $save_dir \
    --do_train \
    --do_eval \
    --train_data_dir ../mind/MINDlarge_train \
    --eval_data_dir ../mind/MINDlarge_dev \
    --news_fields title \
    --filter_null_behavior \
    --logging_steps 500 \
    --save_steps 1000 \
    --evaluation_strategy steps \
    --eval_steps 1000 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --user_model mean \
    --max_history_len 50 \
    --news_max_len 32 \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --overwrite_output_dir \
    --fp16 \
    --gradient_checkpointing # False
    # --normalize_news \
    # --normalize_user 
