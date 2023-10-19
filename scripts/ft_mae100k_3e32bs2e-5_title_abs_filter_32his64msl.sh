export CUDA_VISIBLE_DEVICES=0

pretrained_dir='/data01/lizehan/news/himae/ckpt/news-mae-base-100k1024bs3e-4-ds/checkpoint-100000'
save_dir='ckpt/ft_mae100k_3e32bs2e-5_32his64msl_title_abs_filter'

python finetune.py \
    --model_name_or_path $pretrained_dir \
    --output_dir $save_dir \
    --do_train \
    --do_eval \
    --train_data_dir ../mind/MINDsmall_train \
    --eval_data_dir ../mind/MINDsmall_dev \
    --news_fields title abstract \
    --filter_null_behavior \
    --logging_steps 500 \
    --save_steps 1000 \
    --evaluation_strategy steps \
    --eval_steps 1000 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --user_model mean \
    --max_history_len 32 \
    --news_max_len 32 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --overwrite_output_dir \
    --fp16
    # --normalize_news \
    # --normalize_user 
