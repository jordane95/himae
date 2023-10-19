export CUDA_VISIBLE_DEVICES=3

pretrained_dir='/data01/lizehan/news/himae/ckpt/himae_large100k32bs2e-4/checkpoint-100000'
save_dir='ckpt/ft_himae100k_3e32bs2e-5'

python finetune.py \
    --model_name_or_path $pretrained_dir \
    --output_dir $save_dir \
    --do_train \
    --do_eval \
    --train_data_dir ../mind/MINDsmall_train \
    --eval_data_dir ../mind/MINDsmall_dev \
    --logging_steps 500 \
    --save_steps 1000 \
    --evaluation_strategy steps \
    --eval_steps 1000 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --user_model mean \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --overwrite_output_dir \
    --fp16
    # --normalize_news \
    # --normalize_user 
