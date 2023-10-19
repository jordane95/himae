export CUDA_VISIBLE_DEVICES=0

python finetune.py \
    --model_name_or_path debug/checkpoint-57645 \
    --output_dir debug \
    --do_eval True \
    --train_data_dir ../mind/MINDsmall_train \
    --eval_data_dir ../mind/MINDsmall_dev \
    --logging_steps 10 \
    --save_steps 10 \
    --learning_rate 3e-5 \
    --num_train_epochs 3 \
    --user_model other \
    --per_device_eval_batch_size 2 \
    --evaluation_strategy steps \
    --eval_steps 10
    # --normalize_news \
    # --normalize_user 
