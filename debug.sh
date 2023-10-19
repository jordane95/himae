export CUDA_VISIBLE_DEVICES=2

python finetune.py \
    --model_name_or_path bert-base-uncased \
    --output_dir dir_debug \
    --do_train \
    --do_eval \
    --train_data_dir ../mind/MINDsmall_train \
    --eval_data_dir ../mind/MINDsmall_dev \
    --logging_steps 10 \
    --save_steps 10 \
    --learning_rate 3e-5 \
    --num_train_epochs 3 \
    --user_model other \
    --per_device_eval_batch_size 2 \
    --evaluation_strategy steps \
    --eval_steps 10 \
    --overwrite_output_dir
    # --normalize_news \
    # --normalize_user 
