export CUDA_VISIBLE_DEVICES=0

python finetune.py \
    --model_name_or_path bert-base-uncased \
    --output_dir mind_large_3e3e-5_16bs_4neg \
    --do_train \
    --do_eval \
    --train_data_dir ../mind/MINDlarge_train \
    --eval_data_dir ../mind/MINDlarge_dev \
    --logging_steps 1000 \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --learning_rate 3e-5 \
    --num_train_epochs 3 \
    --user_model mean \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32
    # --normalize_news \
    # --normalize_user 
