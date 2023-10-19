export CUDA_VISIBLE_DEVICES=1,2

pretrained_dir='/data01/lizehan/news/himae/ckpt/news-mae-base-100k1024bs3e-4-ds/checkpoint-100000'

save_dir='ckpt/pt_himae_100k32bs2e-4_8his_large_abs'

# deepspeed run_mae_pretrain.py --deepspeed ds_config.json \

# python run_himae_pretraining.py \

deepspeed run_himae_pretraining.py --deepspeed ds_config.json \
    --model_name_or_path $pretrained_dir \
    --output_dir $save_dir \
    --do_train \
    --do_eval \
    --train_data_dir ../mind/MINDlarge_train \
    --eval_data_dir ../mind/MINDsmall_dev \
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
    --negatives_x_device 
    # --normalize_news \
    # --normalize_user 
