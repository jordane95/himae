
export CUDA_VISIBLE_DEVICES=0,1
save_dir="ckpt/news-mae-base-100k1024bs3e-4-ds" 

n_gpus=2

# python -m torch.distributed.launch --nproc_per_node $n_gpus run.py \

deepspeed run_mae_pretrain.py --deepspeed ds_config.json \
    --output_dir $save_dir \
    --data_dir pretrain_data \
    --do_train True \
    --save_steps 10000 \
    --per_device_train_batch_size 256 \
    --gradient_accumulation_steps 2 \
    --max_seq_length 128 \
    --model_name_or_path bert-base-uncased \
    --fp16 \
    --gradient_checkpointing False \
    --learning_rate 3e-4 \
    --warmup_steps 5000 \
    --max_steps 100000 \
    --overwrite_output_dir True \
    --dataloader_num_workers 32 \
    --weight_decay 0.01 \
    --encoder_mlm_probability 0.3 \
    --decoder_mlm_probability 0.5 \
    --n_head_layers 2
