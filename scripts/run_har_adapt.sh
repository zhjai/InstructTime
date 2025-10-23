if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

torchrun --nproc_per_node=2 run_truth_loss.py \
    --dataset har \
    --epoch 100 \
    --lr 1e-5 \
    --model_path results \
    --load_model_path results/gpt2-har-universe/best_model \
    --adapt True >logs/finezero_har.log