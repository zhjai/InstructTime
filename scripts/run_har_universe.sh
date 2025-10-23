if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

torchrun --nproc_per_node=2 run_truth_loss.py \
    --dataset har \
    --epoch 100 \
    --lr 5e-5 \
    --model_path results \
    --adapt False >logs/finezero_har.log