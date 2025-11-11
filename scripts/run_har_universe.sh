torchrun --nproc_per_node=2 run_truth_loss.py \
    --dataset har \
    --lr 1e-3 \
    --batch_size 8 \
    --epochs 100 \
    --save_path results/qwen3-0.6b-har-universe-adam