export TOKENIZERS_PARALLELISM=false

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

torchrun --nproc_per_node=2 run_truth_loss.py \
    --dataset har \
    --epochs 100 \
    --lr 1e-6 \
    --batch_size 8 \
    --model_path results \
    --load_model_path results/qwen3-0.6b-universe/best_model \
    --adapt >logs/finezero_har.log