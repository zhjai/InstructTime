export TOKENIZERS_PARALLELISM=false

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

torchrun --nproc_per_node=2 run_truth_loss.py \
    --dataset har \
    --epochs 100 \
    --lr 0.001 \
    --batch_size 8 \
    --model_path results >logs/finezero_har.log