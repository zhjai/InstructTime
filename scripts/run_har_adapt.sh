python run_truth_loss.py \
    --dataset har \
    --lr 1e-5 \
    --epochs 50 \
    --model_path results/gpt2-har-adapt \
    --load_model_path results/gpt2-har-universe/no_frozen/run_0/best_model \
    --adapt True