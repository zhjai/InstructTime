if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

python run_truth_loss.py \
    --dataset har \
    --epoch 100 \
    --lr 0.001 \
    --adapt False >logs/finezero_har.log