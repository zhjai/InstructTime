import argparse

def get_hyperparams():
    parser = argparse.ArgumentParser(description="Input hyperparams.")

    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_path", type=str, default="./results", help="Directory to save checkpoints and logs.")

    parser.add_argument("--local_model_path", type=str, default="qwen3-0.6b", help="Local path to the base language model.")
    parser.add_argument("--vqvae_root", type=str, default="./vqvae", help="Root directory containing VQ-VAE tokenizers.")
    parser.add_argument("--data_root", type=str, default="./datasets", help="Root directory containing dataset splits.")
    parser.add_argument('--dataset', type=str, default='mix', choices=['har', 'geo', 'sleep', 'mix', 'esr', 'ad', 'dev', 'whale'])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--per_max_token", type=int, default=32, help="The maximum number of tokens for a label.")
    parser.add_argument("--encoder_max_length", type=int, default=230, help="Maximum length of language model input.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--warm_up_ratio", type=float, default=0.05, help="Warm up step for schduler.")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs.")
    parser.add_argument("--adapt", action="store_true", help="If finetune on pretrained model")
    parser.set_defaults(adapt=False)
    parser.add_argument("--pretrained_path", type=str, default="./results", help="Directory containing initial model weights while adapt is true.")

    parser.add_argument("--num_beams", type=int, default=1, help="Number of generation beams.")
    parser.add_argument("--num_return_sequences", type=int, default=1)

    args = parser.parse_args()
    return args
