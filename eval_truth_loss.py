import argparse  # 解析命令行参数
import logging  # 简单日志输出
import os  # 路径处理
import pickle  # 读取样本缓存
from typing import Optional  # 可选类型提示

import torch  # 加载模型权重
from torch.nn.parallel import DistributedDataParallel as DDP  # 多卡封装
from torch.utils.data import DataLoader  # 构建数据加载器
from torch.utils.data.distributed import DistributedSampler  # DDP 采样器

import run_truth_loss as rtl  # 复用训练脚本中的工具函数


def build_arg_parser():
    parser = argparse.ArgumentParser(description="评估 InstructTime HAR 模型性能")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--load_model_path",
        type=str,
        required=True,
        help="包含 pytorch_model.bin 的目录或文件路径",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--per_max_token", type=int, default=32)
    parser.add_argument("--encoder_max_length", type=int, default=230)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="可选：将评估日志写入指定文件",
    )
    return parser


def prepare_logger(log_file: Optional[str]):
    logger = logging.getLogger("eval_truth_loss")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(log_file) if log_file else logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def load_datasets():
    file_path = "datasets/HAR"
    train_path = os.path.join(file_path, "samples_train.pkl")
    test_path = os.path.join(file_path, "samples_test.pkl")
    if not (os.path.isfile(train_path) and os.path.isfile(test_path)):
        raise FileNotFoundError(f"HAR 数据集未找到：{file_path}")
    with open(train_path, "rb") as file:
        samples_train = pickle.load(file)
    with open(test_path, "rb") as file:
        samples_test = pickle.load(file)
    return samples_train, samples_test


def locate_state_dict(path: str) -> str:
    if os.path.isdir(path):
        candidate = os.path.join(path, "pytorch_model.bin")
        if os.path.isfile(candidate):
            return candidate
    if os.path.isfile(path):
        return path
    raise FileNotFoundError(f"未找到模型权重文件: {path}")


def main():
    args = build_arg_parser().parse_args()

    rtl.seed_everything(args.seed)
    distributed, rank, world_size, local_rank, device = rtl.setup_distributed(args.device)
    is_main = rank == 0

    logger = prepare_logger(args.log_file if is_main else None)
    logger.info("开始加载数据与模型")

    samples_train, samples_test = load_datasets()
    text_har, har, _ = samples_train[0]

    tokenizer_har = rtl.load_TStokenizer(rtl.vqvae_path, har.shape, "cpu")
    tokenizer = rtl.MultiTokenizer([tokenizer_har])
    rtl.tokenizer = tokenizer  # test 函数依赖模块级 tokenizer

    test_dataset = rtl.MultiDataset(
        samples_test,
        tokenizer,
        mode="test",
        encoder_max_length=args.encoder_max_length,
        multi="har",
        prefix_text="You will be receiving human physical activities related signals.\n",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=16,
        collate_fn=rtl.collate_fn_test,
        sampler=DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if distributed
        else None,
    )

    eval_args = argparse.Namespace(
        device=device,
        num_beams=args.num_beams,
        num_return_sequences=args.num_return_sequences,
        per_max_token=args.per_max_token,
        encoder_max_length=args.encoder_max_length,
        dataset="har",
    )

    model, _ = rtl.initialize_model(eval_args, tokenizer, [tokenizer_har])
    state_path = locate_state_dict(args.load_model_path)
    state_dict = torch.load(state_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)

    if distributed:
        device_ids = [local_rank] if device.type == "cuda" else None
        output_device = local_rank if device.type == "cuda" else None
        model = DDP(model, device_ids=device_ids, output_device=output_device, find_unused_parameters=False)

    logger.info("开始评估")
    score = rtl.test(model, test_loader, eval_args, logger, out=False)
    if is_main:
        logger.info(f"评估得分: {score:.4f}")
        print("示例输入:", text_har)


if __name__ == "__main__":
    main()
