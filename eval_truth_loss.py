import argparse  # 解析命令行参数
import logging  # 简单日志输出
import os  # 路径处理
import pickle  # 读取样本缓存
import re  # 提取思考/答案
from typing import Optional  # 可选类型提示

import torch  # 加载模型权重
from torch.nn.parallel import DistributedDataParallel as DDP  # 多卡封装
from torch.utils.data import DataLoader  # 构建数据加载器
from torch.utils.data.distributed import DistributedSampler  # DDP 采样器

import pandas as pd  # 导出结果

import run_truth_loss as rtl  # 复用训练脚本中的工具函数


def build_arg_parser():
    parser = argparse.ArgumentParser(description="评估 InstructTime HAR 模型性能")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="包含 pytorch_model.bin 的目录或文件路径")
    parser.add_argument("--data_path", type=str, default="datasets/HAR")
    parser.add_argument("--local_model_path", type=str, default="/data/zhjustc/InstructTime/qwen3-0.6b")
    parser.add_argument("--vqvae_path", type=str, default="./vqvae/HAR")
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
    parser.add_argument(
        "--export_path",
        type=str,
        default=None,
        help="若指定，将额外导出包含 prompt / groundtruth / think / answer 的 CSV 或 XLSX",
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


def load_datasets(data_path: str):
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"HAR 数据集未找到：{data_path}")

    def pick(split: str) -> str:
        split_key = split.lower()
        candidates = [
            fname
            for fname in os.listdir(data_path)
            if fname.lower().endswith(".pkl") and split_key in fname.lower()
        ]
        if not candidates:
            raise FileNotFoundError(f"在 {data_path} 未找到包含 '{split}' 的 PKL 文件")
        if len(candidates) > 1:
            raise ValueError(f"在 {data_path} 找到多个 '{split}' PKL 文件：{candidates}")
        return os.path.join(data_path, candidates[0])

    train_path = pick("train")
    test_path = pick("test")
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


THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def split_think_answer(text: str):
    if not isinstance(text, str):
        return "", ""
    think_match = THINK_PATTERN.search(text)
    answer_match = ANSWER_PATTERN.search(text)
    think = think_match.group(1).strip() if think_match else ""
    answer = answer_match.group(1).strip() if answer_match else text.strip()
    return think, answer


def main():
    args = build_arg_parser().parse_args()

    rtl.seed_everything(args.seed)
    distributed, rank, world_size, local_rank, device = rtl.setup_distributed(args.device)
    is_main = rank == 0

    logger = prepare_logger(args.log_file if is_main else None)
    logger.info("开始加载数据与模型")

    samples_train, samples_test = load_datasets(args.data_path)
    text_har, har, _ = samples_train[0]

    tokenizer_har = rtl.load_TStokenizer(args.vqvae_path, har.shape, "cpu")
    tokenizer = rtl.MultiTokenizer([tokenizer_har], args.local_model_path)
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
        local_model_path=args.local_model_path,
    )

    model = rtl.initialize_model(eval_args, tokenizer, [tokenizer_har])
    state_path = locate_state_dict(args.checkpoint_path)
    state_dict = torch.load(state_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)

    if distributed:
        device_ids = [local_rank] if device.type == "cuda" else None
        output_device = local_rank if device.type == "cuda" else None
        model = DDP(model, device_ids=device_ids, output_device=output_device, find_unused_parameters=False)

    logger.info("开始评估")
    raw_preds, raw_labels = rtl.test(model, test_loader, eval_args, logger, out=True)

    if is_main:
        # 计算准确率
        pred_answers = [rtl.extract_all_information(p)[2] for p in raw_preds]
        label_answers = [rtl.extract_all_information(l)[2] for l in raw_labels]
        score, _, _ = rtl.metric_har(pred_answers, label_answers, logger)
        logger.info(f"评估得分: {score:.4f}")
        print("示例输入:", text_har)

        if args.export_path:
            prompts = []
            for text, _, _ in samples_test:
                idx = text.find("information.\n")
                if idx != -1:
                    prompt = text[: idx + len("information.\n")]
                else:
                    prompt = text
                prompts.append(prompt.strip())

            records = []
            total = min(len(prompts), len(raw_labels), len(raw_preds))
            for i in range(total):
                think, answer = split_think_answer(raw_preds[i])
                records.append(
                    {
                        "prompt": prompts[i],
                        "groundtruth": raw_labels[i],
                        "thinking": think,
                        "generated_answer": answer,
                        "prediction_raw": raw_preds[i],
                    }
                )

            df = pd.DataFrame(records)
            export_path = args.export_path
            os.makedirs(os.path.dirname(export_path) or ".", exist_ok=True)
            if export_path.lower().endswith(".xlsx"):
                df.to_excel(export_path, index=False)
            elif export_path.lower().endswith(".csv"):
                df.to_csv(export_path, index=False)
            else:
                logger.warning("未识别的导出格式，请使用 .csv 或 .xlsx 后缀")


if __name__ == "__main__":
    main()
