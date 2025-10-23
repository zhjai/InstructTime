import argparse  # 解析命令行参数
import os  # 处理路径

import torch  # 保存为 .bin
from safetensors.torch import load_file  # 读取 safetensors 权重


def parse_args():
    parser = argparse.ArgumentParser(
        description="将 safetensors 格式的模型权重转存为 pytorch_model.bin"  # 功能说明
    )
    parser.add_argument(
        "--model-dir",
        default="InstructTime/results/gpt2-har-universe/best_model",
        help="包含 config.json / model.safetensors 的模型目录",  # 默认路径指向当前最佳模型
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="保存 pytorch_model.bin 的目标目录，默认与输入目录相同",  # 允许输出到其他目录
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = args.model_dir
    output_dir = args.output_dir or model_dir  # 默认覆盖原目录

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"未找到模型目录: {model_dir}")  # 输入校验

    os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在

    safetensor_path = os.path.join(model_dir, "model.safetensors")
    if not os.path.isfile(safetensor_path):
        raise FileNotFoundError(f"未找到 model.safetensors: {safetensor_path}")  # 校验权重文件存在

    state_dict = load_file(safetensor_path)  # 读取 safetensors 权重
    target_path = os.path.join(output_dir, "pytorch_model.bin")  # 目标保存路径
    torch.save(state_dict, target_path)  # 直接保存为二进制格式

    print(f"pytorch_model.bin 已保存至: {target_path}")  # 结果提示

if __name__ == "__main__":
    main()
