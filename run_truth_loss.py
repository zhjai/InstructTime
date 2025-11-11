import os  # 导入操作系统接口模块
import torch  # 导入 PyTorch 主库
import random  # 导入随机数模块
import logging  # 导入日志记录模块
from logging.handlers import RotatingFileHandler  # 导入循环文件日志处理器
import pickle  # 导入 pickle 序列化工具
import transformers  # 导入 Transformers 库
import numpy as np  # 导入数值计算库 NumPy
import torch.nn as nn  # 导入神经网络模块
from tqdm import tqdm  # 导入进度条工具 tqdm
from torch.utils.data import DataLoader  # 导入数据加载器
from torch.cuda.amp import autocast, GradScaler  # 导入自动混合精度工具
from transformers import AutoModelForCausalLM, AutoConfig  # 导入自适应的语言模型与配置
import torch.distributed as dist  # 导入分布式训练接口
from torch.nn.parallel import DistributedDataParallel as DDP  # 导入分布式数据并行封装
from torch.utils.data.distributed import DistributedSampler  # 导入分布式采样器

from multimodel import InstructTime, MultiTokenizer  # 导入自定义多模态模型与分词器
from multidataset import MultiDataset  # 导入多任务数据集封装
from args import get_hyperparams  # 导入超参数解析函数
from metrics import metric_ecg, metric_eeg, metric_har, metric_fd, metric_rwc  # 导入多任务评估指标
from utils import extract_all_information, load_TStokenizer  # 导入信息抽取与 tokenizer 加载工具

def setup_distributed(device_pref: str):  # 根据设备偏好配置分布式环境
    """分布式初始化
    - 根据环境变量 WORLD_SIZE/LOCAL_RANK 判断是否多进程多卡
    - 初始化进程组，设置当前进程使用的设备
    - 返回 (是否分布式, 全局rank, world_size, 本地rank, 设备)
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))  # 读取世界大小若缺省默认为 1
    if world_size > 1:  # 多于一个进程则启用分布式
        backend = "nccl" if torch.cuda.is_available() else "gloo"  # 根据硬件选择通信后端
        if not dist.is_initialized():  # 检查进程组是否已初始化
            dist.init_process_group(backend=backend)  # 初始化进程组
        rank = dist.get_rank()  # 获取当前进程 rank
        local_rank = int(os.environ.get("LOCAL_RANK", 0))  # 获取当前节点内 rank
        if torch.cuda.is_available():  # 判断 GPU 是否可用
            torch.cuda.set_device(local_rank)  # 设置本地 CUDA 设备
            device = torch.device(f"cuda:{local_rank}")  # 指定 CUDA 设备对象
        else:  # 无 GPU 时
            device = torch.device("cpu")  # 使用 CPU 设备
        return True, rank, world_size, local_rank, device  # 返回分布式相关信息
    # single process
    device = torch.device(device_pref if torch.cuda.is_available() else "cpu")  # 单进程按偏好选择设备
    return False, 0, 1, 0, device  # 返回非分布式状态

def seed_everything(seed):  # 设置随机种子
    """设置随机种子，保证结果可复现"""
    random.seed(seed)  # 固定 Python 原生随机数
    os.environ["PYTHONHASHSEED"] = str(seed)  # 固定哈希随机种子
    np.random.seed(seed)  # 固定 NumPy 随机数
    torch.manual_seed(seed)  # 固定 CPU 上的 PyTorch 随机数
    torch.cuda.manual_seed(seed)  # 固定当前 GPU 随机数
    torch.cuda.manual_seed_all(seed)  # 固定所有 GPU 随机数
    torch.backends.cudnn.deterministic = True  # 设定 cuDNN 为确定性模式
    torch.backends.cudnn.benchmark = False  # 关闭 benchmark 保证可复现
    torch.backends.cudnn.enabled = True  # 启用 cuDNN 加速

def collate_fn_train(batch):  # 训练集批处理函数
    """训练阶段的 batch 拼装函数
    - 将样本中的 input/attn_mask/label_ids 堆叠为张量
    """
    input_ids = [x["input_ids"] for x in batch]  # 收集所有输入 token
    attention_mask = [x["attn_masks"] for x in batch]  # 收集对应注意力掩码
    label_ids = [x["label_ids"] for x in batch]  # 收集训练标签
    return {
        "input_ids": torch.stack(input_ids),  # 将输入堆叠成批量张量
        "attention_mask": torch.stack(attention_mask),  # 将掩码堆叠
        "label_ids": torch.stack(label_ids),  # 将标签堆叠
    }  # 返回批处理后的数据

def collate_fn_test(batch):  # 测试集批处理函数
    """测试/验证阶段的 batch 拼装函数
    - 返回 input/attn_mask 以及原始文本标签，供生成对比
    """
    input_ids = [x["input_ids"] for x in batch]  # 收集输入 token
    attention_mask = [x["attn_masks"] for x in batch]  # 收集注意力掩码
    labels = [x["label"] for x in batch]  # 收集原始标签文本
    return {
        "input_ids": torch.stack(input_ids),  # 将输入堆叠成张量
        "attention_mask": torch.stack(attention_mask),  # 堆叠注意力掩码
        "labels": labels,  # 保留标签列表方便后续文本对比
    }  # 返回批处理结果

def test(model, TestDataLoader, args, logger, out=False):  # 评估模型生成效果
    """评估函数（自回归生成）
    - DDP 下解包出实际模型
    - 对每个样本自回归生成文本，解析结构化标签，计算任务准确率
    - out=True 时返回可读的预测与标签
    """
    # DDP 包装下需要解包出实际模型
    gen_model = model.module if hasattr(model, "module") else model  # 如为 DDP 则取出原始模型
    gen_model.eval()  # 切换至评估模式

    with torch.no_grad():  # 禁用梯度加速推理
        pred_ids, pred_eeg, pred_har, pred_fd, pred_rwc = [], [], [], [], []  # 初始化各模块预测列表
        labels, labels_eeg, labels_har, labels_fd, labels_rwc = [], [], [], [], []  # 初始化各模块标签列表

        all_extracted_info = []  # 保存解析后的预测信息
        all_sig_labels = []  # 保存解析后的标签信息
        debug_print_limit = 10  # 限制打印条数，便于对比预测与标签
        if out:  # 如需输出详细文本
            print_labels = []  # 记录真实标签文本
            print_preds = []  # 记录生成预测文本
        # 遍历评估集，逐 batch 生成
        for data in tqdm(TestDataLoader, desc="Eval", ncols=120):  # 遍历测试集批次
            input_ids = data["input_ids"].to(args.device)  # 将输入搬到目标设备
            bt_labels = data["labels"]  # 取出标签文本
            
            # 自回归生成，beam 或贪心由参数控制
            outputs = gen_model.generate(
                input_ids=input_ids,  # 指定模型输入
                pad_token_id=tokenizer.pad_token_id,  # 指定填充 token
                num_beams=args.num_beams,  # 使用的 beam 数量
                num_return_sequences=args.num_return_sequences,  # 每条输入生成多少序列
                do_sample=False,  # 关闭采样采用确定性搜索
                max_new_tokens=args.per_max_token,  # 限制生成长度
            )  # 执行序列生成
            
            mask = outputs >= tokenizer.text_vocab_size  # 找到超出文本词表的 token
            outputs[mask] = tokenizer.pad_token_id  # 将异常 token 替换为 pad
            outputs = outputs[:, args.encoder_max_length:]  # 去掉编码器输入部分
            decoded_texts = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]  # 转化为可读文本
            if debug_print_limit > 0:
                for pred_text, label_text in zip(decoded_texts, bt_labels):
                    print(f"[Eval] pred: {pred_text}")
                    print(f"[Eval] label: {label_text}")
                    debug_print_limit -= 1
                    if debug_print_limit <= 0:
                        break
            all_extracted_info.extend([extract_all_information(dt) for dt in decoded_texts])  # 解析生成文本
            all_sig_labels.extend([extract_all_information(label) for label in bt_labels])  # 解析标签文本
            if out:  # 若需要导出文本
                print_labels.extend(bt_labels)  # 累计标签文本
                print_preds.extend(decoded_texts)  # 累计预测文本
        
        for decoded_info, sig_label_info in zip(all_extracted_info, all_sig_labels):  # 遍历解析出的预测与标签
            diagnosis_text, stage_text, har_text, fd_text, rwc_text = decoded_info  # 解包预测文本
            diagnosis_label, stage_label, har_label, fd_label, rwc_label = sig_label_info  # 解包标签文本

            if diagnosis_label:  # ECG 诊断任务
                pred_ids.append(diagnosis_text)  # 保存预测
                labels.append(diagnosis_label)  # 保存标签

            elif stage_label:  # EEG 阶段任务
                pred_eeg.append(stage_text)  # 保存预测
                labels_eeg.append(stage_label)  # 保存标签

            elif har_label:  # HAR 动作任务
                pred_har.append(har_text)  # 保存预测
                labels_har.append(har_label)  # 保存标签
            
            elif fd_label:  # FD 任务
                pred_fd.append(fd_text)  # 保存预测
                labels_fd.append(fd_label)  # 保存标签

            elif rwc_label:  # RWC 任务
                pred_rwc.append(rwc_text)  # 保存预测
                labels_rwc.append(rwc_label)  # 保存标签

        res1, res2, res3, res4, res5 = 0, 0, 0, 0, 0  # 初始化各项指标
        if args.dataset == 'mix':  # 混合数据集
            res1, _, _ = metric_ecg(pred_ids, labels, logger)  # 计算 ECG 指标
            res2, _, _ = metric_eeg(pred_eeg, labels_eeg, logger)  # 计算 EEG 指标
            res3, _, _ = metric_har(pred_har, labels_har, logger)  # 计算 HAR 指标
            res4, _, _ = metric_fd(pred_fd, labels_fd, logger)  # 计算 FD 指标
            res5, _, _ = metric_rwc(pred_rwc, labels_rwc, logger)  # 计算 RWC 指标
        elif args.dataset == 'geo':  # 单 ECG 数据集
            res1, _, _ = metric_ecg(pred_ids, labels, logger)  # 计算 ECG 指标
        elif args.dataset == 'eeg':  # 单 EEG 数据集
            res2, _, _ = metric_eeg(pred_eeg, labels_eeg, logger)  # 计算 EEG 指标
        elif args.dataset == 'fd':  # 单 FD 数据集
            res3, _, _ = metric_fd(pred_fd, labels_fd, logger)  # 计算 FD 指标
        elif args.dataset == 'rwc':  # 单 RWC 数据集
            res5, _, _ = metric_rwc(pred_rwc, labels_rwc, logger)  # 计算 RWC 指标
        else:  # 默认 HAR 数据集
            res4, _, _ = metric_har(pred_har, labels_har, logger)  # 计算 HAR 指标

    if out:  # 若请求输出文本
        return print_preds, print_labels  # 返回预测文本与标签
    else:  # 否则返回指标
        return res1 + res2 + res3 + res4 + res5  # 返回所有指标之和

def setup_logging(run_path):  # 初始化日志配置
    """
    初始化日志记录器：写入 run_path/log.log，INFO 级别
    """
    log_file = os.path.join(run_path, "log.log")  # 拼接日志文件路径


    open(log_file, 'w').close()  # 清空日志文件
    logger = logging.getLogger('training_log')  # 获取或创建训练日志记录器
    logger.setLevel(logging.INFO)  # 设置日志级别为 INFO

    file_handler = RotatingFileHandler(log_file, maxBytes=1024*1024*5, backupCount=2)  # 配置循环日志写入
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")  # 定义日志格式
    file_handler.setFormatter(formatter)  # 绑定格式器

    logger.addHandler(file_handler)  # 将处理器添加到记录器

    return logger  # 返回配置好的日志对象

def initialize_model(args, tokenizer, TStokenizers):  # 构建并初始化模型
    """构建并初始化模型
    - 从本地 Qwen 配置与权重加载文本侧参数
    - 替换输出头为“文本 + HAR 离散 token”的总词表
    - 同步 config.vocab_size 以匹配新的输出维度
    """
    config = AutoConfig.from_pretrained(args.local_model_path)  # 从本地加载 Qwen 配置
    base_model = AutoModelForCausalLM.from_config(config)  # 根据配置构建骨干模型
    model = InstructTime(base_model, TStokenizers, text_embedding=len(tokenizer.textTokenizer)).to(args.device)  # 初始化多模态模型并放至设备

    torch_dtype = config.torch_dtype
    if isinstance(torch_dtype, str):
        torch_dtype = getattr(torch, torch_dtype)
    pretrained_model = AutoModelForCausalLM.from_pretrained(args.local_model_path, torch_dtype=torch_dtype, low_cpu_mem_usage=False)  # 加载预训练 Qwen 权重
    model.load_state_dict(pretrained_model.state_dict(), strict=False)  # 非严格方式加载到多模态模型

    # 先扩展词表（仅文本侧）以适配新增的特殊符号等
    model.resize_token_embeddings(len(tokenizer.textTokenizer))  # 扩展文本嵌入矩阵尺寸
    current_output = model.get_output_embeddings()  # 获取当前输出层
    # 替换输出头为“文本 + 时间序列离散 token”的总大小
    new_output = nn.Linear(
        config.hidden_size,
        tokenizer.vocabSize_all(),
        bias=False,
    ).to(device=args.device, dtype=current_output.weight.dtype)  # 构建新的输出层
    new_output.weight.data[:len(tokenizer.textTokenizer)] = current_output.weight.data  # 保留原有文本部分权重
    model.set_output_embeddings(new_output)  # 替换输出层
    # 同步配置中的词表大小，避免损失 reshape 报错
    model.config.vocab_size = tokenizer.vocabSize_all()  # 同步配置中的词表大小
    
    return model  # 返回模型实例

def train_model(model, args, TrainDataLoader, TestDataLoader, optimizer, scheduler, scaler, logger, run_path, distributed=False, is_main_process=True):  # 训练主循环
    """训练主循环（支持单机多卡 DDP）
    - 前向：自回归目标（传入 labels 触发因果交叉熵）
    - 评估：每个 epoch 结束后生成 + 计算准确率
    - 早停：patience=5，连续未提升即停止
    - DDP：仅 rank0 负责日志/保存/评估，其余 rank 同步停止标志
    """
    best = 0.0  # 记录目前最佳准确率
    tolerance_metric = -float("inf")  # 记录用于早停的对比指标
    patience = 10  # 设定早停耐心轮数
    wait = 0  # 已连续未提升的轮数
        
    # 设置采样器
    train_sampler = getattr(TrainDataLoader, "sampler", None)  # 获取训练数据采样器
    model_dtype = getattr(model, "model_dtype", getattr(getattr(model, "module", None), "model_dtype", torch.float32))

    for epoch in range(args.epochs):  # 遍历训练轮
        if distributed and isinstance(train_sampler, DistributedSampler):  # 分布式情况下重设采样器
            # 设定 epoch 保证各进程采样不同切片
            train_sampler.set_epoch(epoch)  # 将 epoch 传入采样器
        step, train_losses = 0, 0.0  # 初始化步数和损失
        tqdm_iter = tqdm(TrainDataLoader, desc=f"Qwen Epoch {epoch+1}", ncols=120, disable=not is_main_process)  # 主进程显示进度条
        
        model.train()  # 切换模型为训练模式
        amp_dtype = torch.float16 if model_dtype == torch.float16 else None
        current_lr = optimizer.param_groups[0]["lr"]
        for data in tqdm_iter:  # 遍历每个批次

            input_ids = data["input_ids"].to(args.device)  # 将输入转移到目标设备
            attention_mask = data["attention_mask"].to(args.device)  # 将注意力掩码转移到目标设备
            label_ids = data["label_ids"].to(args.device)  # 将标签转移到目标设备
            
            # 混合精度前向与损失计算
            autocast_kwargs = {"enabled": torch.cuda.is_available() and amp_dtype is not None}
            if amp_dtype is not None:
                autocast_kwargs["dtype"] = amp_dtype
            with autocast(**autocast_kwargs):  # 启用混合精度
                outputs = model(
                            input_ids=input_ids,  # 输入 token 张量
                            attention_mask=attention_mask,  # 输入掩码
                            labels=label_ids  # 输入标签触发语言建模损失
                            )  # 前向计算输出
            
            # 反向与优化器更新
            scaler.scale(outputs.loss).backward()  # 放缩损失并反向传播
            scaler.step(optimizer)  # 执行优化步
            scaler.update()  # 更新放缩因子
            scheduler.step()  # 更新学习率
            current_lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]["lr"]
            optimizer.zero_grad()  # 清空梯度

            loss_value = outputs.loss.cpu().item()  # 将损失搬到 CPU 取标量
            train_losses += loss_value  # 累积损失
            step += 1  # 统计步数
            if is_main_process:  # 主进程更新进度信息
                tqdm_iter.set_postfix({
                    "loss": format(train_losses / step, ".4f"),
                    "lr": f"{current_lr:.2e}",
                })  # 显示当前平均损失和学习率
                logger.info(
                    f"Epoch {epoch+1} Step {step}: loss={loss_value:.6f}, lr={current_lr:.6e}"
                )

        final_loss = format(train_losses / step, ".4f")  # 计算最终平均损失
        if is_main_process:  # 仅主进程记录
            logger.info(f"Epoch {epoch+1}\nLoss: {final_loss}; lr={current_lr:.6e}")  # 写入损失日志
        
        # 评估前同步各进程，rank0 执行评估，随后广播分数
        if distributed:  # 分布式模式下同步
            dist.barrier()  # 等待所有进程完成本轮训练
        if is_main_process:  # 主进程执行评估
            res = test(model, TestDataLoader, args, logger, out=False)  # 运行测试获取指标
        else:  # 其他进程占位
            res = None  # 临时存放结果
        if distributed:  # 分布式需同步指标
            obj = [res]  # 将指标封装成列表
            dist.broadcast_object_list(obj, src=0)  # 广播主进程的指标
            res = obj[0]  # 解包同步后的结果
        if is_main_process:  # 主进程打印与记录
            print(f"Epoch {epoch+1} accuracy: {res:.4f}")  # 打印准确率
            logger.info(f"Epoch {epoch+1} accuracy: {res:.4f}")  # 写入日志

        if res > best:  # 若指标刷新最佳
            MODEL_STORED_PATH = run_path + "/best_model"  # 构造最佳模型存储路径
            best = res  # 更新最佳值
            if is_main_process:  # 仅主进程保存模型
                model_to_save = model.module if hasattr(model, "module") else model  # 取出原始模型
                model_to_save.save_pretrained(MODEL_STORED_PATH)  # 保存模型权重与配置
        if res > tolerance_metric:  # 若指标优于早停阈值
            tolerance_metric = res  # 更新阈值
            wait = 0  # 重置等待计数
        else:  # 指标未提升
            wait += 1  # 增加等待次数
            stop = wait >= patience  # 判断是否达到早停条件
            if distributed:  # 分布式需同步停止信号
                flag = [stop]  # 创建停止标记
                dist.broadcast_object_list(flag, src=0)  # 广播给所有进程
                stop = flag[0]  # 更新当前进程状态
            if stop:  # 满足早停条件
                if is_main_process:  # 主进程记录日志
                    logger.info(f"Early stopping at epoch {epoch+1}")  # 写入早停信息
                break  # 退出训练循环

if __name__ == "__main__":  # 程序入口
    args = get_hyperparams()  # 解析运行超参数
    seed_everything(args.seed)  # 固定随机种子
    # init distributed
    distributed, rank, world_size, local_rank, device = setup_distributed(args.device)  # 初始化分布式环境
    args.device = device  # 更新配置中的设备
    is_main_process = (rank == 0)  # 标记是否为主进程

    
    dataset_key = args.dataset.lower()
    alias_map = {"sleep": "eeg", "dev": "fd", "whale": "rwc"}
    dataset_key = alias_map.get(dataset_key, dataset_key)

    DATASET_CONFIG = {
        "geo": {
            "data_subdirs": ["ecg_no_big", "ECG", "ecg", "GEO", "geo"],
            "vqvae_subdirs": ["test_ecg_64_128_40", "ECG", "ecg", "GEO", "geo"],
            "prefix": "You will be receiving electrocardiogram(ECG) related signals.\n",
        },
        "eeg": {
            "data_subdirs": ["eeg_no_big", "EEG", "eeg"],
            "vqvae_subdirs": ["test_eeg_64_256_25", "EEG", "eeg"],
            "prefix": "You will be receiving electroencephalogram(EEG) related signals.\n",
        },
        "fd": {
            "data_subdirs": ["device_no_big", "FD"],
            "vqvae_subdirs": ["test_fd_64_512_40", "FD"],
            "prefix": "You will be receiving industrial equipment related signals.\n",
        },
        "har": {
            "data_subdirs": ["har_no_big", "HAR"],
            "vqvae_subdirs": ["test_har_64_256_1", "HAR"],
            "prefix": "You will be receiving human physical activities related signals.\n",
        },
        "rwc": {
            "data_subdirs": ["rwc_no_big", "RWC"],
            "vqvae_subdirs": ["test_rwc_64_384_32", "RWC"],
            "prefix": "You will be receiving sound related signals.\n",
        },
    }
    MIX_ORDER = ["geo", "eeg", "har", "fd", "rwc"]
    MIX_PREFIX = ("You will be receiving signals from five domains: electrocardiogram, "
                  "electroencephalogram, industrial equipment, sound and physical activities.\n")

    def extract_text_signal(sample):
        if isinstance(sample, dict):
            text = sample.get("text", "")
            signal = sample.get("ts")
        elif isinstance(sample, (list, tuple)) and len(sample) >= 2:
            text = sample[0]
            signal = sample[1]
        else:
            raise ValueError("Unsupported sample format; expected dict or tuple/list with text and signal")
        if signal is None:
            raise ValueError("Signal tensor missing in sample; ensure dataset format is (text, ts, label) or dict with 'ts'.")
        return text, signal

    def count_vl(samples):
        return sum(
            1
            for s in samples
            if isinstance(s, dict) and isinstance(s.get("vl_response"), str) and s["vl_response"].strip()
        )

    def load_dataset_bundle(key):
        if key not in DATASET_CONFIG:
            raise ValueError(f"Unsupported dataset '{key}'. Available: {list(DATASET_CONFIG.keys())}")
        info = DATASET_CONFIG[key]
        data_dir = None
        train_path = test_path = None
        for candidate in info["data_subdirs"]:
            candidate_dir = os.path.join(args.data_root, candidate)
            candidate_train = os.path.join(candidate_dir, "samples_train.pkl")
            candidate_test = os.path.join(candidate_dir, "samples_test.pkl")
            if os.path.isfile(candidate_train) and os.path.isfile(candidate_test):
                data_dir = candidate_dir
                train_path = candidate_train
                test_path = candidate_test
                break
        if data_dir is None:
            searched = [os.path.join(args.data_root, cand) for cand in info["data_subdirs"]]
            raise FileNotFoundError(
                f"Missing dataset split for '{key}'. Checked directories: {searched}."
            )
        with open(train_path, 'rb') as f:
            train_samples = pickle.load(f)
        with open(test_path, 'rb') as f:
            test_samples = pickle.load(f)
        text_example, signal = extract_text_signal(train_samples[0])
        total_msg = (
            f"{key.upper()} samples: total {len(train_samples) + len(test_samples)} "
            f"(train {len(train_samples)}, test {len(test_samples)})"
        )
        print(total_msg)
        print(text_example)
        log_messages.append(total_msg)
        log_messages.append(f"{key.upper()} train file: {train_path}")
        log_messages.append(f"{key.upper()} test file: {test_path}")
        log_messages.append(f"{key.upper()} example text: {text_example}")
        vl_train = count_vl(train_samples)
        vl_test = count_vl(test_samples)
        if vl_train or vl_test:
            log_messages.append(
                f"{key.upper()} train samples with vl_response: {vl_train}/{len(train_samples)}"
            )
            log_messages.append(
                f"{key.upper()} test samples with vl_response: {vl_test}/{len(test_samples)}"
            )
        tokenizer_path = None
        for candidate in info["vqvae_subdirs"]:
            candidate_path = os.path.join(args.vqvae_root, candidate)
            if os.path.isdir(candidate_path):
                tokenizer_path = candidate_path
                break
        if tokenizer_path is None:
            searched_tok = [os.path.join(args.vqvae_root, cand) for cand in info["vqvae_subdirs"]]
            raise FileNotFoundError(
                f"Tokenizer path not found for '{key}'. Checked directories: {searched_tok}."
            )

        return {
            "key": key,
            "train": train_samples,
            "test": test_samples,
            "text": text_example,
            "signal": signal,
            "prefix": info["prefix"],
            "tokenizer_path": tokenizer_path,
            "train_path": train_path,
            "test_path": test_path,
        }

    if dataset_key == "mix":
        selected_keys = MIX_ORDER
    else:
        selected_keys = [dataset_key]

    log_messages = []
    bundles = [load_dataset_bundle(key) for key in selected_keys]

    example_inputs = [(bundle["key"], bundle["text"]) for bundle in bundles]

    if dataset_key == "mix":
        samples_train_combined = []
        samples_test_combined = []
        for bundle in bundles:
            samples_train_combined.extend(bundle["train"])
            samples_test_combined.extend(bundle["test"])
        random.shuffle(samples_train_combined)
        random.shuffle(samples_test_combined)
        PREFIX_TEXT = MIX_PREFIX
    else:
        bundle = bundles[0]
        samples_train_combined = bundle["train"]
        samples_test_combined = bundle["test"]
        PREFIX_TEXT = bundle["prefix"]

    print('preprocess done')

    TStokenizers = []
    for bundle in bundles:
        tokenizer_path = bundle["tokenizer_path"]
        TStokenizers.append(load_TStokenizer(tokenizer_path, bundle["signal"].shape, 'cpu'))
        log_messages.append(f"{bundle['key'].upper()} tokenizer path: {tokenizer_path}")

    tokenizer = MultiTokenizer(TStokenizers, args.local_model_path)

    TrainDataset = MultiDataset(
        samples_train_combined,  # 训练数据
        tokenizer,  # 分词器
        mode="train",  # 指定模式
        encoder_max_length=args.encoder_max_length,  # 最大编码长度
        multi=args.dataset,  # 数据集标识
        prefix_text=PREFIX_TEXT,  # 前缀文本
    )  # 构建训练数据集
    # 如果是分布式训练则使用 DistributedSampler
    train_sampler = DistributedSampler(TrainDataset, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None  # 分布式采样器
    TrainDataLoader = DataLoader(
        TrainDataset,  # 数据集
        batch_size=args.batch_size,  # 批大小
        shuffle=not distributed,  # 非分布式时随机打乱
        sampler=train_sampler,  # 分布式采样器
        num_workers=16,  # 线程数
        collate_fn=collate_fn_train,  # 批处理函数
    )  # 构建训练数据加载器
    TestDataset = MultiDataset(
        samples_test_combined,  # 测试数据
        tokenizer,  # 分词器
        mode="test",  # 指定测试模式
        encoder_max_length=args.encoder_max_length,  # 最大编码长度
        multi=args.dataset,  # 数据集标识
        prefix_text=PREFIX_TEXT,  # 前缀文本
    )  # 构建测试数据集
    TestDataLoader = DataLoader(
        TestDataset,  # 测试集
        batch_size=args.batch_size,  # 批大小
        shuffle=False,  # 关闭随机打乱
        num_workers=16,  # 线程数
        collate_fn=collate_fn_test,  # 批处理函数
    )  # 构建测试数据加载器

    num = 1  # 指定运行次数
    for run in range(num):  # 遍历每次运行
        model = initialize_model(args, tokenizer, TStokenizers)  # 初始化模型实例
        if args.adapt:  # 若启用适配模式
            state_file = os.path.join(args.pretrained_path, 'pytorch_model.bin')  # 预训练权重文件
            if not os.path.isfile(state_file):
                raise FileNotFoundError(f"未找到预训练权重: {state_file}")  # 明确缺失提示
            model_state_dict = torch.load(state_file, map_location=args.device)  # 加载预训练权重
            model.load_state_dict(model_state_dict, strict=False)  # 以非严格模式加载
        if distributed:  # 若处于分布式模式
            device_ids = [local_rank] if device.type == "cuda" else None  # 指定使用的 GPU
            output_device = local_rank if device.type == "cuda" else None  # 指定输出设备
            model = DDP(model, device_ids=device_ids, output_device=output_device, find_unused_parameters=False)  # 封装为分布式数据并行
        model_subpath = args.save_path  # 组合模型保存路径
        if is_main_process:  # 主进程输出调试信息
            print(f"Saving outputs under {model_subpath}")  # 打印路径

        os.makedirs(model_subpath, exist_ok=True)  # 确保模型目录存在
        run_path = os.path.join(model_subpath, f"run_{run}")  # 每次运行的存储目录
        os.makedirs(run_path, exist_ok=True)  # 创建运行目录
        logger = setup_logging(run_path) if is_main_process else logging.getLogger(f"rank_{rank}")  # 主进程创建日志记录器
        if is_main_process:
            for msg in log_messages:
                logger.info(msg)

        for param in model.parameters():  # 遍历模型参数
            param.requires_grad = True  # 确保参数可训练

        # baseline evaluation only once on main process
        if distributed:
            dist.barrier()
        if is_main_process:
            baseline = test(model, TestDataLoader, args, logger, out=False)
            print(f"Pre-training accuracy: {baseline:.4f}")
            logger.info(f"Pre-training accuracy: {baseline:.4f}")
        else:
            baseline = None
        if distributed:
            obj = [baseline]
            dist.broadcast_object_list(obj, src=0)
            baseline = obj[0]
            
        model_dtype = getattr(model, "model_dtype", getattr(getattr(model, "module", None), "model_dtype", torch.float32))

        param_dict = [{"params": model.parameters(), "lr": args.lr}]  # 构建优化器参数组
        optimizer = torch.optim.Adam(param_dict, weight_decay=1e-5)  # 初始化 Adam 优化器
        scheduler = transformers.optimization.get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=args.epochs * len(TrainDataLoader) * args.warm_up_ratio, num_training_steps=args.epochs * len(TrainDataLoader)
        )  # 构建余弦退火调度
        scaler = GradScaler(enabled=torch.cuda.is_available() and model_dtype == torch.float16)  # 创建混合精度缩放器（仅在 fp16 时启用）

        if is_main_process:  # 主进程记录日志
            logger.info(f"Begin training for run {run}")  # 记录训练开始信息
        train_model(model, args, TrainDataLoader, TestDataLoader, optimizer, scheduler, scaler, logger, run_path, distributed=distributed, is_main_process=is_main_process)  # 启动训练

        # 仅主进程执行评估并保存结果
        if is_main_process:  # 主进程负责评估与导出
            eval_model = initialize_model(args, tokenizer, TStokenizers)  # 初始化评估模型
            best_model_path = os.path.join(run_path, 'best_model')  # 最佳模型存放目录
            state_path = os.path.join(best_model_path, 'pytorch_model.bin')  # 最佳权重文件
            if os.path.exists(state_path):  # 检查最佳模型是否存在
                model_state_dict = torch.load(state_path, map_location=args.device)  # 加载最佳权重
                eval_model.load_state_dict(model_state_dict)  # 加载到评估模型
                logger.info(f"Test best model for run {run}")  # 记录评估信息
                print_preds, print_labels = test(eval_model, TestDataLoader, args, logger, out=True)  # 获取预测与标签文本

                save_path = os.path.join(run_path, 'output.txt')  # 输出文件路径
                with open(save_path, 'w', encoding='utf-8') as file:  # 打开输出文件
                    for name, example_text in example_inputs:
                        file.write(f"Input Sequence ({name.upper()}): \n{PREFIX_TEXT + example_text}\n")
                        file.write('\n')

                    limit = min(500, len(print_labels))
                    for i in range(limit):
                        j = i * args.num_return_sequences
                        for k in range(args.num_return_sequences):
                            file.write(f"Generated Text: {print_preds[j + k]}\n")
                        file.write(f"Actual Label: {print_labels[i]}\n")
                        file.write('\n')


                logger.handlers.clear()  # 清理日志处理器避免重复添加
