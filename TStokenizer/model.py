import torch  # 导入 PyTorch 主模块
from torch import nn  # 导入神经网络模块
from torch.nn import functional as F  # 导入常用函数式接口
from torch.nn.init import xavier_normal_, constant_  # 导入权重初始化方法

"""
模型文件概览（中文注释版）：
 - TCN：时间卷积网络骨干，用于抓取序列全局上下文；
 - ResidualBlock_b：带空洞卷积与残差连接的基本模块；
 - Quantize：向量量化层，采用 EMA 更新码本（VQ-VAE 风格）；
 - Encoder / Decoder：对输入/输出做线性映射并堆叠 TCN；
 - TStokenizer：完整的时间序列 tokenizer，负责编码->分块->量化->重建，
   并返回离散 token id 与重建损失。

TCN 实现基于 RecBole 项目修改
################################################

参考代码：
    - https://github.com/fajieyuan/nextitnet
    - https://github.com/initlisk/nextitnet_pytorch

符号约定：
    B: batch 大小
    L: 序列长度（时间步数）
    D: 隐藏维度 / 特征维度
    C: 原始信号通道数
"""

class TCN(nn.Module):
    def __init__(self, args=None, **kwargs):
        super(TCN, self).__init__()  # 初始化父类

        # 支持两种参数传入方式：命名空间 args 或关键字参数 kwargs
        if args is not None:
            d_model = args.d_model  # 隐藏维度
            self.embedding_size = args.d_model  # 嵌入维度
            self.residual_channels = args.d_model  # 残差通道数
            self.block_num = args.block_num  # 残差块数量
            # 展开空洞系数列表，例如 [1,4] * 4 -> [1,4,1,4,1,4,1,4]
            self.dilations = args.dilations * self.block_num
            self.kernel_size = args.kernel_size  # 卷积核大小
            self.enabel_res_parameter = args.enable_res_parameter  # 是否启用残差缩放参数
            self.dropout = args.dropout  # dropout 概率
            self.device = args.device  # 设备信息
            self.data_shape = args.data_shape  # 输入数据形状 (L, C)
        else:
            d_model = kwargs['d_model']  # 隐藏维度
            self.embedding_size = kwargs['d_model']  # 嵌入维度
            self.residual_channels = kwargs['d_model']  # 残差通道数
            self.block_num = kwargs['block_num']  # 残差块数量
            self.dilations = kwargs['dilations'] * self.block_num  # 空洞系数
            self.data_shape = kwargs['data_shape']  # 输入数据形状
            self.kernel_size = 3  # 默认卷积核大小
            self.enabel_res_parameter = 1  # 默认启用残差缩放
            self.dropout = 0.1  # 默认 dropout 概率

        self.max_len = self.data_shape[0]  # 序列长度 L
        print(self.max_len)  # 打印长度，方便调试

        # 构建残差块列表，空洞因子控制感受野大小
        rb = [
            ResidualBlock_b(
                self.residual_channels, self.residual_channels, kernel_size=self.kernel_size, dilation=dilation,
                enable_res_parameter=self.enabel_res_parameter, dropout=self.dropout
            ) for dilation in self.dilations
        ]
        self.residual_blocks = nn.Sequential(*rb)  # 顺序堆叠所有残差块

        # 每个时间步的线性投影层（保持维度一致）
        # self.output = nn.Linear(self.residual_channels, self.num_class)
        self.output = nn.Linear(d_model, d_model)  # 输出线性层
        # 将隐藏特征映射回原始通道的线性层
        self.broadcast_head = nn.Linear(d_model, self.data_shape[1])

        self.apply(self._init_weights)  # 初始化线性层参数

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):  # 仅对线性层初始化
            xavier_normal_(module.weight.data)  # Xavier 初始化权重
            if module.bias is not None:  # 若存在偏置
                constant_(module.bias.data, 0.1)  # 偏置初始化为常数 0.1

    def forward(self, x):
        # 期待输入形状 [B, L, D]
        dilate_outputs = self.residual_blocks(x)  # 经过所有残差块
        x = dilate_outputs  # 更新 x
        return self.output(x)  # 返回线性投影结果 [B, L, D]


class ResidualBlock_b(nn.Module):
    r"""
    含两个空洞卷积的残差模块，具有如下特点：
      - 使用 Conv2d 的 (1, K) 卷积核模拟沿时间轴的一维卷积；
      - 通过 conv_pad 做左侧填充，保证因果性不泄漏未来信息；
      - 每层卷积后接 LayerNorm + ReLU + Dropout 提升稳定性；
      - 可选的残差缩放参数 self.a，用于学习残差权重。
    """

    def __init__(self, in_channel, out_channel, kernel_size=10, dilation=None, enable_res_parameter=False, dropout=0):
        super(ResidualBlock_b, self).__init__()  # 初始化父类

        # 第一层空洞卷积：通过 conv_pad 调整形状后进行
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(1, kernel_size), padding=0, dilation=dilation)
        self.dropout1 = nn.Dropout(dropout)  # 第一层 dropout
        self.ln1 = nn.LayerNorm(out_channel, eps=1e-8)  # 第一层 LayerNorm
        # 第二层卷积，空洞率翻倍以扩大感受野
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(1, kernel_size), padding=0, dilation=dilation * 2)
        self.dropout2 = nn.Dropout(dropout)  # 第二层 dropout
        self.ln2 = nn.LayerNorm(out_channel, eps=1e-8)  # 第二层 LayerNorm

        self.dilation = dilation  # 当前空洞率
        self.kernel_size = kernel_size  # 卷积核大小

        self.enable = enable_res_parameter  # 是否启用可学习残差缩放
        # 可学习残差系数，当 enable 为 True 时生效
        self.a = nn.Parameter(torch.tensor(1e-8))

    def forward(self, x):  # x: [batch_size, seq_len, embed_size]
        x_pad = self.conv_pad(x, self.dilation)  # 填充后形状 [B, embed, 1, L_pad]
        out = self.dropout1(self.conv1(x_pad).squeeze(2).permute(0, 2, 1))  # 恢复形状 [B, L, D]
        out = F.relu(self.ln1(out))  # LayerNorm + ReLU
        out_pad = self.conv_pad(out, self.dilation * 2)  # 第二次填充（空洞率翻倍）
        out2 = self.dropout2(self.conv2(out_pad).squeeze(2).permute(0, 2, 1))  # 第二层卷积
        out2 = F.relu(self.ln2(out2))  # LayerNorm + ReLU

        if self.enable:  # 可学习残差分支
            x = self.a * out2 + x
        else:  # 默认直接残差相加
            x = out2 + x

        return x  # 返回残差输出
        # return self.skipconnect(x, self.ffn)  # 可选的替代实现（未使用）

    def conv_pad(self, x, dilation):
        """
        对输入进行左侧零填充并调整维度：
        - 输入 [B, L, D] -> [B, D, 1, L]，以便使用 Conv2d 模拟一维卷积；
        - 左侧填充长度为 (kernel_size - 1) * dilation，保证因果性。
        """
        inputs_pad = x.permute(0, 2, 1)  # 交换维度 -> [B, D, L]
        inputs_pad = inputs_pad.unsqueeze(2)  # 插入维度 -> [B, D, 1, L]
        pad = nn.ZeroPad2d(((self.kernel_size - 1) * dilation, 0, 0, 0))  # 仅左侧填充
        inputs_pad = pad(inputs_pad)  # 应用填充
        return inputs_pad  # 返回填充后的张量
    
# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


# Borrowed from https://github.com/deepmind/sonnet and ported it to PyTorch

class Quantize(nn.Module):
    """
    向量量化模块（VQ-VAE），支持 EMA 码本更新与直通估计：
      - 将连续隐向量映射到最近的码字上，输出离散索引；
      - 通过直通梯度技巧，反向传播时保留梯度；
      - 使用指数滑动平均，稳定更新码字向量。
    参数说明：
        dim: 每个时间步向量的维度；
        n_embed: 码本大小（离散 token 数量）；
        decay: EMA 衰减系数；
        eps: 数值稳定用的极小值；
        beta: 承诺损失权重。
    """
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5, beta=0.25):
        super().__init__()  # 初始化父类

        self.dim = dim  # 向量维度
        self.n_embed = n_embed  # 码本条目数
        self.decay = decay  # EMA 衰减因子
        self.eps = eps  # 稳定项
        self.beta = beta  # 承诺损失系数

        embed = torch.randn(dim, n_embed)  # 随机初始化码本
        torch.nn.init.kaiming_uniform_(embed)  # 使用 Kaiming 均匀初始化
        self.register_buffer("embed", embed)  # 注册码本为缓冲区（不参与梯度）
        self.register_buffer("cluster_size", torch.zeros(n_embed))  # 聚类计数
        self.register_buffer("embed_avg", embed.clone())  # 聚类均值

    def forward(self, input):
        # input: [..., dim]，先展平最后一维参与距离计算
        flatten = input.reshape(-1, self.dim)
        # 计算到每个码字的平方欧氏距离
        dist = (
                flatten.pow(2).sum(1, keepdim=True)
                - 2 * flatten @ self.embed
                + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)  # 取距离最近的码字索引
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)  # 独热编码
        embed_ind = embed_ind.view(*input.shape[:-1])  # 恢复索引张量形状
        quantize = self.embed_code(embed_ind)  # 查询码本得到量化结果

        if self.training:
            # EMA 更新码本统计量
            embed_onehot_sum = embed_onehot.sum(0)  # 累积计数
            embed_sum = flatten.transpose(0, 1) @ embed_onehot  # 累积向量和

            self.cluster_size.data.mul_(self.decay).add_(  # 更新计数均值
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)  # 更新向量和
            n = self.cluster_size.sum()  # 总计数
            cluster_size = (  # 归一化计数，防止除零
                    (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)  # 求均值更新码本
            self.embed.data.copy_(embed_normalized)  # 写回码本

        diff = (quantize.detach() - input).pow(2).mean()  # 码本重构损失
        commit_loss = (quantize - input.detach()).pow(2).mean()  # 承诺损失
        diff += commit_loss * self.beta  # 合并损失
        # 直通估计器：前向用量化结果，反向梯度传给编码器输出
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind  # 返回量化向量、损失、离散索引

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))  # 查询码本向量

class Encoder(nn.Module):
    def __init__(self, feat_num, hidden_dim, block_num, data_shape, dilations=[1, 4]):
        super().__init__()  # 初始化父类
        self.input_projection = nn.Linear(feat_num, hidden_dim)  # 输入线性映射
        self.blocks = TCN(args=None, d_model=hidden_dim, block_num=block_num, data_shape=data_shape,  # 堆叠 TCN
                          dilations=dilations)

    def forward(self, input):
        return self.blocks(self.input_projection(input))  # [B, L, C] -> [B, L, D]


class Decoder(nn.Module):
    def __init__(self, feat_num, hidden_dim, block_num, data_shape, dilations=[1, 4]):
        super().__init__()  # 初始化父类
        self.output_projection = nn.Linear(hidden_dim, feat_num)  # 输出线性映射
        self.blocks = TCN(args=None, d_model=hidden_dim, block_num=block_num, data_shape=data_shape,  # 堆叠 TCN
                          dilations=dilations)

    def forward(self, input):
        return self.output_projection(self.blocks(input))  # [B, L, D] -> [B, L, C]


class TStokenizer(nn.Module):
    def __init__(
            self,
            data_shape=(5000, 12),
            hidden_dim=64,
            n_embed=1024,
            block_num=4,
            wave_length=32,
    ):
        super().__init__()  # 初始化父类
        self.enc = Encoder(data_shape[1], hidden_dim, block_num, data_shape)  # 编码器
        self.wave_patch = (wave_length, hidden_dim)  # 时间块大小 (时间长, 隐藏维)
        self.quantize_input = nn.Conv2d(1, hidden_dim, kernel_size=self.wave_patch, stride=self.wave_patch)  # 分块
        self.quantize = Quantize(hidden_dim, n_embed)  # 向量量化器
        self.quantize_output = nn.Conv1d(int(data_shape[0] / wave_length), data_shape[0], kernel_size=1)  # 上采样
        self.dec = Decoder(data_shape[1], hidden_dim, block_num, data_shape)  # 解码器
        self.n_embed = n_embed  # 码本大小
        self.hidden_dim = hidden_dim  # 隐藏维度

    def get_name(self):
        return 'vqvae'  # 返回模型名称

    def forward(self, input):
        enc = self.enc(input)  # [B, L, C] -> [B, L, D]
        enc = enc.unsqueeze(1)  # 调整形状以适配 Conv2d -> [B, 1, L, D]
        quant = self.quantize_input(enc).squeeze(-1).transpose(1, 2)  # 非重叠分块 -> [B, L//wl, D]
        quant, diff, id = self.quantize(quant)  # 量化 -> (量化结果, 损失, token id)
        quant = self.quantize_output(quant)  # 上采样回原始长度 -> [B, L, D]
        dec = self.dec(quant)  # 解码重建 -> [B, L, C]
        return dec, diff, id  # 返回重建信号、VQ 损失、离散索引

    def get_embedding(self, id):
        return self.quantize.embed_code(id)  # 使用码本将离散 id 转回嵌入

    def decode_ids(self, id):
        quant = self.get_embedding(id)  # 码本查询 -> 嵌入序列
        quant = self.quantize_output(quant)  # 上采样恢复时序长度
        dec = self.dec(quant)  # 解码回原始信号

        return dec

if __name__ == '__main__':
    model = TStokenizer()
    a = torch.randn(2, 5000, 8)
    tmp = model(a)
    print(1)
