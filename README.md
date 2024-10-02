# 从零实现 Transfromer | Transformer from scratch

这是我用 PyTorch 实现 NLP 领域经典模型 **Transformer** 的项目代码。

为了更好地理解 **Transformer** 的网络架构，欢迎访问我的博客[深入浅出Transformer](https://nijikadesu.github.io/2024/09/27/dive-into-transformer/)，希望这能给你一些帮助。

## 项目背景

这个项目的目的是帮助我更好的了解 Transformer 的网络架构和诸如多头自注意力、位置嵌入等机制，也希望通过动手编程提高 PyTorch 的熟练度。

我在学习的过程中对代码和知识进行了一些整理和重构，使其更加系统化且容易理解，本着帮助更多像我一样希望通过 Transformer 入门 NLP 的萌新们的目的，我将项目代码作了比较详细的注释并托管在这个仓库，
如果你发现这个仓库对你相关知识的学习贡献了一些帮助，请为我点一个 Star ，非常感谢！

## 项目结构
```text
└── data                                # 训练用数据集
│   ├──english.txt                      # 英文数据集
│   └──kannada.txt                      # 卡纳达语数据集
├── data_utils                          # 数据处理工具
│   ├──read_file.py                     # 读取文本数据，并切片欲处理
│   └──tokenizer.py                     # 文本转数字 token ，并加入位置嵌入
├── mechanism                           # Transformer 模型机制
│   ├──attention_mask.py                # 注意力掩码
│   ├──feed_forward.py                  # 前向传播神经网络
│   ├──layer_normalization.py           # 层归一化
│   ├──multihead-attention.py           # 多头自注意力
│   ├──multihead-cross-attention.py     # 多头交叉注意力
│   └──positional_encoding.py           # 位置嵌入
├── structure                           # 模型架构
│   ├──encoder.py                       # 编码器
│   └──decoder.py                       # 解码器
├── config.py                           # 训练配置
├── dataset.py                          # 数据集，继承 nn.Dataset
├── train.py                            # 训练文件
└── transformer.py                      # 完整 Transformer 模型
```

## 环境配置
```text
python==3.10
torch==2.2.2
torchvision==0.17.2
numpy==1.23.5
```

## 使用方法
- 在 `./mechanism` 文件夹中，我们将 Transformer 进行分解，学习其模型机制。

- 在 `./structrue` 文件夹中，我们将在上述机制结合起来，并组合成编码器，解码器。

- 在 `./transformer` 文件夹中，我们将编码器解码器进行堆叠组合，便构成了 Transformer 的整体架构。

在我们成功构建 Transformer 后，我们还提供了一个实战项目，用我们从零构建的 Transformer 训练一个翻译器，训练代码在 `./data` 中给出，训练文件在 `./train.py` 中给出，
训练所需要的模型参数在 `./config.py` 中给出。