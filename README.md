# Lifelong Learning Lab | 终身学习实验室

> 在有限算力下，研究人工神经网络如何持续学习、适应变化，并尽可能保持旧能力与学习新任务的可塑性。

**研究原则：先观察，再提出机制假设；用可复现实验支持或反驳假设。** 不将 Agent 的解释当作实验结论，也不将 MNIST 结果直接推广到 LLM、人脑或机器人。

## 研究状态（2026-11-16）

| 实验 | 核心问题 | 状态 |
|---|---|---|
| [Lab 01](labs/lab01/README.md) | A→B 后旧任务准确率如何变化？Replay 有用吗？ | 已运行，单种子已记录 |
| [Lab 02](labs/lab02/README.md) | 训练时长与随机种子怎样影响遗忘估计？ | 已运行；有报告，原始 exp02 脚本与完整 CSV 待同步 |
| [Lab 03](labs/lab03/PROTOCOL.md) | 连续学习许多任务后，学习新任务的能力是否退化？ | 进行中，结果尚未同步 |
| Lab 04（计划） | Continual Backprop 能否维护可塑性？ | 未开始 |

## 快速开始

```powershell
git clone https://github.com/yixuanqi/lifelong-learning-lab.git
cd lifelong-learning-lab/labs/lab01
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
python -m pip install matplotlib
python smoke_test.py
python train.py --device auto
```

首次运行将下载 MNIST。优先复用已安装的 CUDA PyTorch，**不要为安装 torchvision 意外替换 GPU 环境**。

## 目录

- `labs/lab01/`：灾难性遗忘和 Replay 的训练代码、运行说明
- `labs/lab02/`：五随机种子实验报告及待补齐的数据
- `labs/lab03/`：可塑性丧失实验协议
- `research/ROADMAP.md`：半年研究路线
- `research/EXPERIMENT_LOG.md`：实验进展日志
- `research/papers/`：论文笔记

## 实验质量规则

每次实验记录：假设、模型/数据版本、种子、配置、训练预算、评估方法、原始数据、图表和反例。
结果分别标为 **verified**（有原始数据可检查）、**reported**（仅报告支持）与 **planned**（尚未运行）。
不上传公司测量数据、商业机密、个人隐私、未授权的数据集或重量级模型文件。
PR 应带对照实验和复现说明；失败结果也要保留。

详见 [Roadmap](research/ROADMAP.md) 和 [Contributing](CONTRIBUTING.md)。
