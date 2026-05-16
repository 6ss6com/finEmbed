# FinEmbed - 金融Embedding微调训练

> 训练一个专门用于股票/财经领域的向量化模型，让RAG知识检索更准确

## 项目目标

基于开源中文Embedding模型 `bge-large-zh-v1.5`，使用金融领域数据微调，训练一个专门理解股票/财经知识的向量化模型。

## 为什么做这个

1. **提升RAG准确率** - 通用Embedding对金融术语理解差，专业模型检索更准
2. **覆盖所有知识工作** - 股票监控/记忆系统/内容生成都能受益
3. **商业价值** - 可出租/售卖训练好的模型

## 技术路线

```
基础模型: bge-large-zh-v1.5 (开源中文Embedding)
训练数据: 
  - 股票公告（东方财富/巨潮）
  - 财经新闻（新浪/腾讯财经）
  - 微信聊天记录（F:\微信聊天记录）
  - OpenClaw记忆数据
训练框架: SentenceTransformers
硬件需求: V100 16GB (FP16训练约14GB)
训练周期: 2-4小时
```

## 训练数据来源

| 数据源 | 说明 | 路径 |
|--------|------|------|
| 股票公告 | 上市公司公告PDF/TXT | cangku/ |
| 财经新闻 | 爬取或下载 | 待定 |
| 记忆数据 | OpenClaw向量记忆 | C:\Users\Administrator\.openclaw\workspace\memory |
| 微信记录 | 对话语料 | F:\微信聊天记录 |
| distill-ai | 人格知识库 | C:\Users\Administrator\.openclaw\workspace\distill-ai |

## 目录结构

```
finEmbed/
├── README.md              # 本文件
├── requirements.txt       # 依赖
├── train.py               # 训练脚本
├── data/                  # 训练数据
├── models/                # 输出模型
├── configs/               # 配置文件
├── scripts/               # 辅助脚本
└── eval/                  # 评估脚本
```

## 快速开始

```bash
cd F:\training\finEmbed
pip install sentence-transformers torch datasets
python train.py --epochs 3 --batch_size 16
```

## 预期收益

- RAG检索准确率 +30%
- 覆盖股票/记忆/所有知识场景
- 训练好的模型可发布到HuggingFace

## 团队分工

| 角色 | 负责 |
|------|------|
| 金（主人） | 提供训练数据/硬件资源 |
| 本（AI） | 训练脚本/模型优化/评估 |

## License

Apache 2.0 - 开源合作，开放贡献