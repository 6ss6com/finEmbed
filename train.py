"""
FinEmbed - 金融Embedding微调训练
训练专业金融向量化模型

Usage:
    python train.py --epochs 3 --batch_size 16
"""

import os
import sys
import argparse
import logging
from datetime import datetime

import torch
from sentence_transformers import SentenceTransformer, InputExample, LoggingHandler
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
from datasets import load_dataset
import numpy as np

# 配置
MODEL_NAME = 'BAAI/bge-large-zh-v1.5'  # 基础Embedding模型
OUTPUT_DIR = r'F:\training\finEmbed\models'

logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    handlers=[
        LoggingHandler(),
        logging.FileHandler(f'F:\\training\\finEmbed\\logs\\train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class FinancialEmbeddingTrainer:
    """金融Embedding训练器"""
    
    def __init__(self, model_name=MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"设备: {self.device}")
    
    def load_model(self):
        """加载基础模型"""
        logger.info(f"加载基础模型: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        logger.info("模型加载完成")
    
    def prepare_data(self):
        """准备训练数据"""
        logger.info("准备训练数据...")
        
        # 从多个来源加载数据
        train_examples = []
        
        # 1. 从memory文件加载对话
        memory_dir = r'C:\Users\Administrator\.openclaw\workspace\memory'
        if os.path.exists(memory_dir):
            for f in os.listdir(memory_dir):
                if f.endswith('.md'):
                    path = os.path.join(memory_dir, f)
                    with open(path, 'r', encoding='utf-8') as file:
                        content = file.read()
                        # 简单分句
                        sentences = [s.strip() for s in content.split('。') if len(s.strip()) > 10]
                        for i in range(len(sentences)-1):
                            train_examples.append(InputExample(
                                texts=[sentences[i], sentences[i+1]],
                                label=float(min(1.0, (i+1)/10))  # 模拟相似度
                            ))
        
        # 2. 从distill-ai加载人格对话
        distill_dir = r'C:\Users\Administrator\.openclaw\workspace\distill-ai\distill\personas'
        if os.path.exists(distill_dir):
            for f in os.listdir(distill_dir):
                if f.endswith('.json'):
                    # 加载人格描述作为训练数据
                    import json
                    path = os.path.join(distill_dir, f)
                    with open(path, 'r', encoding='utf-8') as file:
                        persona = json.load(file)
                        if 'description' in persona:
                            train_examples.append(InputExample(
                                texts=[persona['name'], persona['description']],
                                label=0.9
                            ))
        
        logger.info(f"训练数据: {len(train_examples)} 条")
        return train_examples
    
    def train(self, train_examples, epochs=3, batch_size=16, warmup_steps=100):
        """训练模型"""
        if not self.model:
            self.load_model()
        
        logger.info("开始训练...")
        
        # 创建数据加载器
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        
        # 评估器（用训练数据的一小部分）
        evaluator = EmbeddingSimilarityEvaluator.from_examples(
            train_examples[:min(100, len(train_examples))],
            batch_size=batch_size
        )
        
        # 训练
        self.model.fit(
            train_objectives=[(train_dataloader, None)],
            evaluator=evaluator,
            epochs=epochs,
            warmup_steps=warmup_steps,
            output_path=OUTPUT_DIR,
            show_progress_bar=True
        )
        
        logger.info(f"训练完成！模型保存到: {OUTPUT_DIR}")
    
    def evaluate(self):
        """评估模型"""
        if not self.model:
            logger.error("模型未加载")
            return
        
        logger.info("评估模型...")
        
        # 简单的金融术语测试
        test_pairs = [
            ('茅台', '贵州茅台'),
            ('股票', '证券'),
            ('股息', '分红'),
            ('市盈率', 'PE'),
            ('北向资金', '外资'),
        ]
        
        embeddings = self.model.encode([p[0] for p in test_pairs] + [p[1] for p in test_pairs])
        
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity(embeddings[:len(test_pairs)], embeddings[len(test_pairs):])
        
        for i, (a, b) in enumerate(test_pairs):
            logger.info(f"  {a} <-> {b}: {sims[i][i]:.3f}")
        
        logger.info("评估完成")


def main():
    parser = argparse.ArgumentParser(description='FinEmbed 训练')
    parser.add_argument('--epochs', type=int, default=3, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--warmup', type=int, default=100, help='预热步数')
    args = parser.parse_args()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(r'F:\training\finEmbed\logs', exist_ok=True)
    
    trainer = FinancialEmbeddingTrainer()
    examples = trainer.prepare_data()
    
    if len(examples) == 0:
        logger.error("没有训练数据！")
        return
    
    trainer.train(examples, epochs=args.epochs, batch_size=args.batch_size, warmup_steps=args.warmup)
    trainer.evaluate()


if __name__ == '__main__':
    main()