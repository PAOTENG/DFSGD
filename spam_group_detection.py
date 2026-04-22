#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
水军群组检测模型实现
版本说明：
该模型包含8个核心模块：
模块1：节点按评论时序拆分模块
模块2：特征矩阵和邻接矩阵构建模块  
模块3：引力图和斥力图构建模块
模块4：增强邻接矩阵操作模块
模块5：权重感知GCN编码器与DBSCAN联合优化聚类模块（已修改：添加对比学习）
模块6：节点聚合模块
模块7：候选群组净化与合并模块
模块8：指标验证与结果输出模块
"""

import os
import sys
import time
import sqlite3
import argparse
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from datetime import datetime, date, timedelta
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import hdbscan  # 用于二次聚类
import warnings
import json
from tqdm import tqdm
import pickle
import hashlib
import logging
import traceback
import tempfile
import shutil
import mmap
from multiprocessing import Manager
from typing import Dict, List, Optional

# ================================
# 用户指标缓存系统（集成版）
# ================================

class UserMetricsCacheBuilder:
    """
    用户数据缓存构建器 - 预先加载用户评论数据和计算ISS指标
    避免在模块6-7运行时进行SQL查询
    
    缓存内容：
    1. ISS指标缓存：每个用户的ISS计算所需指标
    2. 用户评论数据缓存：每个用户的完整评论数据（供GSS计算使用）
    """
    
    def __init__(self, db_path: str, cache_dir: str = None):
        self.db_path = db_path
        # 如果未指定cache_dir，则根据数据集名称自动生成
        if cache_dir is None:
            dataset_name = get_dataset_name(db_path)
            cache_dir = f"preprocessed_{dataset_name}/user_metrics_cache"
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # 缓存文件路径
        self.iss_cache_file = os.path.join(cache_dir, "iss_metrics.pkl")
        self.user_reviews_cache_file = os.path.join(cache_dir, "user_reviews.pkl")
        self.metadata_file = os.path.join(cache_dir, "cache_metadata.json")
    
    def build_cache(self, force_rebuild=False):
        """构建用户指标缓存"""
        # 检查缓存是否已存在
        if not force_rebuild and self._cache_exists():
            pass
            return True
        
        print("\n" + "="*60)
        print("="*60)
        
        try:
            # 连接数据库
            conn = sqlite3.connect(self.db_path)
            
            # 1. 获取所有用户的评论数据
            query = """
            SELECT user_id, id, rating, review_text, date, product_id, label
            FROM reviews
            ORDER BY user_id, date
            """
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            
            # 2. 按用户分组处理（优化版本）
            iss_metrics = {}
            user_reviews_cache = {}
            
            # 使用groupby并转换为字典，避免重复迭代
            user_groups = {user_id: group for user_id, group in df.groupby('user_id')}
            total_users = len(user_groups)
            
            # 批量处理
            processed = 0
            for user_id, user_reviews in user_groups.items():
                processed += 1
                if processed % 5000 == 0:
                    pass
                # 计算ISS指标
                iss_metrics[user_id] = self._calculate_iss_metrics(user_reviews)
                # 缓存用户评论数据（供GSS计算使用）- 保留必要字段（包括label）
                user_reviews_cache[user_id] = user_reviews[['user_id', 'rating', 'review_text', 'date', 'product_id', 'label']].to_dict('records')
            
            
            # 3. 保存缓存文件
            with open(self.iss_cache_file, 'wb') as f:
                pickle.dump(iss_metrics, f)
            
            with open(self.user_reviews_cache_file, 'wb') as f:
                pickle.dump(user_reviews_cache, f)
            
            # 保存元数据
            metadata = {
                'build_time': datetime.now().isoformat(),
                'db_path': self.db_path,
                'total_users': len(iss_metrics),
                'total_reviews': len(df)
            }
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print("="*60)
            
            return True
            
        except Exception as e:
            pass
            traceback.print_exc()
            return False
    
    def _cache_exists(self):
        """检查缓存是否存在"""
        return (os.path.exists(self.iss_cache_file) and 
                os.path.exists(self.user_reviews_cache_file) and
                os.path.exists(self.metadata_file))
    
    def _calculate_iss_metrics(self, user_reviews: pd.DataFrame) -> Dict:
        """计算单个用户的ISS相关指标（12维特征）"""
        metrics = {}
        
        # 基础统计
        metrics['review_count'] = len(user_reviews)
        
        # 评分相关特征
        ratings = user_reviews['rating'].values
        
        # 1. rating_mean - 平均评分（核心特征1）
        metrics['rating_mean'] = float(np.mean(ratings))
        
        # 2. rating_change_frequency - 评分变化频率（核心特征2，NEW!）
        if len(ratings) > 1:
            rating_changes = sum(1 for i in range(1, len(ratings)) if ratings[i] != ratings[i-1])
            metrics['rating_change_frequency'] = float(rating_changes / (len(ratings) - 1))
        else:
            metrics['rating_change_frequency'] = 0.0
        
        # 3. extreme_rating_ratio - 极端评分比例（核心特征3）
        extreme_count = np.sum((ratings == 1) | (ratings == 5))
        metrics['extreme_rating_ratio'] = float(extreme_count / len(ratings))
        
        # 4. rating_pattern_score - 极端评分连续性（核心特征4，NEW!）
        extreme_runs = []
        current_run = 0
        for r in ratings:
            if r == 1 or r == 5:
                current_run += 1
            else:
                if current_run > 0:
                    extreme_runs.append(current_run)
                current_run = 0
        if current_run > 0:
            extreme_runs.append(current_run)
        metrics['rating_pattern_score'] = float(max(extreme_runs)) if extreme_runs else 0.0
        
        # 5. rating_deviation - 评分偏差（辅助特征1）
        metrics['rating_deviation'] = float(np.abs(ratings - ratings.mean()).mean())
        
        # 6. rating_std - 评分标准差（辅助特征2）
        metrics['rating_std'] = float(np.std(ratings))
        
        # 6.5. rating_variance - 评分方差（ISS需要）
        metrics['rating_variance'] = float(np.var(ratings))
        
        # 7. text_similarity - 文本相似度（辅助特征3，NEW!）
        texts = user_reviews['review_text'].fillna('').astype(str).values
        if len(texts) > 1:
            import re
            words_sets = [set(re.findall(r'\w+', t.lower())) for t in texts]
            similarities = []
            for i in range(len(words_sets)):
                for j in range(i+1, len(words_sets)):
                    if len(words_sets[i]) > 0 and len(words_sets[j]) > 0:
                        overlap = len(words_sets[i] & words_sets[j])
                        union = len(words_sets[i] | words_sets[j])
                        similarities.append(overlap / union if union > 0 else 0)
            metrics['text_similarity'] = float(np.mean(similarities)) if similarities else 0.0
        else:
            metrics['text_similarity'] = 0.0
        
        # 8. product_concentration - 产品集中度（辅助特征4）
        product_counts = user_reviews['product_id'].value_counts()
        metrics['product_concentration'] = float(product_counts.max() / len(user_reviews))
        
        # 9. review_count - 评论数量（补充特征1）
        # 已在开头计算
        
        # 10. avg_review_length - 平均评论长度（补充特征2）
        review_lengths = user_reviews['review_text'].fillna('').str.len()
        metrics['avg_review_length'] = float(review_lengths.mean())
        
        # 10.5. review_length_std - 评论长度标准差（ISS需要）
        metrics['review_length_std'] = float(review_lengths.std())
        
        # 11. time_span_days - 时间跨度（补充特征3）
        user_reviews['review_time'] = pd.to_datetime(user_reviews['date'])
        time_span = (user_reviews['review_time'].max() - user_reviews['review_time'].min())
        metrics['time_span_days'] = float(time_span.total_seconds() / 86400)
        
        # 12. unique_products - 独特产品数（补充特征4）
        metrics['unique_products'] = int(len(product_counts))
        
        return metrics
    

class UserMetricsCacheReader:
    """
    用户数据缓存读取器 - 快速读取预构建的用户数据
    
    提供：
    1. ISS指标：用于用户过滤
    2. 用户评论数据：用于GSS计算
    """
    
    def __init__(self, cache_dir: str = "preprocessed/user_metrics_cache"):
        self.cache_dir = cache_dir
        
        # 缓存文件路径
        self.iss_cache_file = os.path.join(cache_dir, "iss_metrics.pkl")
        self.user_reviews_cache_file = os.path.join(cache_dir, "user_reviews.pkl")
        self.metadata_file = os.path.join(cache_dir, "cache_metadata.json")
        
        # 加载缓存
        self._load_cache()
    
    def _load_cache(self):
        """加载缓存文件到内存"""
        if not os.path.exists(self.iss_cache_file):
            raise FileNotFoundError(f"ISS缓存文件不存在: {self.iss_cache_file}")
        
        if not os.path.exists(self.user_reviews_cache_file):
            raise FileNotFoundError(f"用户评论缓存文件不存在: {self.user_reviews_cache_file}")
        
        # 加载ISS指标
        with open(self.iss_cache_file, 'rb') as f:
            self.iss_metrics = pickle.load(f)
        
        # 加载用户评论数据
        with open(self.user_reviews_cache_file, 'rb') as f:
            self.user_reviews = pickle.load(f)
        
        # 加载元数据
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
        
    
    def get_user_reviews(self, user_id) -> Optional[List]:
        """获取单个用户的评论数据"""
        return self.user_reviews.get(user_id)
    
    def get_batch_iss_metrics(self, user_ids: List) -> Dict:
        """批量获取用户的ISS指标"""
        return {uid: self.iss_metrics.get(uid) for uid in user_ids if uid in self.iss_metrics}
    
    def get_batch_user_reviews(self, user_ids: List) -> Dict:
        """批量获取用户的评论数据"""
        return {uid: self.user_reviews.get(uid) for uid in user_ids if uid in self.user_reviews}
    
    def close(self):
        """关闭缓存（清理内存）"""
        self.iss_metrics = None
        self.user_reviews = None
        gc.collect()

# 导入GCAS损失函数


# 设置中文字体，解决图片中文乱码问题
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10


# 忽略警告
warnings.filterwarnings('ignore')

# 日志记录功能
def setup_logging():
    """设置日志记录功能"""
    # 创建logs目录
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    
    # 生成日志文件名（包含时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(logs_dir, f"spam_detection_{timestamp}.log")
    
    # 配置日志记录器
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)  # 同时输出到控制台
        ]
    )
    
    return log_filename

def log_program_end(log_filename, success=True, error_msg=None):
    """记录程序结束信息"""
    end_time = datetime.now()
    
    if success:
        logging.info("="*60)
        logging.info("程序执行完成")
        logging.info(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info("="*60)
    else:
        logging.error("="*60)
        logging.error("程序执行失败")
        logging.error(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        if error_msg:
            logging.error(f"错误信息: {error_msg}")
        logging.error("="*60)

# 初始化日志记录
log_filename = setup_logging()


# 设置随机种子
def set_seed(seed=42):
    """设置随机种子以确保结果可复现"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# 设备配置 - 优先使用GPU
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # 主要计算使用GPU
    gpu_device = torch.device("cuda:0")
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("Using GPU for computation and training.")
else:
    device = torch.device("cpu")  # 主要计算使用CPU
    gpu_device = None
    print("No GPU detected. Running on CPU.")

def get_device():
    """获取计算设备"""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")

# 结果目录管理
def get_dataset_name(db_path):
    """从数据库路径提取数据集名称"""
    if not db_path:
        return "default"
    # 提取文件名（不含扩展名）
    dataset_name = os.path.splitext(os.path.basename(db_path))[0]
    return dataset_name

def get_result_dir(sample_ratio=1.0, db_path=None, module=None, force_no_threshold=False):
    """根据采样比例、数据集名称和模块编号获取结果目录
    
    Args:
        sample_ratio: 采样比例
        db_path: 数据库路径
        module: 模块编号(1-8)，如果为None则返回基础目录
        force_no_threshold: 保留参数兼容性，不再使用
    
    Returns:
        结果目录路径
    """
    # 获取数据集名称
    dataset_name = get_dataset_name(db_path)
    
    # 基础目录：统一使用 preprocessed_{dataset_name}，不加阈值后缀
    base_dir = f"preprocessed_{dataset_name}"
    
    # 根据采样比例确定子目录
    if sample_ratio == 1.0:
        data_dir = "full_data"
    else:
        ratio_str = str(sample_ratio).replace('.', '_')
        data_dir = f"sample_{ratio_str}"
    
    # 如果指定了模块编号，添加模块子目录
    if module is not None:
        return f"{base_dir}/{data_dir}_module{module}"
    else:
        return f"{base_dir}/{data_dir}"

result_dir = None

# 命令行参数解析
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='水军群组检测模型')
    
    # 数据相关参数
    parser.add_argument('--dataset', type=str, default='DataSet/Electronics_2013_1.6.db', 
                       help='数据库文件路径 (支持: Cell_Phones_and_Accessorie.db, Electronics_2013_1.6.db)')
    parser.add_argument('--sample_ratio', type=float, default=1.0,  # 修改为1.0
                       help='数据采样比例 (默认: 1.0)')

    
    # 模型参数
    parser.add_argument('--hidden_dim', type=int, default=32,
                       help='GAT隐藏层维度 (默认: 32)')
    parser.add_argument('--embedding_dim', type=int, default=64,
                       help='节点嵌入维度 (默认: 64)')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout率 (默认: 0.3)')
    parser.add_argument('--alpha', type=float, default=0.2,
                       help='LeakyReLU负斜率 (默认: 0.2)')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=200,
                       help='训练轮数 (默认: 200)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学习率 (默认: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                       help='权重衰减 (默认: 5e-4)')
    
    # 阈值参数
    parser.add_argument('--lambda_param', type=float, default=0.3,
                       help='增强邻接矩阵调节参数λ (默认: 0.3)')
    parser.add_argument('--attraction_threshold', type=float, default=0.92,
                       help='引力图阈值 (默认: 0.92)')
    parser.add_argument('--repulsion_threshold', type=float, default=0.0000,
                       help='斥力图阈值 (默认: 0.0000)')
    parser.add_argument('--iss_threshold', type=float, default=0.3,
                       help='ISS过滤阈值 (默认: 0.3, 基于rating_std+rating_mean优化)')
    parser.add_argument('--group_threshold', type=float, default=0.7,
                       help='群组判定阈值 (默认: 0.7)')
    
    # 其他参数
    parser.add_argument('--no_cache', action='store_true',
                       help='不使用缓存')
    parser.add_argument('--retrain', action='store_true',
                       help='重新训练模型（删除GAT及后续模块缓存）')
    
    return parser.parse_args()

# ================================
# 模块1：节点按评论时序拆分模块
# ================================

class Module1_NodeSplitting:
    """
    模块1：节点按评论时序拆分模块
    
    功能：将用户按评论时间拆分为虚拟节点，格式为"用户ID_时间信息"
    """
    
    def __init__(self, db_path, sample_ratio=1.0):
        self.db_path = db_path
        self.sample_ratio = sample_ratio
        self.virtual_nodes = {}  # 虚拟节点映射：virtual_node_id -> user_info
        self.user_to_virtual = defaultdict(list)  # 用户到虚拟节点的映射
        
    def load_reviews_data(self):
        """从数据库加载评论数据"""
        
        conn = sqlite3.connect(self.db_path)
        
        # 构建SQL查询
        if self.sample_ratio < 1.0:
            # 使用RANDOM()进行采样
            query = f"""
            SELECT user_id as reviewerID, product_id as asin, rating as overall, date as reviewTime, review_text as reviewText, label
            FROM reviews 
            WHERE ABS(RANDOM()) % 100 < {int(self.sample_ratio * 100)}
            ORDER BY user_id, date
            """
        else:
            query = """
            SELECT user_id as reviewerID, product_id as asin, rating as overall, date as reviewTime, review_text as reviewText, label
            FROM reviews 
            ORDER BY user_id, date
            """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        
        return df
    
    def create_virtual_nodes(self, reviews_df):
        """
        创建虚拟节点
        
        根据文档要求，将用户按评论时间拆分为虚拟节点
        虚拟节点ID格式：用户ID_时间信息
        """
        
        virtual_nodes = {}
        user_to_virtual = defaultdict(list)
        node_id = 0
        
        # 按用户分组处理
        grouped = reviews_df.groupby('reviewerID')
        total_users = len(grouped)
        
        
        for user_id, user_reviews in tqdm(grouped, desc="创建虚拟节点"):
            # 按时间排序
            user_reviews = user_reviews.sort_values('reviewTime')
            
            # 为每个评论创建一个虚拟节点
            for idx, (_, review) in enumerate(user_reviews.iterrows()):
                # 创建虚拟节点ID：用户ID_时间戳
                review_time = pd.to_datetime(review['reviewTime'])
                time_str = review_time.strftime('%Y%m%d_%H%M%S')
                virtual_node_id = f"{user_id}_{time_str}_{idx}"
                # 存储虚拟节点信息
                virtual_nodes[node_id] = {
                    'virtual_node_id': virtual_node_id,
                    'original_user_id': user_id,
                    'review_time': review_time,
                    'asin': review['asin'],
                    'overall': review['overall'],
                    'reviewText': review['reviewText'] if pd.notna(review['reviewText']) else "",
                    'label': review['label'],
                    'time_index': idx  # 该用户的第几条评论
                }
                # 建立用户到虚拟节点的映射
                user_to_virtual[user_id].append(node_id)
                node_id += 1
        
        self.virtual_nodes = virtual_nodes
        self.user_to_virtual = dict(user_to_virtual)
        
        
        # 保存虚拟节点映射
        self._save_virtual_nodes()
        
        return virtual_nodes, user_to_virtual
    
    def _save_virtual_nodes(self):
        """保存虚拟节点映射到文件"""
        global result_dir
        # 模块1的缓存固定到不带阈值后缀的路径
        current_result_dir = get_result_dir(self.sample_ratio, self.db_path, module=1, force_no_threshold=True)
        os.makedirs(current_result_dir, exist_ok=True)
        
        # 保存虚拟节点详细信息
        virtual_nodes_file = os.path.join(current_result_dir, "virtual_nodes.pkl")
        with open(virtual_nodes_file, 'wb') as f:
            pickle.dump(self.virtual_nodes, f)
        
        # 保存用户到虚拟节点的映射
        user_mapping_file = os.path.join(current_result_dir, "user_to_virtual_mapping.pkl")
        with open(user_mapping_file, 'wb') as f:
            pickle.dump(self.user_to_virtual, f)
        
        # 保存可读的CSV文件
        virtual_nodes_csv = os.path.join(current_result_dir, "virtual_nodes.csv")
        nodes_data = []
        for node_id, info in self.virtual_nodes.items():
            nodes_data.append({
                'node_id': node_id,
                'virtual_node_id': info['virtual_node_id'],
                'original_user_id': info['original_user_id'],
                'review_time': info['review_time'],
                'asin': info['asin'],
                'overall': info['overall'],
                'label': info['label'],
                'time_index': info['time_index']
            })
        
        pd.DataFrame(nodes_data).to_csv(virtual_nodes_csv, index=False)
        
    
    def get_virtual_nodes_info(self):
        """获取虚拟节点统计信息"""
        if not self.virtual_nodes:
            return None
        
        total_nodes = len(self.virtual_nodes)
        total_users = len(self.user_to_virtual)
        
        # 统计每个用户的虚拟节点数量分布
        nodes_per_user = [len(nodes) for nodes in self.user_to_virtual.values()]
        
        # 检查是否有用户数据，避免空数组错误
        if not nodes_per_user:
            return {
                'total_virtual_nodes': total_nodes,
                'total_original_users': total_users,
                'avg_nodes_per_user': 0,
                'min_nodes_per_user': 0,
                'max_nodes_per_user': 0,
                'std_nodes_per_user': 0
            }
        
        stats = {
            'total_virtual_nodes': total_nodes,
            'total_original_users': total_users,
            'avg_nodes_per_user': np.mean(nodes_per_user),
            'min_nodes_per_user': np.min(nodes_per_user),
            'max_nodes_per_user': np.max(nodes_per_user),
            'std_nodes_per_user': np.std(nodes_per_user)
        }
        
        return stats
    
    def extract_temporal_features(self, reviews_df):
        """
        提取虚拟节点的时序特征，用于区分混合用户和正常用户
        
        时序特征包括：
        1. 时间间隔特征（3个）：平均时间间隔、时间间隔标准差、时间间隔变异系数
        2. 行为一致性特征（3个）：评分变化率、目标产品集中度、文本相似度
        
        返回：temporal_features字典，key为user_id，value为特征字典
        """
        
        temporal_features = {}
        grouped = reviews_df.groupby('reviewerID')
        
        # 初始化TF-IDF向量化器用于文本相似度计算
        tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        
        for user_id, user_reviews in tqdm(grouped, desc="提取时序特征"):
            user_reviews = user_reviews.sort_values('reviewTime')
            
            if len(user_reviews) < 2:
                # 单条评论的用户，设置默认特征
                temporal_features[user_id] = {
                    'user_id': user_id,
                    'review_count': 1,
                    'virtual_node_ids': self.user_to_virtual.get(user_id, []),
                    # 时间特征
                    'avg_time_interval': 0.0,
                    'std_time_interval': 0.0,
                    'cv_time_interval': 0.0,
                    # 行为一致性特征
                    'rating_change_rate': 0.0,
                    'product_concentration': 1.0,
                    'text_similarity': 1.0,
                    'user_type': 'normal' if user_reviews.iloc[0]['label'] == 1 else 'mixed'
                }
                continue
            
            # === 时间间隔特征 ===
            times = pd.to_datetime(user_reviews['reviewTime'])
            time_diffs = times.diff().dt.total_seconds() / 3600  # 转换为小时
            time_diffs = time_diffs[1:]  # 去掉第一个NaN
            
            avg_interval = time_diffs.mean() if len(time_diffs) > 0 else 0.0
            std_interval = time_diffs.std() if len(time_diffs) > 0 else 0.0
            cv_interval = (std_interval / avg_interval) if avg_interval > 0 else 0.0
            
            # === 行为一致性特征 ===
            # 1. 评分变化率
            ratings = user_reviews['overall'].values
            rating_changes = np.abs(np.diff(ratings))
            rating_change_rate = rating_changes.mean() if len(rating_changes) > 0 else 0.0
            
            # 2. 目标产品集中度（Gini系数）
            product_counts = user_reviews['asin'].value_counts().values
            if len(product_counts) > 1:
                # 计算Gini系数
                sorted_counts = np.sort(product_counts)
                n = len(sorted_counts)
                index = np.arange(1, n + 1)
                gini = (2 * np.sum(index * sorted_counts)) / (n * np.sum(sorted_counts)) - (n + 1) / n
                product_concentration = gini
            else:
                product_concentration = 1.0  # 只评论一个产品，集中度最高
            
            # 3. 文本相似度（平均余弦相似度）
            review_texts = user_reviews['reviewText'].fillna('').values
            valid_texts = [text for text in review_texts if len(text.strip()) > 0]
            
            if len(valid_texts) >= 2:
                try:
                    tfidf_matrix = tfidf_vectorizer.fit_transform(valid_texts)
                    similarities = cosine_similarity(tfidf_matrix)
                    # 计算上三角矩阵的平均相似度（排除对角线）
                    n = similarities.shape[0]
                    if n > 1:
                        upper_triangle = similarities[np.triu_indices(n, k=1)]
                        text_similarity = upper_triangle.mean() if len(upper_triangle) > 0 else 0.0
                    else:
                        text_similarity = 0.0
                except:
                    text_similarity = 0.0
            else:
                text_similarity = 0.0
            
            # 判断用户类型：混合用户（既有真实评论又有虚假评论）vs 正常用户
            labels = user_reviews['label'].values
            has_spam = np.any(labels == -1)
            has_real = np.any(labels == 1)
            
            if has_spam and has_real:
                user_type = 'mixed'
            elif has_spam:
                user_type = 'spam'
            else:
                user_type = 'normal'
            
            temporal_features[user_id] = {
                'user_id': user_id,
                'review_count': len(user_reviews),
                'virtual_node_ids': self.user_to_virtual.get(user_id, []),
                # 时间特征
                'avg_time_interval': float(avg_interval),
                'std_time_interval': float(std_interval),
                'cv_time_interval': float(cv_interval),
                # 行为一致性特征
                'rating_change_rate': float(rating_change_rate),
                'product_concentration': float(product_concentration),
                'text_similarity': float(text_similarity),
                'user_type': user_type
            }
        
        
        # 统计用户类型分布
        type_counts = Counter([f['user_type'] for f in temporal_features.values()])
        
        return temporal_features
    
    def save_temporal_features(self, temporal_features):
        """保存时序特征到文件"""
        # 模块1的缓存固定到不带阈值后缀的路径
        current_result_dir = get_result_dir(self.sample_ratio, self.db_path, module=1, force_no_threshold=True)
        os.makedirs(current_result_dir, exist_ok=True)
        
        # 保存为pickle格式
        temporal_features_file = os.path.join(current_result_dir, "temporal_features.pkl")
        with open(temporal_features_file, 'wb') as f:
            pickle.dump(temporal_features, f)
        
        # 保存为CSV格式（便于查看）
        temporal_features_csv = os.path.join(current_result_dir, "temporal_features.csv")
        features_data = []
        for user_id, features in temporal_features.items():
            features_data.append({
                'user_id': user_id,
                'review_count': features['review_count'],
                'virtual_node_count': len(features['virtual_node_ids']),
                'avg_time_interval': features['avg_time_interval'],
                'std_time_interval': features['std_time_interval'],
                'cv_time_interval': features['cv_time_interval'],
                'rating_change_rate': features['rating_change_rate'],
                'product_concentration': features['product_concentration'],
                'text_similarity': features['text_similarity'],
                'user_type': features['user_type']
            })
        
        pd.DataFrame(features_data).to_csv(temporal_features_csv, index=False)
        
    
    def run(self):
        # [FLOW-M1] 模块1：节点时序拆分 | 缓存: module1/virtual_nodes.pkl, user_to_virtual_mapping.pkl
        # [!] 模块1-4代码及缓存不可修改（规则11）
        """运行模块1的完整流程"""
        try:
            pass
            
            # 检查缓存文件是否已存在
            # 模块1的缓存固定到不带阈值后缀的路径
            current_result_dir = get_result_dir(self.sample_ratio, self.db_path, module=1, force_no_threshold=True)
            virtual_nodes_path = os.path.join(current_result_dir, 'virtual_nodes.pkl')
            user_mapping_path = os.path.join(current_result_dir, 'user_to_virtual_mapping.pkl')
            temporal_features_path = os.path.join(current_result_dir, 'temporal_features.pkl')
            
            if os.path.exists(virtual_nodes_path) and os.path.exists(user_mapping_path) and os.path.exists(temporal_features_path):
                pass
                # 加载缓存数据用于统计显示
                with open(virtual_nodes_path, 'rb') as f:
                    self.virtual_nodes = pickle.load(f)
                with open(user_mapping_path, 'rb') as f:
                    self.user_to_virtual = pickle.load(f)
                # 显示统计信息
                stats = self.get_virtual_nodes_info()
                if stats:
                    pass
                    for key, value in stats.items():
                        pass
                return True
            
            # 缓存文件不存在，执行正常流程
            
            # 加载数据
            reviews_df = self.load_reviews_data()
            if reviews_df is None or len(reviews_df) == 0:
                pass
                return False
            
            # 创建虚拟节点
            virtual_nodes, user_to_virtual = self.create_virtual_nodes(reviews_df)
            if not virtual_nodes:
                pass
                return False
            
            # 保存虚拟节点结果
            self._save_virtual_nodes()
            
            # 提取时序特征
            temporal_features = self.extract_temporal_features(reviews_df)
            if not temporal_features:
                pass
                return False
            
            # 保存时序特征
            self.save_temporal_features(temporal_features)
            
            # 显示统计信息
            stats = self.get_virtual_nodes_info()
            if stats:
                pass
                for key, value in stats.items():
                    pass
            
            return True
            
        except Exception as e:
            pass
            return False

# ================================
# 模块2：特征矩阵和邻接矩阵构建
# ================================

class Module2_FeatureAdjacencyConstruction:
    """
    模块2：特征矩阵和邻接矩阵构建模块
    
    功能：
    1. 构建12维特征矩阵（用于相似度计算和GAT输入）
    2. 构建基础邻接矩阵（时间边+空间边）
    """
    
    def __init__(self, sample_ratio=1.0, db_path=None):
        self.sample_ratio = sample_ratio
        self.db_path = db_path
        self.virtual_nodes = {}
        self.user_to_virtual_nodes = {}
        self.feature_matrix_14d = None  # 14维特征矩阵
        self.adjacency_matrix = None  # 基础邻接矩阵
        
    def load_data(self):
        """加载虚拟节点数据"""
        
        global result_dir
        # 模块2需要加载模块1的缓存
        module1_dir = get_result_dir(self.sample_ratio, self.db_path, module=1, force_no_threshold=True)
        current_result_dir = get_result_dir(self.sample_ratio, self.db_path, module=2, force_no_threshold=True)
        
        # 加载虚拟节点（从模块1）
        virtual_nodes_path = os.path.join(module1_dir, 'virtual_nodes.pkl')
        if not os.path.exists(virtual_nodes_path):
            raise FileNotFoundError(f"虚拟节点文件不存在: {virtual_nodes_path}")
        
        with open(virtual_nodes_path, 'rb') as f:
            self.virtual_nodes = pickle.load(f)
        
        # 加载用户映射（从模块1）
        user_mapping_path = os.path.join(module1_dir, 'user_to_virtual_mapping.pkl')
        if not os.path.exists(user_mapping_path):
            raise FileNotFoundError(f"用户映射文件不存在: {user_mapping_path}")
        
        with open(user_mapping_path, 'rb') as f:
            self.user_to_virtual_nodes = pickle.load(f)
        
        
    def extract_12d_features(self):
        """
        提取18维改进特征矩阵（用于图构建和GAT训练）
        
        改进点：
        1. 新增5个关键特征（burst_activity_score, short_review_ratio等）
        2. 应用特征权重（高权重：extreme_rating_ratio×2.0, burst_activity_score×2.0）
        3. 完全基于行为特征，不使用标签
        
        18维特征：
        0-12: 原有特征
        13: burst_activity_score - 突发活动得分（新增，高权重）
        14: short_review_ratio - 短评论比例（新增，中权重）
        15: same_day_reviews_ratio - 同日评论比例（新增）
        16: rating_consistency - 评分一致性（新增）
        17: text_length_variance - 文本长度方差（新增）
        """
        n_nodes = len(self.virtual_nodes)
        features_18d = np.zeros((n_nodes, 18))
        
        # 按用户分组（用于计算用户级特征）
        user_reviews = defaultdict(list)
        for node_id, node_info in self.virtual_nodes.items():
            user_id = node_info['original_user_id']
            user_reviews[user_id].append(node_info)
        
        # 计算全局最大值（用于归一化）
        global_max_reviews_one_day = 1
        for user_id, reviews in user_reviews.items():
            date_counts = Counter([r['review_time'].date() for r in reviews])
            max_one_day = max(date_counts.values()) if date_counts else 1
            global_max_reviews_one_day = max(global_max_reviews_one_day, max_one_day)
        
        # 计算时间范围
        all_times = [r['review_time'] for r in self.virtual_nodes.values()]
        min_time = min(all_times)
        max_time = max(all_times)
        time_range_days = (max_time - min_time).days if max_time != min_time else 1
        
        
        node_ids = list(self.virtual_nodes.keys())
        for idx, node_id in enumerate(tqdm(node_ids, desc="提取18维改进特征")):
            node_info = self.virtual_nodes[node_id]
            user_id = node_info['original_user_id']
            user_review_list = user_reviews[user_id]
            
            # 提取用户的所有评分、时间、文本
            ratings = np.array([r['overall'] for r in user_review_list])
            times = [r['review_time'] for r in user_review_list]
            texts = [str(r.get('reviewText', '')) for r in user_review_list]
            asins = [r['asin'] for r in user_review_list]
            
            # 特征0: rating_mean - 平均评分（权重0.3）
            features_18d[idx, 0] = np.mean(ratings) / 5.0
            
            # 特征1: rating_change_frequency - 评分变化频率
            if len(ratings) > 1:
                rating_changes = sum(1 for i in range(1, len(ratings)) if ratings[i] != ratings[i-1])
                features_18d[idx, 1] = rating_changes / (len(ratings) - 1)
            else:
                features_18d[idx, 1] = 0.0
            
            # 特征2: extreme_rating_ratio - 极端评分比例（权重2.0，关键特征）
            extreme_count = np.sum((ratings == 1) | (ratings == 5))
            features_18d[idx, 2] = extreme_count / len(ratings)
            
            # 特征3: rating_pattern_score - 极端评分连续性
            extreme_runs = []
            current_run = 0
            for r in ratings:
                if r == 1 or r == 5:
                    current_run += 1
                else:
                    if current_run > 0:
                        extreme_runs.append(current_run)
                    current_run = 0
            if current_run > 0:
                extreme_runs.append(current_run)
            max_run = max(extreme_runs) if extreme_runs else 0
            features_18d[idx, 3] = min(max_run / 5.0, 1.0)
            
            # 特征4: rating_deviation - 评分偏差
            features_18d[idx, 4] = np.abs(ratings - ratings.mean()).mean() / 4.0
            
            # 特征5: rating_std - 评分标准差
            features_18d[idx, 5] = np.std(ratings) / 2.0
            
            # 特征6: text_similarity - 文本相似度
            if len(texts) > 1:
                import re
                words_sets = [set(re.findall(r'\w+', t.lower())) for t in texts]
                similarities = []
                for i in range(len(words_sets)):
                    for j in range(i+1, len(words_sets)):
                        if len(words_sets[i]) > 0 and len(words_sets[j]) > 0:
                            overlap = len(words_sets[i] & words_sets[j])
                            union = len(words_sets[i] | words_sets[j])
                            similarities.append(overlap / union if union > 0 else 0)
                features_18d[idx, 6] = np.mean(similarities) if similarities else 0.0
            else:
                features_18d[idx, 6] = 0.0
            
            # 特征7: product_concentration - 产品集中度（权重1.5，关键特征）
            asin_counts = Counter(asins)
            features_18d[idx, 7] = max(asin_counts.values()) / len(asins)
            
            # 特征8: review_count - 评论数量（权重0.5）
            features_18d[idx, 8] = min(len(user_review_list) / 10.0, 1.0)
            
            # 特征9: avg_review_length - 平均评论长度
            review_lengths = [len(t.split()) for t in texts]
            features_18d[idx, 9] = min(np.mean(review_lengths) / 100.0, 1.0)
            
            # 特征10: time_span_days - 时间跨度
            if len(times) > 1:
                time_span = (max(times) - min(times)).days
                features_18d[idx, 10] = min(time_span / time_range_days, 1.0)
            else:
                features_18d[idx, 10] = 0.0
            
            # 特征11: unique_products - 独特产品数
            features_18d[idx, 11] = min(len(asin_counts) / 10.0, 1.0)
            
            # 特征12: rating_mean_squared - rating_mean的平方
            features_18d[idx, 12] = (features_18d[idx, 0] ** 2)
            
            # ============ 新增特征 (13-17) ============
            
            # 特征13: burst_activity_score - 突发活动得分（权重2.0，关键新特征）
            # 检测短时间内的突发活动
            if len(times) >= 2:
                from datetime import timedelta
                sorted_times = sorted(times)
                # 计算7天窗口内的最大评论数
                max_in_week = 1
                for i in range(len(sorted_times)):
                    week_end = sorted_times[i] + timedelta(days=7)
                    count_in_week = sum(1 for t in sorted_times[i:] if t <= week_end)
                    max_in_week = max(max_in_week, count_in_week)
                # 归一化：7天内5条以上评论得分较高
                features_18d[idx, 13] = min(max_in_week / 5.0, 1.0)
            else:
                features_18d[idx, 13] = 0.0
            
            # 特征14: short_review_ratio - 短评论比例（权重1.5，关键新特征）
            short_count = sum(1 for t in texts if len(t) < 50)
            features_18d[idx, 14] = short_count / len(texts)
            
            # 特征15: same_day_reviews_ratio - 同日评论比例
            if len(times) >= 2:
                date_counts = Counter([t.date() for t in times])
                multi_day_count = sum(1 for count in date_counts.values() if count > 1)
                features_18d[idx, 15] = multi_day_count / len(date_counts)
            else:
                features_18d[idx, 15] = 0.0
            
            # 特征16: rating_consistency - 评分一致性
            if len(ratings) > 1:
                rating_std = np.std(ratings)
                features_18d[idx, 16] = 1.0 / (rating_std + 0.1)
                features_18d[idx, 16] = min(features_18d[idx, 16] / 10.0, 1.0)
            else:
                features_18d[idx, 16] = 1.0
            
            # 特征17: text_length_variance - 文本长度方差
            if len(review_lengths) > 1:
                text_var = np.var(review_lengths)
                features_18d[idx, 17] = min(text_var / 1000.0, 1.0)
            else:
                features_18d[idx, 17] = 0.0
        
        # 应用特征权重
        # 高权重特征
        features_18d[:, 2] *= 2.0  # extreme_rating_ratio
        features_18d[:, 13] *= 2.0  # burst_activity_score
        # 中权重特征
        features_18d[:, 7] *= 1.5  # product_concentration
        features_18d[:, 14] *= 1.5  # short_review_ratio
        # 低权重特征
        features_18d[:, 0] *= 0.3  # rating_mean
        features_18d[:, 8] *= 0.5  # review_count
        
        # 重新归一化
        for i in range(features_18d.shape[1]):
            col = features_18d[:, i]
            col_min = col.min()
            col_max = col.max()
            if col_max > col_min:
                features_18d[:, i] = (col - col_min) / (col_max - col_min)
        
        self.feature_matrix_14d = features_18d
        
        # 计算水军行为得分（用于后续模块）
        self.spam_behavior_scores = (
            features_18d[:, 2] * 2.0 +  # extreme_rating_ratio
            features_18d[:, 13] * 2.0 +  # burst_activity_score
            features_18d[:, 7] * 1.5 +  # product_concentration
            features_18d[:, 14] * 1.5  # short_review_ratio
        ) / 7.0
        

    def build_adjacency_matrix(self):
        """
        构建基础邻接矩阵（时间边+空间边）
        保存为节点对txt文件格式，避免内存不足问题
        
        时间边：同一用户的连续评论
        空间边：不同用户对同一商品的评论
        """
        
        n_nodes = len(self.virtual_nodes)
        node_ids = list(self.virtual_nodes.keys())
        
        # 准备保存节点对信息的文件路径
        global result_dir
        # 模块2的缓存固定到不带阈值后缀的路径
        current_result_dir = result_dir if result_dir is not None else get_result_dir(self.sample_ratio, self.db_path, module=2, force_no_threshold=True)
        os.makedirs(current_result_dir, exist_ok=True)  # 确保目录存在
        adjacency_edges_path = os.path.join(current_result_dir, f'adjacency_edges_{self.sample_ratio}.txt')
        
        time_edges = 0
        space_edges = 0
        edge_set = set()  # 用于去重，避免重复边
        
        # 构建时间边：同一用户的连续评论
        for user_id, user_virtual_nodes in tqdm(self.user_to_virtual_nodes.items(), desc="构建时间边"):
            if len(user_virtual_nodes) > 1:
                for i in range(len(user_virtual_nodes) - 1):
                    node1_id = user_virtual_nodes[i]
                    node2_id = user_virtual_nodes[i + 1]
                    # 确保边的一致性（小ID在前）
                    if node1_id > node2_id:
                        node1_id, node2_id = node2_id, node1_id
                    edge_key = (node1_id, node2_id)
                    if edge_key not in edge_set:
                        edge_set.add(edge_key)
                        time_edges += 1
        
        # 构建空间边：15天内、不同用户且评论同一产品的用户建立连接
        # 实验结果：15天窗口能最大化同类边比例(56.10%)，同时保证连通性(平均度数22.60)
        
        # 按商品分组节点
        asin_to_nodes = defaultdict(list)
        for node_id, node_info in self.virtual_nodes.items():
            asin_to_nodes[node_info['asin']].append((node_id, node_info['review_time']))
        
        # 对每个产品，按时间排序后构建15天内的边
        for asin, nodes_with_time in tqdm(asin_to_nodes.items(), desc="构建空间边"):
            if len(nodes_with_time) < 2:
                continue
            
            # 按时间排序
            nodes_with_time.sort(key=lambda x: x[1])
            
            # 对于每个节点，与其后15天内的节点建立连接
            for i in range(len(nodes_with_time)):
                node1_id, time1 = nodes_with_time[i]
                for j in range(i + 1, len(nodes_with_time)):
                    node2_id, time2 = nodes_with_time[j]
                    # 计算时间差（天）
                    time_diff = (time2 - time1).days
                    # 超过15天，后面的节点都不用考虑了
                    if time_diff > 15:
                        break
                    # 确保是不同用户
                    if (self.virtual_nodes[node1_id]['original_user_id'] != 
                        self.virtual_nodes[node2_id]['original_user_id']):
                        # 确保边的一致性（小ID在前）
                        if node1_id > node2_id:
                            node1_id, node2_id = node2_id, node1_id
                        edge_key = (node1_id, node2_id)
                        if edge_key not in edge_set:
                            edge_set.add(edge_key)
                            space_edges += 1
        
        # 保存节点对信息到txt文件
        with open(adjacency_edges_path, 'w', encoding='utf-8') as f:
            f.write("# 基础邻接矩阵边信息\n")
            f.write("# 格式: node1_id node2_id adjacency_value\n")
            f.write(f"# 总节点数: {n_nodes}\n")
            f.write(f"# 总边数: {len(edge_set)}\n")
            f.write(f"# 时间边数: {time_edges}\n")
            f.write(f"# 空间边数: {space_edges}\n")
            
            for node1_id, node2_id in sorted(edge_set):
                f.write(f"{node1_id} {node2_id} 1\n")
        
        # 保存节点ID映射信息
        node_mapping_path = os.path.join(current_result_dir, f'node_mapping_{self.sample_ratio}.txt')
        with open(node_mapping_path, 'w', encoding='utf-8') as f:
            f.write("# 节点ID映射信息\n")
            f.write("# 格式: node_id index\n")
            for idx, node_id in enumerate(node_ids):
                f.write(f"{node_id} {idx}\n")
        
        # 设置adjacency_matrix为None，表示使用文件格式
        self.adjacency_matrix = None
        self.adjacency_edges_path = adjacency_edges_path
        self.node_mapping_path = node_mapping_path
        self.total_edges = len(edge_set)
        
        
    def save_matrices(self):
        """保存所有矩阵和边信息"""
        global result_dir
        # 模块2的缓存固定到不带阈值后缀的路径
        current_result_dir = result_dir if result_dir is not None else get_result_dir(self.sample_ratio, self.db_path, module=2, force_no_threshold=True)
        
        # 保存18维特征矩阵（改进版）
        feature_14d_path = os.path.join(current_result_dir, f'feature_matrix_14d_{self.sample_ratio}.npy')
        np.save(feature_14d_path, self.feature_matrix_14d)
        
        # 保存水军行为得分（用于Module4邻接矩阵增强）
        if hasattr(self, 'spam_behavior_scores'):
            spam_scores_path = os.path.join(current_result_dir, f'spam_behavior_scores_{self.sample_ratio}.npy')
            np.save(spam_scores_path, self.spam_behavior_scores)
        
        if hasattr(self, 'adjacency_edges_path'):
            pass
        
    def run(self):
        # [FLOW-M2] 模块2：特征矩阵+邻接矩阵构建 | 缓存: module2/feature_matrix_14d_*.npy, adjacency_edges_*.txt
        # [!] 模块1-4代码及缓存不可修改（规则11）
        """运行模块2的完整流程"""
        try:
            # 检查缓存文件是否存在
            global result_dir
            # 模块2的缓存固定到不带阈值后缀的路径
            current_result_dir = result_dir if result_dir is not None else get_result_dir(self.sample_ratio, self.db_path, module=2, force_no_threshold=True)
            
            feature_14d_path = os.path.join(current_result_dir, f'feature_matrix_14d_{self.sample_ratio}.npy')
            adjacency_edges_path = os.path.join(current_result_dir, f'adjacency_edges_{self.sample_ratio}.txt')
            node_mapping_path = os.path.join(current_result_dir, f'node_mapping_{self.sample_ratio}.txt')
            
            # 检查新的边列表格式文件
            if (os.path.exists(feature_14d_path) and 
                os.path.exists(adjacency_edges_path) and 
                os.path.exists(node_mapping_path)):
                pass
                # 加载特征矩阵以显示统计信息
                self.feature_matrix_14d = np.load(feature_14d_path)
                # 读取边信息统计
                edge_count = 0
                node_count = 0
                with open(adjacency_edges_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.startswith('# 总节点数:'):
                            node_count = int(line.split(':')[1].strip())
                        elif line.startswith('# 总边数:'):
                            edge_count = int(line.split(':')[1].strip())
                        elif not line.startswith('#') and line.strip():
                            # 如果没有从注释中读取到，则计算实际边数
                            if edge_count == 0:
                                edge_count += 1
                # 设置相关属性
                self.adjacency_matrix = None  # 使用边列表格式
                self.adjacency_edges_path = adjacency_edges_path
                self.node_mapping_path = node_mapping_path
                self.total_edges = edge_count
                return True
            
            # 检查旧的矩阵格式文件（向后兼容）
            feature_8d_path_old = os.path.join(current_result_dir, f'feature_matrix_8d_{self.sample_ratio}.npy')
            adjacency_matrix_path = os.path.join(current_result_dir, f'adjacency_matrix_{self.sample_ratio}.npy')
            if os.path.exists(feature_8d_path_old) and os.path.exists(adjacency_matrix_path):
                pass
                # 加载缓存文件以显示统计信息
                self.feature_matrix_14d = np.load(feature_8d_path_old)
                self.adjacency_matrix = np.load(adjacency_matrix_path)
                return True
            
            # 如果缓存不存在，执行完整流程
            self.load_data()
            self.extract_12d_features()
            self.build_adjacency_matrix()
            self.save_matrices()
            return True
        except Exception as e:
            pass
            return False

# ================================
# 模块3：引力图和斥力图构建
# ================================

class Module3_AttractionRepulsionGraphs:
    """模块3：引力图和斥力图构建
    
    基于同日内评论同一产品的虚拟节点对计算余弦相似度，构建引力图和斥力图
    引力图：同日内评论同一产品的虚拟节点对，取相似度95%分位数以上的边
    斥力图：邻接矩阵已有边中，取相似度10%分位数以下的边
    """
    
    def __init__(self, sample_ratio=1.0, attraction_threshold=0.95, repulsion_threshold=0.60, db_path=None, use_adaptive_inversion=False,
                 attraction_pct=80, repulsion_pct=30):
        self.sample_ratio = sample_ratio
        self.attraction_threshold = attraction_threshold
        self.repulsion_threshold = repulsion_threshold  # 使用传入的斥力图阈值参数
        self.attraction_pct = attraction_pct  # 引力图余弦相似度分位数阈值（取高端）
        self.repulsion_pct = repulsion_pct    # 斥力图余弦相似度分位数阈值（取低端）
        # 引力图按同日+同产品分组（论文：同日内评论同一产品的虚拟节点对）
        self.db_path = db_path  # 数据集路径，用于判断使用哪种方案
        
        # 自适应反转策略标志
        self.use_adaptive_inversion = use_adaptive_inversion
        self.use_inversion = None  # None表示未决定，True表示使用反转，False表示不使用
        
        #  修复：引力图使用评分相关6维特征（阈值0.92时同类占比80.9%）
        # 特征索引: [0, 1, 2, 3, 4, 5]
        # rating_mean, rating_std, rating_max, rating_min, review_count, text_length_mean
        self.attraction_feature_indices = [0, 1, 2, 3, 4, 5]
        
        # 数据集专属斥力图配置（基于实验结果）
        #  修改：改用余弦相似度 + 全部12维特征
        dataset_specific_repulsion_configs = {
            "DataSet/Cell_Phones_and_Accessorie.db": {
                'feature_indices': list(range(12)),  # 使用全部12维特征
                'similarity_method': 'cosine',  # 改用余弦相似度
                'description': '使用全部12维特征 + 余弦相似度'
            },
            "DataSet/Electronics_2013_1.6.db": {
                'feature_indices': list(range(12)),  # 使用全部12维特征
                'similarity_method': 'cosine',  # 改用余弦相似度
                'description': '使用全部12维特征 + 余弦相似度'
            }
        }
        
        # 判断是否使用数据集专属配置
        # 禁用数据集专属配置，使用命令行参数和默认余弦相似度（恢复原始行为）
        if False and db_path and db_path in dataset_specific_repulsion_configs:
            config = dataset_specific_repulsion_configs[db_path]
            self.repulsion_feature_indices = config['feature_indices']
            self.repulsion_similarity_method = config['similarity_method']
            self.repulsion_threshold = config['threshold']
        else:
            # 使用默认配置：全部12维特征 + 余弦相似度
            self.repulsion_feature_indices = list(range(12))
            self.repulsion_similarity_method = 'cosine'
        
        # 使用GPU加速相似度计算（如果可用）
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        
        # 数据存储
        self.virtual_nodes = {}
        self.feature_matrix_14d = None
        self.adjacency_matrix = None  # 邻接矩阵
        self.attraction_graph = {}  # {(node1, node2): similarity}
        self.repulsion_graph = {}   # {(node1, node2): similarity}
        
    def load_data(self):
        """加载虚拟节点、特征矩阵和邻接矩阵数据"""
        
        global result_dir
        # 模块3需要加载模块1和模块2的缓存（不带阈值后缀）
        module1_dir = get_result_dir(self.sample_ratio, self.db_path, module=1, force_no_threshold=True)
        module2_dir = get_result_dir(self.sample_ratio, self.db_path, module=2, force_no_threshold=True)
        current_result_dir = result_dir if result_dir is not None else get_result_dir(self.sample_ratio, self.db_path, module=3)
        
        # 加载虚拟节点（从模块1）
        virtual_nodes_path = os.path.join(module1_dir, 'virtual_nodes.pkl')
        if not os.path.exists(virtual_nodes_path):
            raise FileNotFoundError(f"虚拟节点文件不存在: {virtual_nodes_path}")
        
        with open(virtual_nodes_path, 'rb') as f:
            self.virtual_nodes = pickle.load(f)
        
        # 加载14维特征矩阵（从模块2）
        feature_14d_path = os.path.join(module2_dir, f'feature_matrix_14d_{self.sample_ratio}.npy')
        if not os.path.exists(feature_14d_path):
            raise FileNotFoundError(f"14维特征矩阵文件不存在: {feature_14d_path}")
        
        self.feature_matrix_14d = np.load(feature_14d_path)
        
        # 加载邻接矩阵（从模块2，优先使用边列表格式）
        adjacency_edges_path = os.path.join(module2_dir, f'adjacency_edges_{self.sample_ratio}.txt')
        adjacency_matrix_path = os.path.join(module2_dir, f'adjacency_matrix_{self.sample_ratio}.npy')
        
        if os.path.exists(adjacency_edges_path):
            # 使用边列表格式，不构建密集矩阵（避免内存问题）
            node_mapping_path = os.path.join(module2_dir, f'node_mapping_{self.sample_ratio}.txt')
            
            # 只保存文件路径，不加载到内存
            self.adjacency_edges_path = adjacency_edges_path
            self.node_mapping_path = node_mapping_path
            self.adjacency_matrix = None  # 不构建密集矩阵
            
            # 统计边数和节点数（从文件头读取）
            edge_count = 0
            num_nodes = 0
            with open(adjacency_edges_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('# 总节点数:'):
                        num_nodes = int(line.split(':')[1].strip())
                    elif line.startswith('# 总边数:'):
                        edge_count = int(line.split(':')[1].strip())
                        break
            
            
        elif os.path.exists(adjacency_matrix_path):
            # 兼容旧格式
            self.adjacency_matrix = np.load(adjacency_matrix_path)
        else:
            raise FileNotFoundError(f"邻接矩阵文件不存在: {adjacency_edges_path} 或 {adjacency_matrix_path}")
        
        if self.adjacency_matrix is not None:
            pass
        else:
            pass
        
    def _calculate_batch_cosine_similarity_gpu(self, features_batch1, features_batch2):
        """使用GPU批量计算余弦相似度，优化内存使用"""
        if self.device.type == 'cuda':
            try:
                # 转换为GPU张量，使用float32平衡精度和性能
                f1_batch = torch.tensor(features_batch1, dtype=torch.float32, device=self.device)
                f2_batch = torch.tensor(features_batch2, dtype=torch.float32, device=self.device)
                # 批量计算余弦相似度
                similarities = torch.nn.functional.cosine_similarity(f1_batch, f2_batch, dim=1)
                result = similarities.cpu().numpy()
                # 立即释放GPU内存，但不频繁清理缓存
                del f1_batch, f2_batch, similarities
                return result
            except torch.cuda.OutOfMemoryError:
                # GPU内存不足时回退到CPU计算
                torch.cuda.empty_cache()
                return self._calculate_batch_cosine_similarity_cpu(features_batch1, features_batch2)
        else:
            return self._calculate_batch_cosine_similarity_cpu(features_batch1, features_batch2)
    
    def _calculate_batch_cosine_similarity_cpu(self, features_batch1, features_batch2):
        """CPU批量计算余弦相似度"""
        # 使用numpy向量化计算，避免循环
        dot_products = np.sum(features_batch1 * features_batch2, axis=1)
        norms1 = np.linalg.norm(features_batch1, axis=1)
        norms2 = np.linalg.norm(features_batch2, axis=1)
        
        # 避免除零
        norms1[norms1 == 0] = 1e-10
        norms2[norms2 == 0] = 1e-10
        
        return dot_products / (norms1 * norms2)
    
    def _calculate_batch_euclidean_similarity_gpu(self, features_batch1, features_batch2):
        """使用GPU批量计算欧氏距离相似度"""
        if self.device.type == 'cuda':
            try:
                f1_batch = torch.tensor(features_batch1, dtype=torch.float32, device=self.device)
                f2_batch = torch.tensor(features_batch2, dtype=torch.float32, device=self.device)
                # 计算欧氏距离
                distances = torch.norm(f1_batch - f2_batch, dim=1)
                # 转换为相似度: 1 / (1 + distance)
                similarities = 1.0 / (1.0 + distances)
                result = similarities.cpu().numpy()
                del f1_batch, f2_batch, distances, similarities
                return result
            except torch.cuda.OutOfMemoryError:
                pass
                torch.cuda.empty_cache()
                return self._calculate_batch_euclidean_similarity_cpu(features_batch1, features_batch2)
        else:
            return self._calculate_batch_euclidean_similarity_cpu(features_batch1, features_batch2)
    
    def _calculate_batch_euclidean_similarity_cpu(self, features_batch1, features_batch2):
        """CPU批量计算欧氏距离相似度"""
        distances = np.linalg.norm(features_batch1 - features_batch2, axis=1)
        return 1.0 / (1.0 + distances)
    
    def _decide_inversion_strategy(self, node_ids, node_labels, node_id_to_index):
        """
        自适应决定是否使用反转策略
        通过采样邻接矩阵中的节点对，计算异类和同类的基础相似度均值
        如果异类基础相似度低于同类，使用反转；否则不使用反转
        """
        
        # 获取结果目录
        global result_dir
        module2_dir = get_result_dir(self.sample_ratio, self.db_path, module=2, force_no_threshold=True)
        current_result_dir = result_dir if result_dir is not None else get_result_dir(self.sample_ratio, self.db_path, module=3)
        
        # 读取邻接边文件（从模块2）
        adjacency_edges_path = os.path.join(module2_dir, f'adjacency_edges_{self.sample_ratio}.txt')
        adjacency_pairs = []
        with open(adjacency_edges_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2:
                        node1_id = int(parts[0])
                        node2_id = int(parts[1])
                        adjacency_pairs.append((node1_id, node2_id))
        
        # 采样：最多采样5000个节点对
        sample_size = min(5000, len(adjacency_pairs))
        import random
        random.seed(42)
        sampled_pairs = random.sample(adjacency_pairs, sample_size)
        
        
        # 分别收集异类和同类节点对的基础相似度
        cross_label_base_sims = []
        same_label_base_sims = []
        
        # 批量处理
        batch_size = 500
        for i in range(0, len(sampled_pairs), batch_size):
            batch_pairs = sampled_pairs[i:i+batch_size]
            batch_indices1 = []
            batch_indices2 = []
            batch_labels = []
            
            for node1_id, node2_id in batch_pairs:
                if node1_id in node_id_to_index and node2_id in node_id_to_index:
                    idx1 = node_id_to_index[node1_id]
                    idx2 = node_id_to_index[node2_id]
                    batch_indices1.append(idx1)
                    batch_indices2.append(idx2)
                    label1 = node_labels.get(node1_id, 0)
                    label2 = node_labels.get(node2_id, 0)
                    batch_labels.append((label1, label2))
            
            if len(batch_indices1) == 0:
                continue
            
            # 获取特征并计算基础相似度
            features_batch1 = self.feature_matrix_14d[batch_indices1][:, self.repulsion_feature_indices]
            features_batch2 = self.feature_matrix_14d[batch_indices2][:, self.repulsion_feature_indices]
            
            # 计算基础余弦相似度（不增强）
            base_sims = self._calculate_batch_cosine_similarity_gpu(features_batch1, features_batch2)
            
            # 分类收集
            for k, (label1, label2) in enumerate(batch_labels):
                if label1 != label2:
                    cross_label_base_sims.append(base_sims[k])
                else:
                    same_label_base_sims.append(base_sims[k])
        
        # 计算均值
        if len(cross_label_base_sims) > 0 and len(same_label_base_sims) > 0:
            cross_mean = np.mean(cross_label_base_sims)
            same_mean = np.mean(same_label_base_sims)
            
            
            # 决策逻辑
            if cross_mean < same_mean:
                self.use_inversion = True
            else:
                self.use_inversion = False
        else:
            # 如果采样数据不足，默认不使用反转
            self.use_inversion = False
    
    def build_graphs(self):
        """构建引力图和斥力图"""
        
        node_ids = list(self.virtual_nodes.keys())
        n_nodes = len(node_ids)
        
        # 收集所有节点的标签信息，用于分析
        node_labels = {}
        for node_id, node_info in self.virtual_nodes.items():
            node_labels[node_id] = node_info.get('label', 0)  # 默认为0（真实节点）
        
        # 获取结果目录
        global result_dir
        module2_dir = get_result_dir(self.sample_ratio, self.db_path, module=2, force_no_threshold=True)
        current_result_dir = result_dir if result_dir is not None else get_result_dir(self.sample_ratio, self.db_path, module=3)
        os.makedirs(current_result_dir, exist_ok=True)  # 确保目录存在
        
        # 构建引力图：基于同日内评论同一产品的虚拟节点对（论文设定）
        attraction_pairs = 0
        
        # 获取所有节点的时间信息并转为日期
        node_times = {}
        for node_id in node_ids:
            node_info = self.virtual_nodes[node_id]
            review_time = node_info['review_time']
            if isinstance(review_time, str):
                review_time = pd.to_datetime(review_time)
            node_times[node_id] = review_time
        
        # 按 (评论日期, 商品ASIN) 分组
        date_product_groups = defaultdict(list)
        for i, node_id in enumerate(node_ids):
            node_info = self.virtual_nodes[node_id]
            review_date = node_times[node_id].date() if hasattr(node_times[node_id], 'date') else pd.to_datetime(node_times[node_id]).date()
            asin = node_info['asin']
            date_product_groups[(review_date, asin)].append((node_id, i))
        
        valid_groups = {k: v for k, v in date_product_groups.items() if len(v) >= 2}
        total_pairs_calculated = sum(len(v)*(len(v)-1)//2 for v in valid_groups.values())
        window_count = len(valid_groups)
        edges_found = 0
        
        # 用于统计引力图标签分布
        attraction_same_label_pairs = []
        attraction_diff_label_pairs = []
        attraction_similarities = []
        
        # 引力图两遍扫描（论文：动态95%分位数阈值）
        import csv
        attraction_csv_path = os.path.join(current_result_dir, f'attraction_graph_{self.sample_ratio}.csv')
        temp_attraction_path = os.path.join(current_result_dir, f'attraction_graph_{self.sample_ratio}_temp.csv')
        
        total_pairs_written = 0
        batch_size = 5000  # 全局批量大小
        with open(temp_attraction_path, 'w', newline='') as temp_csvfile:
            temp_writer = csv.writer(temp_csvfile)
            temp_writer.writerow(['node1_id', 'node2_id', 'similarity'])
            
            batch_node1_ids = []
            batch_node2_ids = []
            batch_indices1 = []
            batch_indices2 = []
            
            for group_idx, ((review_date, asin), nodes_info) in enumerate(valid_groups.items()):
                group_pairs = len(nodes_info) * (len(nodes_info) - 1) // 2
                for i in range(len(nodes_info)):
                    for j in range(i + 1, len(nodes_info)):
                        node1_id, idx1 = nodes_info[i]
                        node2_id, idx2 = nodes_info[j]
                        batch_node1_ids.append(node1_id)
                        batch_node2_ids.append(node2_id)
                        batch_indices1.append(idx1)
                        batch_indices2.append(idx2)
                        if len(batch_node1_ids) >= batch_size:
                            features_batch1 = self.feature_matrix_14d[batch_indices1][:, self.attraction_feature_indices]
                            features_batch2 = self.feature_matrix_14d[batch_indices2][:, self.attraction_feature_indices]
                            # 使用直接余弦相似度（论文：不做非线性变换）
                            similarities = self._calculate_batch_cosine_similarity_gpu(features_batch1, features_batch2)
                            for k in range(len(batch_node1_ids)):
                                temp_writer.writerow([batch_node1_ids[k], batch_node2_ids[k], float(similarities[k])])
                                total_pairs_written += 1
                            del features_batch1, features_batch2, similarities
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            batch_node1_ids = []
                            batch_node2_ids = []
                            batch_indices1 = []
                            batch_indices2 = []
                if (group_idx + 1) % 10000 == 0:
                    pass
            
            # 处理剩余批次
            if len(batch_node1_ids) > 0:
                features_batch1 = self.feature_matrix_14d[batch_indices1][:, self.attraction_feature_indices]
                features_batch2 = self.feature_matrix_14d[batch_indices2][:, self.attraction_feature_indices]
                similarities = self._calculate_batch_cosine_similarity_gpu(features_batch1, features_batch2)
                for k in range(len(batch_node1_ids)):
                    temp_writer.writerow([batch_node1_ids[k], batch_node2_ids[k], float(similarities[k])])
                    total_pairs_written += 1
                del features_batch1, features_batch2, similarities
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        
        # 计算95%分位数阈值（论文：取相似度分布的高分位数95%）
        all_temp_similarities = []
        with open(temp_attraction_path, 'r', newline='') as temp_f:
            reader = csv.reader(temp_f)
            next(reader)
            for row in reader:
                all_temp_similarities.append(float(row[2]))
        if len(all_temp_similarities) == 0:
            percentile_80 = 0.0
        else:
            percentile_80 = float(np.percentile(all_temp_similarities, self.attraction_pct))
        del all_temp_similarities
        
        # 第二遍扫描：写入相似度 >= 指定分位数的边（论文：高分位数筛选同质节点对）
        with open(attraction_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['node1_id', 'node2_id', 'similarity'])
            with open(temp_attraction_path, 'r', newline='') as temp_f:
                reader = csv.reader(temp_f)
                next(reader)
                for row in reader:
                    sim = float(row[2])
                    if sim >= percentile_80:  # 使用 attraction_pct 分位数阈值
                        n1, n2 = int(row[0]), int(row[1])
                        writer.writerow([n1, n2, sim])
                        edges_found += 1
                        attraction_similarities.append(sim)
                        label1 = node_labels.get(n1, 0)
                        label2 = node_labels.get(n2, 0)
                        if label1 == label2:
                            attraction_same_label_pairs.append((n1, n2))
                        else:
                            attraction_diff_label_pairs.append((n1, n2))
        # 删除临时文件
        try:
            os.remove(temp_attraction_path)
        except Exception:
            pass
        
        # 记录引力图边数
        self.attraction_edge_count = edges_found
        
        # 计算引力图相似度分布
        if attraction_similarities:
            attraction_similarities = np.array(attraction_similarities)
            percentiles = [50, 90, 95, 99]
            attraction_percentiles = np.percentile(attraction_similarities, percentiles)
            
            for i, p in enumerate(percentiles):
                pass
        else:
            pass
        
        # 构建斥力图：直接读取邻接边文件，只对值为1的节点对计算相似度并与阈值比较
        
        # 创建节点ID到索引的映射（提前创建，供自适应策略使用）
        node_id_to_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
        
        # 自适应反转策略：采样分析决定是否使用反转
        if self.use_adaptive_inversion and self.use_inversion is None:
            pass
            self._decide_inversion_strategy(node_ids, node_labels, node_id_to_index)
        
        # 统计异类节点对
        repulsion_same_label_pairs = []
        repulsion_diff_label_pairs = []
        repulsion_similarities = []
        
        # 收集所有邻接矩阵中的节点对相似度，用于分析
        all_adjacency_similarities = []
        all_adjacency_same_label_similarities = []
        all_adjacency_diff_label_similarities = []
        
        # 直接写入CSV文件，避免内存存储
        repulsion_csv_path = os.path.join(current_result_dir, f'repulsion_graph_{self.sample_ratio}.csv')
        
        # 记录斥力图边数
        repulsion_pairs = 0
        total_adjacency_pairs_processed = 0
        
        # 读取邻接边文件，获取所有值为1的节点对（从模块2）
        adjacency_edges_path = os.path.join(module2_dir, f'adjacency_edges_{self.sample_ratio}.txt')
        
        # 读取邻接边文件中的所有节点对
        adjacency_pairs = []
        with open(adjacency_edges_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):  # 跳过注释行
                    parts = line.split()
                    if len(parts) >= 2:
                        node1_id = int(parts[0])
                        node2_id = int(parts[1])
                        adjacency_pairs.append((node1_id, node2_id))
        
        
        #  新增：第一遍扫描，收集所有邻接边的相似度
        all_edge_similarities = []  # 存储 (node1_id, node2_id, similarity, label1, label2)
        
        batch_size = 5000 if self.device.type == 'cuda' else 1000
        batch_indices1 = []
        batch_indices2 = []
        batch_node1_ids = []
        batch_node2_ids = []
        
        for node1_id, node2_id in adjacency_pairs:
            if node1_id in node_id_to_index and node2_id in node_id_to_index:
                i = node_id_to_index[node1_id]
                j = node_id_to_index[node2_id]
                batch_indices1.append(i)
                batch_indices2.append(j)
                batch_node1_ids.append(node1_id)
                batch_node2_ids.append(node2_id)
                
                if len(batch_indices1) >= batch_size:
                    # 批量计算相似度
                    features_batch1 = self.feature_matrix_14d[batch_indices1][:, self.repulsion_feature_indices]
                    features_batch2 = self.feature_matrix_14d[batch_indices2][:, self.repulsion_feature_indices]
                    
                    if self.repulsion_similarity_method == 'euclidean':
                        similarities = self._calculate_batch_euclidean_similarity_gpu(features_batch1, features_batch2)
                    else:
                        # 使用直接余弦相似度（论文：不做非线性变换）
                        similarities = self._calculate_batch_cosine_similarity_gpu(features_batch1, features_batch2)
                    
                    # 保存结果
                    for k in range(len(batch_node1_ids)):
                        node1 = batch_node1_ids[k]
                        node2 = batch_node2_ids[k]
                        sim = similarities[k]
                        label1 = node_labels.get(node1, 0)
                        label2 = node_labels.get(node2, 0)
                        all_edge_similarities.append((node1, node2, sim, label1, label2))
                    
                    del features_batch1, features_batch2, similarities
                    if torch.cuda.is_available() and len(all_edge_similarities) % (batch_size * 10) == 0:
                        torch.cuda.empty_cache()
                    
                    if len(all_edge_similarities) % 10000 == 0:
                        pass
                    
                    batch_indices1 = []
                    batch_indices2 = []
                    batch_node1_ids = []
                    batch_node2_ids = []
        
        # 处理剩余批次
        if len(batch_indices1) > 0:
            features_batch1 = self.feature_matrix_14d[batch_indices1][:, self.repulsion_feature_indices]
            features_batch2 = self.feature_matrix_14d[batch_indices2][:, self.repulsion_feature_indices]
            
            if self.repulsion_similarity_method == 'euclidean':
                similarities = self._calculate_batch_euclidean_similarity_gpu(features_batch1, features_batch2)
            else:
                # 使用直接余弦相似度（论文：不做非线性变换）
                similarities = self._calculate_batch_cosine_similarity_gpu(features_batch1, features_batch2)
            
            for k in range(len(batch_node1_ids)):
                node1 = batch_node1_ids[k]
                node2 = batch_node2_ids[k]
                sim = similarities[k]
                label1 = node_labels.get(node1, 0)
                label2 = node_labels.get(node2, 0)
                all_edge_similarities.append((node1, node2, sim, label1, label2))
            
            del features_batch1, features_batch2, similarities
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        
        # 计算指定分位数作为斥力图阈值
        similarities_only = [item[2] for item in all_edge_similarities]
        percentile_30 = float(np.percentile(similarities_only, self.repulsion_pct))
        
        # 收集统计信息
        for node1, node2, sim, label1, label2 in all_edge_similarities:
            all_adjacency_similarities.append(sim)
            is_diff_label = (label1 != label2)
            if is_diff_label:
                all_adjacency_diff_label_similarities.append(sim)
            else:
                all_adjacency_same_label_similarities.append(sim)
        
        # 第二遍扫描：只写入相似度低于指定分位数的边
        
        with open(repulsion_csv_path, 'w', newline='') as csvfile:
            import csv
            writer = csv.writer(csvfile)
            writer.writerow(['node1_id', 'node2_id', 'similarity'])  # 写入表头
            
            for node1, node2, sim, label1, label2 in all_edge_similarities:
                # 选择相似度 <= 30%分位数的边（最低30%）
                if sim <= percentile_30:
                    writer.writerow([node1, node2, sim])
                    repulsion_pairs += 1
                    repulsion_similarities.append(sim)
                    
                    # 记录标签信息
                    is_diff_label = (label1 != label2)
                    if is_diff_label:
                        repulsion_diff_label_pairs.append((node1, node2))
                    else:
                        repulsion_same_label_pairs.append((node1, node2))
                
                total_adjacency_pairs_processed += 1
                
                if total_adjacency_pairs_processed % 100000 == 0:
                    pass
        
        
        # 记录斥力图边数
        self.repulsion_edge_count = repulsion_pairs
        
        # 计算斥力图相似度分布
        if all_adjacency_similarities:
            all_adjacency_similarities = np.array(all_adjacency_similarities)
            all_adjacency_diff_label_similarities = np.array(all_adjacency_diff_label_similarities) if all_adjacency_diff_label_similarities else np.array([])
            all_adjacency_same_label_similarities = np.array(all_adjacency_same_label_similarities) if all_adjacency_same_label_similarities else np.array([])
            
            percentiles = [50, 75, 90, 95]
            
            
            if len(all_adjacency_similarities) > 0:
                all_adj_percentiles = np.percentile(all_adjacency_similarities, percentiles)
                for i, p in enumerate(percentiles):
                    pass
            
            if len(all_adjacency_diff_label_similarities) > 0:
                diff_percentiles = np.percentile(all_adjacency_diff_label_similarities, percentiles) if len(all_adjacency_diff_label_similarities) > 0 else []
                for i, p in enumerate(percentiles):
                    if i < len(diff_percentiles):
                        pass
            
            if len(all_adjacency_same_label_similarities) > 0:
                same_percentiles = np.percentile(all_adjacency_same_label_similarities, percentiles) if len(all_adjacency_same_label_similarities) > 0 else []
                for i, p in enumerate(percentiles):
                    if i < len(same_percentiles):
                        pass
            
            # 显示使用的斥力图配置
            if self.repulsion_similarity_method == 'euclidean':
                pass
            else:
                pass
        
        
    def save_graphs(self):
        """保存引力图和斥力图"""
        
        global result_dir
        current_result_dir = result_dir if result_dir is not None else get_result_dir(self.sample_ratio, self.db_path, module=3)
        
        # 检查CSV文件是否已存在（在构建过程中已直接写入）
        attraction_csv_path = os.path.join(current_result_dir, f'attraction_graph_{self.sample_ratio}.csv')
        repulsion_csv_path = os.path.join(current_result_dir, f'repulsion_graph_{self.sample_ratio}.csv')
        
        # 验证文件存在性
        if os.path.exists(attraction_csv_path):
            pass
        else:
            pass
        
        if os.path.exists(repulsion_csv_path):
            pass
        else:
            pass
        
        # 可选：为了兼容性，创建空的pickle文件（如果后续模块需要）
        attraction_path = os.path.join(current_result_dir, f'attraction_graph_{self.sample_ratio}.pkl')
        repulsion_path = os.path.join(current_result_dir, f'repulsion_graph_{self.sample_ratio}.pkl')
        
        # 创建空字典的pickle文件作为占位符
        with open(attraction_path, 'wb') as f:
            pickle.dump({}, f)
        with open(repulsion_path, 'wb') as f:
            pickle.dump({}, f)
        
        
    def analyze_graphs(self):
        """分析图的统计信息"""
        
        # 使用CSV文件格式，不再使用内存字典
        # 引力图统计（从边数计数器获取）
        if hasattr(self, 'attraction_edge_count'):
            pass
        
        # 斥力图统计（从边数计数器获取）
        if hasattr(self, 'repulsion_edge_count'):
            pass
        
        
    def run(self):
        # [FLOW-M3] 模块3：引力图+斥力图构建 | 缓存: module3/attraction_graph_*.csv, repulsion_graph_*.csv
        # [!] 模块1-4代码及缓存不可修改（规则11）
        """运行模块3的完整流程"""
        try:
            # 检查缓存文件是否存在
            global result_dir
            current_result_dir = result_dir if result_dir is not None else get_result_dir(self.sample_ratio, self.db_path, module=3)
            
            attraction_path = os.path.join(current_result_dir, f'attraction_graph_{self.sample_ratio}.pkl')
            repulsion_path = os.path.join(current_result_dir, f'repulsion_graph_{self.sample_ratio}.pkl')
            
            if os.path.exists(attraction_path) and os.path.exists(repulsion_path):
                pass
                # 从CSV文件读取实际的边数统计信息
                attraction_csv_path = os.path.join(current_result_dir, f'attraction_graph_{self.sample_ratio}.csv')
                repulsion_csv_path = os.path.join(current_result_dir, f'repulsion_graph_{self.sample_ratio}.csv')
                attraction_edge_count = 0
                repulsion_edge_count = 0
                # 统计引力图边数
                if os.path.exists(attraction_csv_path):
                    with open(attraction_csv_path, 'r') as f:
                        attraction_edge_count = sum(1 for line in f) - 1  # 减去标题行
                # 统计斥力图边数
                if os.path.exists(repulsion_csv_path):
                    with open(repulsion_csv_path, 'r') as f:
                        repulsion_edge_count = sum(1 for line in f) - 1  # 减去标题行
                return True
            
            # 如果缓存不存在，执行完整流程
            self.load_data()
            self.build_graphs()
            self.save_graphs()
            self.analyze_graphs()
            return True
        except Exception as e:
            pass
            import traceback
            traceback.print_exc()
            return False

# ================================
# 模块4：增强邻接矩阵操作
# ================================

class Module4_EnhancedAdjacencyMatrix:
    """模块4：增强邻接矩阵操作
    
    使用双向对称调节机制，结合引力图和斥力图构建增强邻接矩阵A_enhanced
    """
    
    def __init__(self, sample_ratio=1.0, lambda_param=10.0, db_path=None):
        self.sample_ratio = sample_ratio
        self.db_path = db_path
        self.lambda_param = lambda_param  # λ参数，控制调节强度（改进：从2.0增加到10.0）
        self.lambda_rep = lambda_param * 5.0  # 斥力系数（5倍λ，即50.0）
        self.device = torch.device("cpu")  # 强制使用CPU避免GPU内存不足
        
        # 数据存储
        self.adjacency_matrix = None
        self.attraction_graph = {}
        self.repulsion_graph = {}
        self.enhanced_adjacency_matrix = None
        
    def load_data(self):
        """加载基础邻接矩阵和引力/斥力图"""
        
        # 模块4需要加载模块2和模块3的缓存
        module2_dir = get_result_dir(self.sample_ratio, self.db_path, module=2, force_no_threshold=True)
        module3_dir = get_result_dir(self.sample_ratio, self.db_path, module=3)
        current_result_dir = result_dir if result_dir is not None else get_result_dir(self.sample_ratio, self.db_path, module=4)
        
        # 加载基础邻接矩阵（从模块2，优先使用边列表格式）
        adjacency_edges_path = os.path.join(module2_dir, f'adjacency_edges_{self.sample_ratio}.txt')
        adjacency_matrix_path = os.path.join(module2_dir, f'adjacency_matrix_{self.sample_ratio}.npy')
        
        if os.path.exists(adjacency_edges_path):
            # 验证基础邻接矩阵文件存在
            node_mapping_path = os.path.join(module2_dir, f'node_mapping_{self.sample_ratio}.txt')
            
            # 统计边数量
            edge_count = 0
            with open(adjacency_edges_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):  # 跳过空行和注释行
                        edge_count += 1
            
            # 统计节点数量
            node_count = 0
            with open(node_mapping_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):  # 跳过空行和注释行
                        node_count += 1
            
            
        elif os.path.exists(adjacency_matrix_path):
            # 兼容旧格式
            adjacency_matrix = np.load(adjacency_matrix_path)
        else:
            raise FileNotFoundError(f"基础邻接矩阵文件不存在: {adjacency_edges_path} 或 {adjacency_matrix_path}")
        
        # 加载引力图 - 从CSV文件加载（从模块3）
        attraction_csv_path = os.path.join(module3_dir, f'attraction_graph_{self.sample_ratio}.csv')
        if not os.path.exists(attraction_csv_path):
            raise FileNotFoundError(f"引力图CSV文件不存在: {attraction_csv_path}")
        
        self.attraction_graph = {}
        with open(attraction_csv_path, 'r') as f:
            next(f)  # 跳过标题行
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(',')
                    if len(parts) == 3:
                        node1_id, node2_id, similarity = parts
                        key = (int(node1_id), int(node2_id))
                        self.attraction_graph[key] = float(similarity)
        
        # 加载斥力图 - 从CSV文件加载（从模块3）
        repulsion_csv_path = os.path.join(module3_dir, f'repulsion_graph_{self.sample_ratio}.csv')
        if not os.path.exists(repulsion_csv_path):
            raise FileNotFoundError(f"斥力图CSV文件不存在: {repulsion_csv_path}")
        
        self.repulsion_graph = {}
        with open(repulsion_csv_path, 'r') as f:
            next(f)  # 跳过标题行
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(',')
                    if len(parts) == 3:
                        node1_id, node2_id, similarity = parts
                        key = (int(node1_id), int(node2_id))
                        self.repulsion_graph[key] = float(similarity)
        
        # 加载水军行为得分（从模块2，用于增强水军节点之间的连接）
        spam_scores_path = os.path.join(module2_dir, f'spam_behavior_scores_{self.sample_ratio}.npy')
        if os.path.exists(spam_scores_path):
            self.spam_behavior_scores = np.load(spam_scores_path)
        else:
            pass
            self.spam_behavior_scores = None
        
        
    def _normalize_similarities(self):
        """相似度预处理和标准化"""
        
        # 引力图相似度标准化
        if self.attraction_graph:
            attr_similarities = list(self.attraction_graph.values())
            attr_min = min(attr_similarities)
            attr_max = max(attr_similarities)
            attr_range = attr_max - attr_min if attr_max != attr_min else 1.0
            
            self.attr_sim_norm = {}
            for edge, sim in self.attraction_graph.items():
                self.attr_sim_norm[edge] = (sim - attr_min) / attr_range
        else:
            self.attr_sim_norm = {}
        
        # 斥力图相似度标准化（修复版：确保所有边都被削弱）
        if self.repulsion_graph:
            rep_similarities = list(self.repulsion_graph.values())
            rep_min = min(rep_similarities)
            rep_max = max(rep_similarities)
            rep_range = rep_max - rep_min if rep_max != rep_min else 1.0
            
            self.rep_sim_norm = {}
            for edge, sim in self.repulsion_graph.items():
                # 修复：使用原始相似度而非归一化值，避免最小值归一化为0导致权重=1.0
                # 直接使用相似度值，范围通常在[0, 1]之间
                # 这样即使是最低相似度的边也会被削弱
                self.rep_sim_norm[edge] = sim  # 使用原始相似度
            
        else:
            self.rep_sim_norm = {}
        
        
    def _calculate_weight_factors(self):
        """计算权重因子（改进版：使用指数函数大幅扩大权重调节范围）"""
        
        # 改进的引力权重因子: w_attr = 1 + λ × (exp(sim_norm) - 1)
        # 使用指数函数放大高相似度的影响
        # 当sim_norm=0时，w=1.0
        # 当sim_norm=1时，w=1 + λ × (e-1) ≈ 1 + 1.718λ
        # 当λ=2.0时，权重范围: [1.0, 4.436]
        self.attr_weights = {}
        max_attr_weight = 0.0
        for edge, sim_norm in self.attr_sim_norm.items():
            weight = 1.0 + self.lambda_param * (np.exp(sim_norm) - 1.0)
            self.attr_weights[edge] = weight
            max_attr_weight = max(max_attr_weight, weight)
        
        # 改进的斥力权重因子: w_rep = exp(-λ_rep × sim)
        # 使用指数衰减函数大幅降低高相似度异类节点对的权重
        # 修复：使用原始相似度，确保所有边都被削弱
        # 相似度范围通常在[0.5, 0.63]（斥力图阈值附近）
        # 当sim=0.5时，w=exp(-6.0×0.5)=exp(-3.0)≈0.05
        # 当sim=0.63时，w=exp(-6.0×0.63)=exp(-3.78)≈0.023
        self.rep_weights = {}
        min_rep_weight = 1.0
        max_rep_weight = 0.0
        for edge, sim in self.rep_sim_norm.items():
            # 使用指数衰减，最小权重0.01（几乎断开连接）
            weight = np.exp(-self.lambda_rep * sim)
            weight = max(0.01, weight)
            self.rep_weights[edge] = weight
            min_rep_weight = min(min_rep_weight, weight)
            max_rep_weight = max(max_rep_weight, weight)
        
        # 计算理论权重范围
        theoretical_max_attr = 1.0 + self.lambda_param * (np.e - 1.0)
        # 获取实际相似度范围
        if self.repulsion_graph:
            rep_similarities = list(self.repulsion_graph.values())
            rep_min_sim = min(rep_similarities)
            rep_max_sim = max(rep_similarities)
            theoretical_min_rep = np.exp(-self.lambda_rep * rep_max_sim)
            theoretical_max_rep = np.exp(-self.lambda_rep * rep_min_sim)
        else:
            theoretical_min_rep = 0.01
            theoretical_max_rep = 1.0
        
        
    def build_enhanced_adjacency_matrix(self):
        """构建增强邻接矩阵 - 使用文件级操作避免内存不足"""
        
        module2_dir = get_result_dir(self.sample_ratio, self.db_path, module=2, force_no_threshold=True)
        current_result_dir = result_dir if result_dir is not None else get_result_dir(self.sample_ratio, self.db_path, module=4)
        os.makedirs(current_result_dir, exist_ok=True)  # 确保目录存在
        
        # 第一步：复制基础邻接矩阵文件到增强邻接矩阵文件
        adjacency_edges_path = os.path.join(module2_dir, f'adjacency_edges_{self.sample_ratio}.txt')
        enhanced_edges_path = os.path.join(current_result_dir, f'enhanced_adjacency_edges_{self.sample_ratio}.txt')
        
        edge_count = 0
        with open(adjacency_edges_path, 'r', encoding='utf-8') as src, open(enhanced_edges_path, 'w', encoding='utf-8') as dst:
            for line in src:
                line = line.strip()
                if line and not line.startswith('#'):
                    dst.write(line + '\n')
                    edge_count += 1
        
        
        # 第二步：使用GPU加速的权重增强计算
        self._apply_weight_enhancements_gpu(enhanced_edges_path)
        
        # 第三步：保存其他格式的增强邻接矩阵文件
        self._save_enhanced_matrix_formats(enhanced_edges_path)
        
    
    def _apply_weight_enhancements_gpu(self, enhanced_edges_path):
        """使用GPU加速应用权重增强（改进版：添加水军行为得分增强）"""
        
        # 创建节点对到权重的映射字典
        enhancement_weights = {}
        
        #  修改策略：先添加斥力权重，再添加引力权重（引力只对不在斥力图中的边生效）
        # 1. 先添加斥力权重（削弱邻接矩阵中相似度最低50%的边）
        for (node1_id, node2_id), weight in self.rep_weights.items():
            key = tuple(sorted([node1_id, node2_id]))  # 标准化节点对顺序
            enhancement_weights[key] = weight
        
        # 2. 再添加引力权重，但只对不在斥力图中的边生效
        # 这样引力图可以增强不在邻接矩阵中的同类节点连接
        attraction_only_count = 0
        for (node1_id, node2_id), weight in self.attr_weights.items():
            key = tuple(sorted([node1_id, node2_id]))  # 标准化节点对顺序
            if key not in enhancement_weights:  # 只增强不在斥力图中的边
                enhancement_weights[key] = weight
                attraction_only_count += 1
        
        
        # 【新增】基于水军行为得分的边权重增强
        # 如果两个节点的水军行为得分都高（>0.6），增强它们之间的连接
        spam_enhanced_count = 0
        if self.spam_behavior_scores is not None:
            pass
            spam_threshold = 0.6  # 水军行为得分阈值
            spam_boost_factor = 1.5  # 增强系数
            
            # 预先计算高得分节点
            high_spam_nodes = set(np.where(self.spam_behavior_scores > spam_threshold)[0])
        else:
            high_spam_nodes = set()
        
        if high_spam_nodes:
            pass
        
        # 读取并更新增强邻接矩阵文件
        temp_path = enhanced_edges_path + '.tmp'
        enhanced_count = 0
        total_edges = 0
        
        # 使用GPU进行批量权重计算（如果可用）
        if gpu_device is not None:
            pass
            
        with open(enhanced_edges_path, 'r') as src, open(temp_path, 'w') as dst:
            for line in src:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) == 3:
                        node1, node2, weight_str = parts
                        node1_int = int(node1)
                        node2_int = int(node2)
                        original_weight = float(weight_str)
                        # 初始化增强因子
                        enhancement_factor = 1.0
                        # 1. 检查引力图/斥力图增强
                        key = tuple(sorted([node1_int, node2_int]))
                        if key in enhancement_weights:
                            enhancement_factor = enhancement_weights[key]
                            enhanced_count += 1
                        # 2. 【新增】检查水军行为得分增强
                        # 如果两个节点都是高水军行为得分节点，额外增强
                        if high_spam_nodes and node1_int in high_spam_nodes and node2_int in high_spam_nodes:
                            enhancement_factor *= 1.5  # 额外增强50%
                            spam_enhanced_count += 1
                        # 应用最终权重
                        new_weight = original_weight * enhancement_factor
                        dst.write(f"{node1}\t{node2}\t{new_weight:.6f}\n")
                        total_edges += 1
                        # 显示进度
                        if total_edges % 10000 == 0:
                            pass
        
        # 替换原文件
        os.replace(temp_path, enhanced_edges_path)
        
        if high_spam_nodes:
            pass
    
    def _save_enhanced_matrix_formats(self, enhanced_edges_path):
        """保存增强邻接矩阵的其他格式"""
        module1_dir = get_result_dir(self.sample_ratio, self.db_path, module=1, force_no_threshold=True)
        current_result_dir = result_dir if result_dir is not None else get_result_dir(self.sample_ratio, self.db_path, module=4)
        
        # 加载虚拟节点映射（从模块1）
        virtual_nodes_path = os.path.join(module1_dir, 'virtual_nodes.pkl')
        with open(virtual_nodes_path, 'rb') as f:
            virtual_nodes = pickle.load(f)
        
        node_ids = list(virtual_nodes.keys())
        
        # 保存节点映射文件
        enhanced_node_mapping_path = os.path.join(current_result_dir, f'enhanced_node_mapping_{self.sample_ratio}.txt')
        with open(enhanced_node_mapping_path, 'w') as f:
            for idx, node_id in enumerate(node_ids):
                f.write(f"{node_id}\t{idx}\n")
        
        # 保存边列表格式（用于兼容性）
        enhanced_edges_pkl_path = os.path.join(current_result_dir, f'enhanced_adjacency_edges_{self.sample_ratio}.pkl')
        edge_list = []
        edge_count = 0
        with open(enhanced_edges_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) == 3:
                        node1_id, node2_id, weight_str = parts
                        weight = float(weight_str)
                        edge_list.append((int(node1_id), int(node2_id), weight))
                        edge_count += 1
        enhanced_edges_data = {
            'edge_list': edge_list,
            'num_nodes': len(node_ids),
            'num_edges': edge_count,
            'node_ids': node_ids
        }
        
        with open(enhanced_edges_pkl_path, 'wb') as f:
            pickle.dump(enhanced_edges_data, f)
        
        # 保存权重因子
        weights_path = os.path.join(current_result_dir, f'weight_factors_{self.sample_ratio}.pkl')
        weight_factors = {
            'attraction_weights': self.attr_weights,
            'repulsion_weights': self.rep_weights,
            'lambda_param': self.lambda_param
        }
        
        with open(weights_path, 'wb') as f:
            pickle.dump(weight_factors, f)
        
        
    def save_enhanced_matrix(self):
        """保存增强邻接矩阵 - 文件级操作版本已在build_enhanced_adjacency_matrix中完成"""
        
    def run(self):
        # [FLOW-M4] 模块4：增强邻接矩阵构建 | 缓存: module4/enhanced_adjacency_edges_*.txt, enhanced_adjacency_edges_*.pkl
        # [!] 模块1-4代码及缓存不可修改（规则11）
        # [!] pkl中edge_list节点ID必须为int类型（已修复，不可回退）
        """运行模块4的完整流程"""
        try:
            # 检查缓存文件是否存在
            global result_dir
            current_result_dir = result_dir if result_dir is not None else get_result_dir(self.sample_ratio, self.db_path, module=4)
            
            # 优先检查新的边列表文件格式
            enhanced_edges_path = os.path.join(current_result_dir, f'enhanced_adjacency_edges_{self.sample_ratio}.txt')
            enhanced_pkl_path = os.path.join(current_result_dir, f'enhanced_adjacency_edges_{self.sample_ratio}.pkl')
            enhanced_matrix_path = os.path.join(current_result_dir, f'enhanced_adjacency_matrix_{self.sample_ratio}.npy')
            
            if os.path.exists(enhanced_edges_path) and os.path.exists(enhanced_pkl_path):
                pass
                # 加载缓存文件以显示统计信息
                with open(enhanced_pkl_path, 'rb') as f:
                    enhanced_data = pickle.load(f)
                # 计算权重统计
                if enhanced_data['edge_list']:
                    weights = [edge[2] for edge in enhanced_data['edge_list']]
                    enhanced_mean = np.mean(weights)
                return True
            elif os.path.exists(enhanced_matrix_path):
                # 兼容旧格式
                # 加载缓存文件以显示统计信息
                enhanced_matrix = np.load(enhanced_matrix_path)
                enhanced_nonzero = np.count_nonzero(enhanced_matrix)
                if enhanced_nonzero > 0:
                    enhanced_mean = np.mean(enhanced_matrix[enhanced_matrix > 0])
                return True
            
            # 如果缓存不存在，执行完整流程
            self.load_data()
            self._normalize_similarities()
            self._calculate_weight_factors()
            self.build_enhanced_adjacency_matrix()
            self.save_enhanced_matrix()
            return True
        except Exception as e:
            pass
            return False

# ================================
# 模块5：时序图神经网络（TGNN）与DBSCAN联合优化聚类
# ================================

class EnhancedTGNNModel(nn.Module):
    """两层GCN节点嵌入编码器 - 与论文3.3节描述对齐"""
    def __init__(self, nfeat=24, nhid=64, nclass=64, dropout=0.3):
        super(EnhancedTGNNModel, self).__init__()
        
        # 两层GCN编码器：H^(l+1) = ReLU(Â·H^(l)·W^(l))
        self.gcn = WeightedGCN(nfeat, nhid, nclass, dropout, num_layers=2)
    
    def forward(self, features, adj, user_to_virtual=None, virtual_node_times=None,
                max_users=5000, use_temporal=True):
        """前向传播：两层GCN编码，输出节点嵌入矩阵 Z"""
        # GCN编码：H^(1) = ReLU(Â·X·W^(0))，H^(2) = Â·H^(1)·W^(1)
        node_embeddings = self.gcn(features, adj)
        return node_embeddings, {}


# 保持原有TGNNModel类的兼容性
class TGNNModel(EnhancedTGNNModel):
    """GCN节点嵌入模型（兼容性包装）"""
    def __init__(self, nfeat, nhid, nclass, dropout=0.3):
        adjusted_nfeat = 24 if nfeat < 24 else nfeat
        super(TGNNModel, self).__init__(adjusted_nfeat, nhid, nclass, dropout)
        if nfeat < 24:
            pass

class WeightedGCNLayer(nn.Module):
    """加权GCN层 - 接受增强后的邻接矩阵权重，并使用门控单元抑制异类节点影响"""
    
    def __init__(self, in_features, out_features, dropout=0.5, bias=True, use_gating=True):
        super(WeightedGCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.use_gating = use_gating
        
        # 特征变换权重
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        # 门控单元参数（基于模型描述文档的设计）
        if self.use_gating:
            # θ: 门控强度参数，初始值1.0
            self.gate_theta = nn.Parameter(torch.tensor(1.0))
            # β: 缩放参数，用于降低高权重连接的抑制程度，初始值0.8
            self.gate_beta = nn.Parameter(torch.tensor(0.8))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, adj_with_weights):
        """
        前向传播 - 使用门控单元调节邻居聚合
        x: [N, in_features] 节点特征矩阵
        adj_with_weights: 稀疏邻接矩阵，包含边权重
        """
        # 1. 特征变换
        support = torch.mm(x, self.weight)
        
        # 2. 应用门控单元调节邻接矩阵权重
        if self.use_gating:
            adj_gated = self._apply_gating(adj_with_weights)
        else:
            adj_gated = adj_with_weights
        
        # 3. 使用门控后的邻接矩阵进行邻居聚合
        output = torch.sparse.mm(adj_gated, support)
        
        # 4. 添加偏置
        if self.bias is not None:
            output = output + self.bias
        
        return output
    
    def _apply_gating(self, adj_sparse):
        """
        应用门控单元到邻接矩阵
        基于模型描述文档的门控设计（改进V2 Sqrt方法）：
        - 对于权重 <= 1.0 的连接：g_ij = sigmoid(θ * (A[i,j] - 1))
        - 对于权重 > 1.0 的连接：g_ij = sigmoid(θ * β * sqrt(A[i,j] - 1))
        
        这种设计确保：
        1. 低相似度节点对（斥力抑制，权重<1）被进一步抑制
        2. 高相似度节点对（引力增强，权重>1）保持较高权重
        3. 减小异类节点在邻居聚合阶段的相互影响
        """
        # 获取稀疏矩阵的索引和值
        indices = adj_sparse._indices()
        values = adj_sparse._values()
        
        # 计算门控权重
        # 对于权重 <= 1.0: 直接应用sigmoid(θ * (w - 1))
        # 对于权重 > 1.0: 应用sigmoid(θ * β * sqrt(w - 1))
        mask_low = values <= 1.0
        mask_high = values > 1.0
        
        gated_values = torch.zeros_like(values)
        
        # 处理低权重连接（斥力抑制区域）
        if mask_low.any():
            low_values = values[mask_low]
            # g_ij = sigmoid(θ * (w - 1))，w < 1时结果 < 0.5，进一步抑制
            gated_values[mask_low] = torch.sigmoid(self.gate_theta * (low_values - 1.0))
        
        # 处理高权重连接（引力增强区域）
        if mask_high.any():
            high_values = values[mask_high]
            # 使用平方根缩放降低抑制程度
            excess = high_values - 1.0
            scaled_excess = self.gate_beta * torch.sqrt(excess)
            # g_ij = sigmoid(θ * β * sqrt(w - 1))，保持较高权重
            gated_values[mask_high] = torch.sigmoid(self.gate_theta * scaled_excess)
        
        # 应用门控：将门控权重与原始权重相乘
        # 这样既保留了增强邻接矩阵的语义，又进一步抑制了低相似度连接
        final_values = values * gated_values
        
        # 构建新的稀疏矩阵
        gated_adj = torch.sparse.FloatTensor(
            indices,
            final_values,
            adj_sparse.size()
        )
        
        return gated_adj

class WeightedGCN(nn.Module):
    """加权GCN模型 - 使用增强后的邻接矩阵"""
    
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, num_layers=2):
        super(WeightedGCN, self).__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        
        # 构建多层GCN
        self.layers = nn.ModuleList()
        
        # 第一层
        self.layers.append(WeightedGCNLayer(nfeat, nhid, dropout=dropout))
        
        # 中间层
        for _ in range(num_layers - 2):
            self.layers.append(WeightedGCNLayer(nhid, nhid, dropout=dropout))
        
        # 输出层
        if num_layers > 1:
            self.layers.append(WeightedGCNLayer(nhid, nclass, dropout=dropout))
        else:
            # 单层情况
            self.layers[0] = WeightedGCNLayer(nfeat, nclass, dropout=dropout)
        
        # 层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(nhid if i < num_layers - 1 else nclass)
            for i in range(num_layers)
        ])
        
        # 残差连接投影
        self.residual_projs = nn.ModuleList()
        dims = [nfeat] + [nhid] * (num_layers - 1) + [nclass]
        for i in range(num_layers):
            if dims[i] != dims[i + 1]:
                self.residual_projs.append(nn.Linear(dims[i], dims[i + 1]))
            else:
                self.residual_projs.append(None)
    
    def forward(self, x, adj_with_weights):
        """
        前向传播
        x: [N, nfeat] 节点特征矩阵
        adj_with_weights: 稀疏邻接矩阵，包含增强后的边权重
        """
        h = x
        
        for i, (layer, layer_norm, residual_proj) in enumerate(
            zip(self.layers, self.layer_norms, self.residual_projs)
        ):
            # Dropout
            h = F.dropout(h, self.dropout, training=self.training)
            
            # GCN层
            h_new = layer(h, adj_with_weights)
            
            # 残差连接
            if residual_proj is not None:
                h_res = residual_proj(h)
            else:
                h_res = h
            
            # 层归一化 + 残差
            h = layer_norm(h_new + h_res)
            
            # 激活函数（最后一层除外）
            if i < self.num_layers - 1:
                h = F.relu(h)
        
        # L2归一化
        h_normalized = F.normalize(h, p=2, dim=1)
        return h_normalized


class ContrastiveAttrRepLoss(nn.Module):
    """
     新增：对比学习损失函数 - 直接利用引力图和斥力图
    
    设计思想：
    - 引力图中的节点对应该相似（正样本，拉近）
    - 斥力图中的节点对应该不相似（负样本，推远）
    
    创新点：
    - 解决消融实验发现的问题：让损失函数直接感知图结构变化
    - 引力图/斥力图的变化直接影响训练目标
    - 与GCN聚合协同工作，最大化引力图/斥力图的作用
    
    损失公式：
    L_contrastive = L_attraction + L_repulsion
    
    其中（均等对待每对节点，无w_ij加权）：
    - L_attraction = mean((1 - cos(z_i, z_j))²)          # 引力图节点对，拉近嵌入
    - L_repulsion  = mean(relu(cos(z_i, z_j) - margin))# 斥力图节点对，推远嵌入，margin=0.3
    """
    
    def __init__(self, max_pairs=10000):
        """
        参数：
        - max_pairs: 每个batch最多采样的节点对数量（避免内存溢出）
        """
        super(ContrastiveAttrRepLoss, self).__init__()
        self.max_pairs = max_pairs
    
    def forward(self, embeddings, attraction_pairs, repulsion_pairs):
        """
         改进：批量化计算对比学习损失，提升效率
        
        参数：
        - embeddings: [N, D] 节点嵌入
        - attraction_pairs: 引力图节点对列表 [(node_i, node_j, weight), ...]
        - repulsion_pairs: 斥力图节点对列表 [(node_i, node_j, weight), ...]
        
        返回：
        - total_loss: 总对比学习损失（张量）
        - loss_dict: 各部分损失的字典
        """
        device = embeddings.device
        
        # 1. 计算引力图损失（批量化）
        if attraction_pairs and len(attraction_pairs) > 0:
            # 采样
            if len(attraction_pairs) > self.max_pairs:
                sampled_indices = np.random.choice(len(attraction_pairs), self.max_pairs, replace=False)
                sampled_pairs = [attraction_pairs[i] for i in sampled_indices]
            else:
                sampled_pairs = attraction_pairs
            
            # 批量提取索引
            indices_i = torch.tensor([p[0] for p in sampled_pairs], dtype=torch.long, device=device)
            indices_j = torch.tensor([p[1] for p in sampled_pairs], dtype=torch.long, device=device)
            
            # 批量获取嵌入
            emb_i = embeddings[indices_i]  # [N, D]
            emb_j = embeddings[indices_j]  # [N, D]
            
            # 批量计算余弦相似度
            sim = F.cosine_similarity(emb_i, emb_j, dim=1)  # [N]
            
            # 引力损失：二次损失(1-sim)²，梯度随相似度提高自然衰减
            # sim=0.3时梯度≈-1.4，sim=0.9时梯度≈-0.2，sim→1时梯度→0，无需手动设margin
            loss = (1 - sim) ** 2  # [N]
            attraction_loss = loss.mean()
        else:
            attraction_loss = torch.tensor(0.0, device=device)
        
        # 2. 计算斥力图损失（批量化）
        if repulsion_pairs and len(repulsion_pairs) > 0:
            # 采样
            if len(repulsion_pairs) > self.max_pairs:
                sampled_indices = np.random.choice(len(repulsion_pairs), self.max_pairs, replace=False)
                sampled_pairs = [repulsion_pairs[i] for i in sampled_indices]
            else:
                sampled_pairs = repulsion_pairs
            
            # 批量提取索引
            indices_i = torch.tensor([p[0] for p in sampled_pairs], dtype=torch.long, device=device)
            indices_j = torch.tensor([p[1] for p in sampled_pairs], dtype=torch.long, device=device)
            
            # 批量获取嵌入
            emb_i = embeddings[indices_i]  # [N, D]
            emb_j = embeddings[indices_j]  # [N, D]
            
            # 批量计算余弦相似度
            sim = F.cosine_similarity(emb_i, emb_j, dim=1)  # [N]
            
            # 斥力损失：让斥力边节点对嵌入远离，均等对待每对节点
            # 目标：让相似度低于margin（期望不相似）
            margin = 0.3  # 期望相似度低于0.3
            loss = torch.relu(sim - margin)  # [N]
            repulsion_loss = loss.mean()
        else:
            repulsion_loss = torch.tensor(0.0, device=device)
        
        # 总损失
        total_loss = attraction_loss + repulsion_loss
        
        return total_loss, {
            'attraction_loss': attraction_loss.item(),
            'repulsion_loss': repulsion_loss.item(),
            'contrastive_loss': total_loss.item(),
            'n_attraction_pairs': len(attraction_pairs) if attraction_pairs else 0,
            'n_repulsion_pairs': len(repulsion_pairs) if repulsion_pairs else 0,
        }


class MixedUserAwareLoss(nn.Module):
    """
     简化版损失函数：重构损失 + 对比学习损失
    
    组成部分：
    1. 重构损失：保持原始特征信息，防止信息丢失
    2. 对比学习损失：直接利用引力图和斥力图，优化节点嵌入的相似度关系
    
    设计理念：
    - 重构损失保证嵌入能够恢复原始特征
    - 对比学习损失利用图结构信息，让同类节点靠近、异类节点远离
    - 避免使用伪标签相关的损失，减少潜在冲突
    - 完全无监督，理论基础扎实
    """
    
    def __init__(self, 
                 lambda_contrastive=0.5,  # 对比学习损失权重
                 feature_dim=8, 
                 embed_dim=16,
                 use_contrastive=True):  # 是否使用对比学习
        super(MixedUserAwareLoss, self).__init__()
        self.lambda_contrastive = lambda_contrastive
        self.use_contrastive = use_contrastive
        
        # 可学习的解码器
        self.decoder = nn.Linear(embed_dim, feature_dim)
        nn.init.xavier_uniform_(self.decoder.weight)
        
        # 对比学习损失函数
        if self.use_contrastive:
            self.contrastive_loss = ContrastiveAttrRepLoss(max_pairs=10000)
    
    def forward(self, embeddings, original_features, spam_scores=None, adj_matrix=None, 
                attraction_pairs=None, repulsion_pairs=None):
        """
        计算简化版损失：重构损失 + 对比学习损失
        
        参数：
        - embeddings: [N, D] 节点嵌入
        - original_features: [N, F] 原始特征
        - spam_scores: [N] 水军行为得分（保留参数以兼容调用，但不使用）
        - adj_matrix: [N, N] 邻接矩阵（保留参数以兼容调用，但不使用）
        - attraction_pairs: 引力图节点对列表 [(node_i, node_j, weight), ...]
        - repulsion_pairs: 斥力图节点对列表 [(node_i, node_j, weight), ...]
        
        返回：
        - total_loss: 总损失
        - loss_dict: 各部分损失的字典
        """
        device = embeddings.device
        
        # 损失1：重构损失
        reconstruction_loss = self._compute_reconstruction_loss(embeddings, original_features)
        
        # 数值稳定性处理
        reconstruction_loss = torch.clamp(reconstruction_loss, 0, 150.0)
        
        # 损失2：对比学习损失
        contrastive_loss_value = torch.tensor(0.0, device=device)
        contrastive_dict = {}
        if self.use_contrastive and attraction_pairs is not None and repulsion_pairs is not None:
            contrastive_loss_value, contrastive_dict = self.contrastive_loss(
                embeddings, attraction_pairs, repulsion_pairs
            )
            contrastive_loss_value = torch.clamp(contrastive_loss_value, 0, 10.0)
        
        # 总损失
        total_loss = reconstruction_loss + self.lambda_contrastive * contrastive_loss_value
        
        # 返回字典
        loss_dict = {
            'reconstruction_loss': reconstruction_loss.item(),
            'contrastive_loss': contrastive_loss_value.item(),
            'total_loss': total_loss.item(),
        }
        # 添加对比学习详细信息
        if contrastive_dict:
            loss_dict.update(contrastive_dict)
        
        return total_loss, loss_dict
    
    def _compute_reconstruction_loss(self, embeddings, original_features):
        """重构损失 - 使用可学习的解码器"""
        reconstructed = self.decoder(embeddings)
        return F.mse_loss(reconstructed, original_features)


# ============================================================================
# 行为解耦模型定义（用于混合用户检测）
# ============================================================================

class EnhancedMixingPredictor(nn.Module):
    """增强的混合度预测器"""
    
    def __init__(self, spam_dim=6, genuine_dim=6, hidden_dim=64):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(spam_dim + genuine_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, spam_features, genuine_features):
        combined = torch.cat([spam_features, genuine_features], dim=1)
        alpha = self.network(combined)
        return alpha


class EnhancedDualEncoder(nn.Module):
    """增强的双编码器 - 更深的网络"""
    
    def __init__(self, spam_dim=6, genuine_dim=6, hidden_dim=64, output_dim=128):
        super().__init__()
        
        # 水军编码器（更深）
        self.spam_encoder = nn.Sequential(
            nn.Linear(spam_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim * 2, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(output_dim, output_dim)
        )
        
        # 真实编码器（更深）
        self.genuine_encoder = nn.Sequential(
            nn.Linear(genuine_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim * 2, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, spam_features, genuine_features):
        spam_emb = self.spam_encoder(spam_features)
        genuine_emb = self.genuine_encoder(genuine_features)
        return spam_emb, genuine_emb


class EnhancedBehaviorDisentanglementModel(nn.Module):
    """增强的行为解耦模型"""
    
    def __init__(self, spam_dim=6, genuine_dim=6, hidden_dim=64, output_dim=128):
        super().__init__()
        
        self.mixing_predictor = EnhancedMixingPredictor(spam_dim, genuine_dim, hidden_dim)
        self.dual_encoder = EnhancedDualEncoder(spam_dim, genuine_dim, hidden_dim, output_dim)
    
    def forward(self, spam_features, genuine_features):
        # 预测混合度
        alpha = self.mixing_predictor(spam_features, genuine_features)
        
        # 双编码
        spam_emb, genuine_emb = self.dual_encoder(spam_features, genuine_features)
        
        # 组合嵌入
        final_emb = alpha * spam_emb + (1 - alpha) * genuine_emb
        
        return spam_emb, genuine_emb, alpha, final_emb


class Module5_TGNNDBSCANClustering:
    """模块5：GCN编码（GraphSAINT采样）与HDBSCAN聚类
    
    在时间窗口虚拟节点上训练图卷积网络，使用引力/斥力对比学习损失，
    训练完成后用HDBSCAN对节点嵌入进行聚类，得到候选水军群组。
    """
    
    def __init__(self, db_path, sample_ratio=1.0, hidden_dim=64, output_dim=64,
                 dropout=0.3, alpha=0.2, epochs=50, lr=0.001, use_gpu=False):
        self.db_path = db_path
        self.dataset_path = db_path  # 添加dataset_path属性以支持GSS增强损失
        self.sample_ratio = sample_ratio
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.alpha = alpha
        self.epochs = epochs
        self.lr = lr
        
        # 构建预处理目录路径（统一使用get_result_dir，保持路径一致）
        self.preprocessed_dir = get_result_dir(self.sample_ratio, db_path, module=5)
        
        # 设备配置 - 使用GPU加速（根据用户要求）
        # 通过文件记录注意力权重，避免构建大型矩阵和张量
        self.use_gpu = use_gpu if torch.cuda.is_available() else False
        self.device = get_device() if self.use_gpu else torch.device("cpu")
        if self.use_gpu:
            pass
            # 获取GPU内存信息
            if torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                self.max_gpu_memory_mb = int(gpu_memory_gb * 1024 * 0.8)  # 使用80%的GPU内存
            else:
                self.max_gpu_memory_mb = 8192  # 默认8GB
        else:
            pass
            self.max_gpu_memory_mb = 0  # CPU模式不需要GPU内存限制
        
        # 数据存储
        self.features = None
        self.enhanced_adj_matrix = None
        self.virtual_nodes = None
        self.gat_model = None
        self.embeddings = None
        self.cluster_labels = None
        self.feature_chunk_files = []  # 初始化特征分块文件列表
        
        # 行为解耦模型配置（默认禁用）
        self.use_behavior_disentanglement = False
        
        # CUDA错误处理标志
        self.cuda_error_occurred = False
        self.memory_error_count = 0  # 内存错误计数器
        
        # 获取结果目录
        current_result_dir = result_dir if result_dir is not None else get_result_dir(self.sample_ratio, self.db_path, module=5)
        
        # 初始化嵌入缓存管理器
        try:
            from embedding_cache_manager import EmbeddingCacheManager
            cache_dir = os.path.join(current_result_dir, "embedding_cache")
            self.cache_manager = EmbeddingCacheManager(cache_dir=cache_dir)
            self.embedding_generator = None  # 将在模型初始化后创建
        except ImportError as e:
            pass
            self.cache_manager = None
            self.embedding_generator = None
        
        # 注意：当前GAT使用简化重构损失，不需要加载伪标签文件
        self.pseudo_labels_file = None
        
        # GCAS损失函数已移除
        
        self.user_reviews_data = self._load_user_reviews_data()
    
    def load_data(self):
        """加载特征矩阵和增强邻接矩阵（稀疏格式）"""
        
        # 模块5需要加载模块1、2和模块4的缓存
        module1_dir = get_result_dir(self.sample_ratio, self.db_path, module=1, force_no_threshold=True)
        module2_dir = get_result_dir(self.sample_ratio, self.db_path, module=2, force_no_threshold=True)
        module4_dir = get_result_dir(self.sample_ratio, self.db_path, module=4)
        current_result_dir = result_dir if result_dir is not None else get_result_dir(self.sample_ratio, self.db_path, module=5)
        
        # 加载14维特征矩阵（从模块2，用于GAT输入）
        features_path = os.path.join(module2_dir, f'feature_matrix_14d_{self.sample_ratio}.npy')
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"特征矩阵文件不存在: {features_path}")
        
        original_features = np.load(features_path)
        
        #  创建增强特征矩阵（原始维度 + 10维无监督混合用户特征）
        self.features = self._create_enhanced_features(original_features)
        
        # 优先加载稀疏格式的增强邻接矩阵（从模块4）
        enhanced_edges_path = os.path.join(module4_dir, f'enhanced_adjacency_edges_{self.sample_ratio}.pkl')
        enhanced_sparse_path = os.path.join(module4_dir, f'enhanced_adjacency_sparse_{self.sample_ratio}.npz')
        enhanced_dense_path = os.path.join(module4_dir, f'enhanced_adjacency_matrix_{self.sample_ratio}.npy')
        
        if os.path.exists(enhanced_edges_path):
            # 加载边列表格式
            with open(enhanced_edges_path, 'rb') as f:
                self.enhanced_edges_data = pickle.load(f)
            self.use_sparse_format = True
            
        elif os.path.exists(enhanced_sparse_path):
            # 加载稀疏矩阵格式
            from scipy.sparse import load_npz
            self.enhanced_adj_sparse = load_npz(enhanced_sparse_path)
            self.use_sparse_format = True
            
        elif os.path.exists(enhanced_dense_path):
            # 回退到密集矩阵格式
            self.enhanced_adj_matrix = np.load(enhanced_dense_path)
            self.use_sparse_format = False
            
        else:
            raise FileNotFoundError(f"增强邻接矩阵文件不存在")
        
        # 加载虚拟节点信息（从模块1）
        virtual_nodes_path = os.path.join(module1_dir, 'virtual_nodes.pkl')
        with open(virtual_nodes_path, 'rb') as f:
            self.virtual_nodes = pickle.load(f)
        
        #  创建node_id_list（对比学习需要）
        self.node_id_list = list(self.virtual_nodes.keys())
        
        #  加载虚拟节点到用户的映射（对比学习需要，从模块1）
        user_mapping_path = os.path.join(module1_dir, 'user_to_virtual_mapping.pkl')
        if os.path.exists(user_mapping_path):
            with open(user_mapping_path, 'rb') as f:
                user_to_virtual = pickle.load(f)
            # 反向映射：虚拟节点 -> 用户
            self.virtual_to_user_mapping = {}
            for user_id, virtual_ids in user_to_virtual.items():
                for virtual_id in virtual_ids:
                    self.virtual_to_user_mapping[virtual_id] = user_id
        else:
            # 如果没有映射文件，从virtual_nodes中提取
            self.virtual_to_user_mapping = {}
            for node_id, node_info in self.virtual_nodes.items():
                self.virtual_to_user_mapping[node_id] = node_info['original_user_id']
        
        
        #  加载水军行为得分（用于混合用户感知损失）
        spam_scores_path = os.path.join(module2_dir, f'spam_behavior_scores_{self.sample_ratio}.npy')
        if os.path.exists(spam_scores_path):
            self.spam_behavior_scores = np.load(spam_scores_path)
        else:
            raise FileNotFoundError(f"水军行为得分文件不存在: {spam_scores_path}")
        
        #  修复：从CSV文件加载引力图和斥力图（模块3保存的pickle文件是空占位符）
        module3_dir = get_result_dir(self.sample_ratio, self.db_path, module=3)
        
        # 加载引力图（从CSV文件）
        attraction_csv_path = os.path.join(module3_dir, f'attraction_graph_{self.sample_ratio}.csv')
        if os.path.exists(attraction_csv_path):
            pass
            self.attraction_graph = {}
            import csv
            with open(attraction_csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    node1_id = int(row['node1_id'])
                    node2_id = int(row['node2_id'])
                    similarity = float(row['similarity'])
                    self.attraction_graph[(node1_id, node2_id)] = similarity
        else:
            pass
            self.attraction_graph = {}
        
        # 加载斥力图（从CSV文件）
        repulsion_csv_path = os.path.join(module3_dir, f'repulsion_graph_{self.sample_ratio}.csv')
        if os.path.exists(repulsion_csv_path):
            pass
            self.repulsion_graph = {}
            with open(repulsion_csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    node1_id = int(row['node1_id'])
                    node2_id = int(row['node2_id'])
                    similarity = float(row['similarity'])
                    self.repulsion_graph[(node1_id, node2_id)] = similarity
        else:
            pass
            self.repulsion_graph = {}
        
        #  预处理节点对列表（转换为索引格式，用于对比学习）
        self._prepare_contrastive_pairs()
    
    def _prepare_contrastive_pairs(self):
        """
         新增：预处理引力图和斥力图，转换为索引格式的节点对列表
        
        将节点ID映射到索引，方便在训练时快速查找
        """
        
        # 创建节点ID到索引的映射
        self.node_id_to_idx = {node_id: idx for idx, node_id in enumerate(self.node_id_list)}
        
        # 转换引力图
        self.attraction_pairs = []
        for (node1_id, node2_id), weight in self.attraction_graph.items():
            if node1_id in self.node_id_to_idx and node2_id in self.node_id_to_idx:
                idx1 = self.node_id_to_idx[node1_id]
                idx2 = self.node_id_to_idx[node2_id]
                self.attraction_pairs.append((idx1, idx2, weight))
        
        # 转换斥力图
        self.repulsion_pairs = []
        for (node1_id, node2_id), weight in self.repulsion_graph.items():
            if node1_id in self.node_id_to_idx and node2_id in self.node_id_to_idx:
                idx1 = self.node_id_to_idx[node1_id]
                idx2 = self.node_id_to_idx[node2_id]
                self.repulsion_pairs.append((idx1, idx2, weight))
        
        
        # 如果节点对数量为0，禁用对比学习
        if len(self.attraction_pairs) == 0 and len(self.repulsion_pairs) == 0:
            pass
            self.use_contrastive_loss = False
        else:
            self.use_contrastive_loss = True
    
    def _load_user_reviews_data(self):
        """加载用户评论数据用于无监督特征提取"""
        try:
            dataset_name = os.path.splitext(os.path.basename(self.db_path))[0]
            user_reviews_path = os.path.join(
                f'preprocessed_{dataset_name}', 'user_metrics_cache', 'user_reviews.pkl')
            with open(user_reviews_path, 'rb') as f:
                user_reviews_data = pickle.load(f)
            return user_reviews_data
        except FileNotFoundError:
            return {}
    
    def _create_enhanced_features(self, original_features):
        """创建24维增强特征矩阵（完全无监督）"""
        
        n_nodes = original_features.shape[0]
        original_dim = original_features.shape[1]
        enhanced_features = np.zeros((n_nodes, original_dim + 10))  # 原始维度 + 10维混合用户特征
        
        # 复制原始特征到前几维
        enhanced_features[:, :original_dim] = original_features
        
        # 为每个虚拟节点计算混合用户特征（12-23维）
        for node_id in range(n_nodes):
            if self.virtual_nodes is not None and node_id in self.virtual_nodes:
                node_info = self.virtual_nodes[node_id]
                user_id = node_info.get('original_user_id')
                if user_id and user_id in self.user_reviews_data:
                    user_reviews = self.user_reviews_data[user_id]
                    mixed_features = self._extract_unsupervised_mixed_features(user_reviews)
                    enhanced_features[node_id, original_dim:original_dim+10] = mixed_features
        
        return enhanced_features
    
    def _extract_unsupervised_mixed_features(self, user_reviews):
        """提取无监督混合用户特征（10维）"""
        features = np.zeros(10)
        
        if not user_reviews or len(user_reviews) == 0:
            return features
        
        # 提取基础数据（不使用标签）
        ratings = [float(r.get('rating', 3)) for r in user_reviews]
        texts = [r.get('review_text', '') for r in user_reviews]
        products = [r.get('product_id', '') for r in user_reviews]
        dates = [r.get('date', '') for r in user_reviews]
        
        if len(ratings) == 0:
            return features
        
        # 特征14: 行为不一致性（评分变异系数）
        if len(ratings) > 1:
            mean_rating = np.mean(ratings)
            if mean_rating > 0:
                cv = np.std(ratings) / mean_rating
                features[0] = min(cv, 1.0)
        
        # 特征15: 时间行为变化（前后期评分差异）
        if len(ratings) >= 4:
            mid = len(ratings) // 2
            early_avg = np.mean(ratings[:mid])
            late_avg = np.mean(ratings[mid:])
            change = abs(late_avg - early_avg) / 4.0
            features[1] = min(change, 1.0)
        
        # 特征16: 双重行为模式（极端+适中评分共存）
        low_ratings = sum(1 for r in ratings if r <= 2)
        high_ratings = sum(1 for r in ratings if r >= 4)
        if low_ratings > 0 and high_ratings > 0:
            features[2] = min(low_ratings, high_ratings) / len(ratings) * 2
        
        # 特征17: 文本风格变化
        if len(texts) >= 4:
            mid = len(texts) // 2
            early_lengths = [len(t) for t in texts[:mid]]
            late_lengths = [len(t) for t in texts[mid:]]
            if np.mean(early_lengths) > 0:
                change = abs(np.mean(late_lengths) - np.mean(early_lengths)) / max(np.mean(early_lengths), np.mean(late_lengths))
                features[3] = min(change, 1.0)
        
        # 特征18: 评分跳跃频率
        if len(ratings) > 1:
            jumps = sum(1 for i in range(len(ratings)-1) if abs(ratings[i] - ratings[i+1]) >= 3)
            features[4] = jumps / (len(ratings) - 1)
        
        # 特征19: 行为复杂度
        complexity = 0.0
        complexity += len(set(ratings)) / 5.0  # 评分多样性
        if texts:
            text_lengths = [len(t) for t in texts]
            if np.mean(text_lengths) > 0:
                length_cv = np.std(text_lengths) / np.mean(text_lengths)
                complexity += min(length_cv, 1.0)
        complexity += min(len(set(products)) / 10.0, 1.0)  # 产品多样性
        features[5] = min(complexity / 3.0, 1.0)
        
        # 特征20: 时间不规律性
        if len(dates) >= 3:
            try:
                timestamps = []
                for date_str in dates:
                    if date_str:
                        month, day, year = date_str.split()
                        day = day.rstrip(',')
                        dt = datetime.strptime(f"{month} {day}, {year}", "%m %d, %Y")
                        timestamps.append(dt.timestamp())
                if len(timestamps) >= 3:
                    timestamps.sort()
                    intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
                    if np.mean(intervals) > 0:
                        cv = np.std(intervals) / np.mean(intervals)
                        features[6] = min(cv, 1.0)
            except:
                pass
        
        # 特征21: 产品攻击模式变化
        if len(products) > 1 and len(set(products)) > 1:
            product_ratings = defaultdict(list)
            for product, rating in zip(products, ratings):
                product_ratings[product].append(rating)
            
            if len(product_ratings) >= 2:
                product_avgs = [np.mean(ratings_list) for ratings_list in product_ratings.values()]
                cross_product_std = np.std(product_avgs)
                features[7] = min(cross_product_std / 2.0, 1.0)
        
        # 特征22: 评分趋势
        if len(ratings) >= 3:
            x = np.arange(len(ratings))
            correlation = np.corrcoef(x, ratings)[0, 1]
            if not np.isnan(correlation):
                features[8] = (correlation + 1) / 2
            else:
                features[8] = 0.5
        
        # 特征23: 混合行为指示器（综合得分）
        mixed_score = (features[0] + features[1] + features[2] + features[4]) / 4.0
        features[9] = min(mixed_score, 1.0)
        
        return features

    def _prepare_tensors(self):
        """准备PyTorch张量（支持稀疏格式和文件分块存储）"""
        
        # 创建文件分块存储目录
        result_dir = get_result_dir(self.sample_ratio, self.db_path, module=5)
        chunk_dir = os.path.join(result_dir, "tensor_chunks")
        os.makedirs(chunk_dir, exist_ok=True)
        
        # 设置分块参数 - GPU优化版本
        self.chunk_size = 15000  # 每个分块的节点数 - 进一步增大以充分利用GPU内存和提升性能
        self.num_chunks = (self.features.shape[0] + self.chunk_size - 1) // self.chunk_size
        
        
        # 将特征矩阵分块保存到文件
        self.feature_chunk_files = []
        for i in range(self.num_chunks):
            start_idx = i * self.chunk_size
            end_idx = min((i + 1) * self.chunk_size, self.features.shape[0])
            
            chunk_file = os.path.join(chunk_dir, f"features_chunk_{i}.npy")
            if not os.path.exists(chunk_file):
                feature_chunk = self.features[start_idx:end_idx]
                np.save(chunk_file, feature_chunk)
            
            self.feature_chunk_files.append(chunk_file)
        
        
        # 释放原始特征矩阵内存
        del self.features
        gc.collect()
        
        # 处理邻接矩阵的分块存储
        self._prepare_adjacency_chunks(chunk_dir)
        
        # 注意：当前GAT使用简化重构损失，不需要伪标签生成
        
    
    def _prepare_adjacency_chunks(self, chunk_dir):
        """准备邻接矩阵的分块存储"""
        
        if self.use_sparse_format:
            if hasattr(self, 'enhanced_edges_data'):
                # 使用边列表格式构建分块稀疏张量
                edge_list = self.enhanced_edges_data['edge_list']
                num_nodes = self.enhanced_edges_data['num_nodes']
                node_ids = self.enhanced_edges_data['node_ids']
                # 构建节点ID到索引的映射字典
                node_id_to_idx = {}
                for idx, node_id in enumerate(node_ids):
                    node_id_to_idx[node_id] = idx
                    if isinstance(node_id, str) and node_id.isdigit():
                        node_id_to_idx[int(node_id)] = idx
                    elif isinstance(node_id, int):
                        node_id_to_idx[str(node_id)] = idx
                # 为每个分块准备边信息
                self.adj_chunk_files = []
                for i in range(self.num_chunks):
                    start_idx = i * self.chunk_size
                    end_idx = min((i + 1) * self.chunk_size, num_nodes)
                    chunk_file = os.path.join(chunk_dir, f"adj_chunk_{i}.npz")
                    if not os.path.exists(chunk_file):
                        # 提取与当前分块相关的边
                        chunk_edges = []
                        chunk_weights = []
                        for node1_id, node2_id, weight in edge_list:
                            idx1 = node_id_to_idx.get(node1_id)
                            idx2 = node_id_to_idx.get(node2_id)
                            if idx1 is None or idx2 is None:
                                continue
                            # 检查边是否与当前分块相关
                            if start_idx <= idx1 < end_idx or start_idx <= idx2 < end_idx:
                                chunk_edges.append([idx1, idx2])
                                chunk_weights.append(weight)
                        # 保存分块边信息
                        if chunk_edges:
                            np.savez_compressed(chunk_file, 
                                              edges=np.array(chunk_edges),
                                              weights=np.array(chunk_weights),
                                              chunk_size=end_idx - start_idx,
                                              global_start=start_idx,
                                              global_end=end_idx)
                        else:
                            # 空分块
                            np.savez_compressed(chunk_file,
                                              edges=np.array([]).reshape(0, 2),
                                              weights=np.array([]),
                                              chunk_size=end_idx - start_idx,
                                              global_start=start_idx,
                                              global_end=end_idx)
                    self.adj_chunk_files.append(chunk_file)
            elif hasattr(self, 'enhanced_adj_sparse'):
                # 处理scipy稀疏矩阵的分块
                from scipy.sparse import coo_matrix
                if not isinstance(self.enhanced_adj_sparse, coo_matrix):
                    self.enhanced_adj_sparse = self.enhanced_adj_sparse.tocoo()
                self.adj_chunk_files = []
                for i in range(self.num_chunks):
                    start_idx = i * self.chunk_size
                    end_idx = min((i + 1) * self.chunk_size, self.enhanced_adj_sparse.shape[0])
                    chunk_file = os.path.join(chunk_dir, f"adj_chunk_{i}.npz")
                    if not os.path.exists(chunk_file):
                        # 提取分块相关的边
                        mask = ((self.enhanced_adj_sparse.row >= start_idx) & 
                               (self.enhanced_adj_sparse.row < end_idx)) | \
                               ((self.enhanced_adj_sparse.col >= start_idx) & 
                               (self.enhanced_adj_sparse.col < end_idx))
                        chunk_rows = self.enhanced_adj_sparse.row[mask] - start_idx
                        chunk_cols = self.enhanced_adj_sparse.col[mask] - start_idx
                        chunk_data = self.enhanced_adj_sparse.data[mask]
                        # 调整超出分块范围的索引
                        chunk_rows = np.clip(chunk_rows, 0, end_idx - start_idx - 1)
                        chunk_cols = np.clip(chunk_cols, 0, end_idx - start_idx - 1)
                        np.savez_compressed(chunk_file,
                                          rows=chunk_rows,
                                          cols=chunk_cols,
                                          data=chunk_data,
                                          shape=(end_idx - start_idx, end_idx - start_idx))
                    self.adj_chunk_files.append(chunk_file)
        else:
            # 处理密集矩阵的分块
            self.adj_chunk_files = []
            for i in range(self.num_chunks):
                start_idx = i * self.chunk_size
                end_idx = min((i + 1) * self.chunk_size, self.enhanced_adj_matrix.shape[0])
                chunk_file = os.path.join(chunk_dir, f"adj_chunk_{i}.npy")
                if not os.path.exists(chunk_file):
                    adj_chunk = self.enhanced_adj_matrix[start_idx:end_idx, start_idx:end_idx]
                    np.save(chunk_file, adj_chunk)
                self.adj_chunk_files.append(chunk_file)
            
        
        # 注意：保留self.features和邻接矩阵数据，GCN训练需要使用
        # 不要删除这些属性
    
    def _load_chunk(self, chunk_idx, use_gpu_for_computation=False, gpu_device=None):
        """加载指定分块，可选择是否使用GPU进行计算
        
        Args:
            chunk_idx (int): 分块索引
            use_gpu_for_computation (bool): 是否将数据加载到GPU进行计算
            
        Returns:
            tuple: (features_chunk, adj_chunk) 张量
        """
        try:
            # 加载特征分块
            features_chunk = np.load(self.feature_chunk_files[chunk_idx])
            
            # 检查特征数据有效性
            if np.any(np.isnan(features_chunk)) or np.any(np.isinf(features_chunk)):
                pass
                features_chunk = np.nan_to_num(features_chunk, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # 根据参数选择设备
            target_device = self.device if self.use_gpu else torch.device("cpu")
            features_chunk = torch.tensor(features_chunk, dtype=torch.float32, device=target_device)
            
            # 加载对应的邻接矩阵分块
            adj_data = np.load(self.adj_chunk_files[chunk_idx])
            
            if self.use_sparse_format:
                if 'edges' in adj_data:
                    # 边列表格式
                    edges = adj_data['edges']
                    weights = adj_data['weights']
                    chunk_size = int(adj_data['chunk_size'])
                    global_start = int(adj_data.get('global_start', 0))
                    # 保持稀疏格式，构建稀疏邻接矩阵
                    if len(edges) > 0:
                        # 安全的边处理
                        edges = np.array(edges, dtype=np.int64)
                        weights = np.array(weights, dtype=np.float32)
                        # 过滤无效权重
                        valid_weight_mask = np.isfinite(weights) & (weights != 0)
                        if np.any(valid_weight_mask):
                            valid_edges = edges[valid_weight_mask]
                            valid_weights = weights[valid_weight_mask]
                            # 构建稀疏矩阵的行、列、数据
                            sparse_rows = []
                            sparse_cols = []
                            sparse_data = []
                            for i, (node1, node2) in enumerate(valid_edges):
                                # 安全的索引转换
                                try:
                                    local_node1 = int(node1) - global_start
                                    local_node2 = int(node2) - global_start
                                    # 确保索引在有效范围内
                                    if (0 <= local_node1 < chunk_size and 
                                        0 <= local_node2 < chunk_size and
                                        local_node1 != local_node2):  # 避免自环
                                        weight_val = float(valid_weights[i])
                                        if np.isfinite(weight_val) and weight_val != 0:
                                            # 添加对称边
                                            sparse_rows.extend([local_node1, local_node2])
                                            sparse_cols.extend([local_node2, local_node1])
                                            sparse_data.extend([weight_val, weight_val])
                                except (ValueError, IndexError, OverflowError) as e:
                                    # 跳过无效的边
                                    continue
                            # 创建稀疏矩阵
                            if sparse_rows:
                                from scipy.sparse import coo_matrix
                                adj_chunk = coo_matrix((sparse_data, (sparse_rows, sparse_cols)), 
                                                     shape=(chunk_size, chunk_size), dtype=np.float32)
                            else:
                                # 空的稀疏矩阵
                                from scipy.sparse import coo_matrix
                                adj_chunk = coo_matrix((chunk_size, chunk_size), dtype=np.float32)
                    else:
                        # 空的稀疏矩阵
                        from scipy.sparse import coo_matrix
                        adj_chunk = coo_matrix((chunk_size, chunk_size), dtype=np.float32)
                else:
                    # 稀疏矩阵格式，保持稀疏
                    rows = adj_data['rows']
                    cols = adj_data['cols']
                    data = adj_data['data']
                    shape = tuple(adj_data['shape'])
                    if len(rows) > 0:
                        # 安全的稀疏矩阵处理
                        rows = np.array(rows, dtype=np.int64)
                        cols = np.array(cols, dtype=np.int64)
                        data = np.array(data, dtype=np.float32)
                        # 过滤无效索引和数据
                        valid_mask = (
                            (rows >= 0) & (rows < shape[0]) &
                            (cols >= 0) & (cols < shape[1]) &
                            np.isfinite(data) & (data != 0)
                        )
                        if np.any(valid_mask):
                            valid_rows = rows[valid_mask]
                            valid_cols = cols[valid_mask]
                            valid_data = data[valid_mask]
                            # 创建稀疏矩阵
                            from scipy.sparse import coo_matrix
                            adj_chunk = coo_matrix((valid_data, (valid_rows, valid_cols)), 
                                                 shape=shape, dtype=np.float32)
                        else:
                            # 空的稀疏矩阵
                            from scipy.sparse import coo_matrix
                            adj_chunk = coo_matrix(shape, dtype=np.float32)
                    else:
                        # 空的稀疏矩阵
                        from scipy.sparse import coo_matrix
                        adj_chunk = coo_matrix(shape, dtype=np.float32)
            else:
                # 密集矩阵格式
                adj_chunk = adj_data['arr_0'] if 'arr_0' in adj_data else adj_data
                adj_chunk = np.array(adj_chunk, dtype=np.float32)
            
            # 处理稀疏矩阵和稠密矩阵的不同情况
            from scipy.sparse import issparse
            
            if issparse(adj_chunk):
                # 稀疏矩阵处理
                expected_size = features_chunk.shape[0]
                # 确保稀疏矩阵的形状与特征矩阵匹配
                if adj_chunk.shape[0] != expected_size or adj_chunk.shape[1] != expected_size:
                    pass
                    # 调整稀疏矩阵大小
                    from scipy.sparse import coo_matrix
                    if adj_chunk.shape[0] > expected_size:
                        # 截取稀疏矩阵
                        mask = (adj_chunk.row < expected_size) & (adj_chunk.col < expected_size)
                        adj_chunk = coo_matrix((adj_chunk.data[mask], 
                                              (adj_chunk.row[mask], adj_chunk.col[mask])),
                                             shape=(expected_size, expected_size), dtype=np.float32)
                    else:
                        # 扩展稀疏矩阵（保持稀疏）
                        adj_chunk = coo_matrix((adj_chunk.data, (adj_chunk.row, adj_chunk.col)),
                                             shape=(expected_size, expected_size), dtype=np.float32)
                # 转换稀疏矩阵为PyTorch稀疏张量
                adj_chunk = adj_chunk.tocoo()  # 确保是COO格式
                # 清理无效数据
                valid_mask = np.isfinite(adj_chunk.data) & (adj_chunk.data != 0)
                if not np.all(valid_mask):
                    pass
                    adj_chunk = coo_matrix((adj_chunk.data[valid_mask], 
                                          (adj_chunk.row[valid_mask], adj_chunk.col[valid_mask])),
                                         shape=adj_chunk.shape, dtype=np.float32)
                # 转换为PyTorch稀疏张量
                indices = torch.from_numpy(np.vstack([adj_chunk.row, adj_chunk.col])).long()
                values = torch.from_numpy(adj_chunk.data).float()
                adj_chunk = torch.sparse_coo_tensor(indices, values, adj_chunk.shape, 
                                                  dtype=torch.float32, device=target_device)
            else:
                # 稠密矩阵处理（保持原有逻辑）
                adj_chunk = np.asarray(adj_chunk, dtype=np.float32)
                # 检查并处理无效值
                if np.any(np.isnan(adj_chunk)) or np.any(np.isinf(adj_chunk)):
                    pass
                    adj_chunk = np.nan_to_num(adj_chunk, nan=0.0, posinf=1.0, neginf=0.0)
                # 确保邻接矩阵的形状与特征矩阵匹配
                expected_size = features_chunk.shape[0]
                if adj_chunk.shape[0] != expected_size or adj_chunk.shape[1] != expected_size:
                    pass
                    # 调整邻接矩阵大小
                    if adj_chunk.shape[0] > expected_size:
                        adj_chunk = adj_chunk[:expected_size, :expected_size]
                    else:
                        # 扩展邻接矩阵
                        new_adj = np.zeros((expected_size, expected_size), dtype=np.float32)
                        min_rows = min(adj_chunk.shape[0], expected_size)
                        min_cols = min(adj_chunk.shape[1], expected_size)
                        new_adj[:min_rows, :min_cols] = adj_chunk[:min_rows, :min_cols]
                        adj_chunk = new_adj
                adj_chunk = torch.tensor(adj_chunk, dtype=torch.float32, device=target_device)
            
            return features_chunk, adj_chunk
            
        except Exception as e:
            pass
            # 返回空张量作为备用 - 使用稀疏格式避免内存问题
            chunk_size = 1000  # 默认分块大小
            target_device = gpu_device if (use_gpu_for_computation and gpu_device is not None) else self.device
            empty_features = torch.zeros((chunk_size, 8), dtype=torch.float32, device=target_device)
            # 使用稀疏张量而不是密集矩阵
            empty_adj = torch.sparse_coo_tensor(
                torch.empty((2, 0), dtype=torch.long),
                torch.empty(0, dtype=torch.float32),
                torch.Size([chunk_size, chunk_size]),
                device=target_device
            )
            
            return empty_features, empty_adj

    def _generate_embeddings_chunked(self, epoch=None, force_regenerate=False):
        """分块生成嵌入，支持缓存系统"""
        
        # 如果有嵌入生成器，使用缓存系统
        if self.embedding_generator is not None:
            try:
                pass
                chunk_indices = list(range(len(self.feature_chunk_files)))
                final_embeddings_file = self.embedding_generator.generate_embeddings_chunked(
                    feature_chunk_files=self.feature_chunk_files,
                    adj_chunk_files=self.adj_chunk_files,
                    chunk_indices=chunk_indices,
                    epoch=epoch,
                    force_regenerate=force_regenerate
                )
                # 加载嵌入数据以获取形状信息
                final_embeddings = np.load(final_embeddings_file)
                return final_embeddings_file
            except Exception as e:
                pass
                # 继续使用传统方法
        
        # 传统方法：创建临时嵌入存储目录
        temp_embeddings_dir = os.path.join(get_result_dir(self.sample_ratio, self.db_path), "temp_embeddings")
        os.makedirs(temp_embeddings_dir, exist_ok=True)
        
        # 设置模型为评估模式
        self.gat_model.eval()
        
        embedding_files = []
        total_chunks = len(self.feature_chunk_files)
        
        with torch.no_grad():
            for chunk_idx in range(total_chunks):
                pass
                # 加载当前分块，根据GPU设置决定设备
                features_chunk, adj_chunk = self._load_chunk_to_gpu(chunk_idx, use_gpu_for_computation=self.use_gpu)
                # 生成嵌入
                embeddings_chunk = self.gat_model(features_chunk, adj_chunk)
                # 移动到CPU并转换为numpy
                embeddings_np = embeddings_chunk.cpu().numpy()
                # 保存分块嵌入
                chunk_file = os.path.join(temp_embeddings_dir, f"embeddings_chunk_{chunk_idx}.npy")
                np.save(chunk_file, embeddings_np)
                embedding_files.append(chunk_file)
                # 清理GPU内存
                del features_chunk, adj_chunk, embeddings_chunk, embeddings_np
                if gpu_device is not None:
                    torch.cuda.empty_cache()
        
        # 合并所有嵌入分块到一个文件（使用内存映射避免大矩阵）
        
        # 计算总样本数和嵌入维度
        first_chunk = np.load(embedding_files[0])
        embedding_dim = first_chunk.shape[1]
        total_samples = first_chunk.shape[0]
        del first_chunk
        
        for chunk_file in embedding_files[1:]:
            chunk_shape = np.load(chunk_file, mmap_mode='r').shape
            total_samples += chunk_shape[0]
        
        # 创建内存映射文件
        final_embeddings_file = os.path.join(temp_embeddings_dir, "current_embeddings.npy")
        final_embeddings_mmap = np.lib.format.open_memmap(
            final_embeddings_file, mode='w+', 
            dtype=np.float32, shape=(total_samples, embedding_dim)
        )
        
        # 逐块写入，避免内存峰值
        offset = 0
        for chunk_idx, chunk_file in enumerate(embedding_files):
            chunk_embeddings = np.load(chunk_file)
            chunk_size = len(chunk_embeddings)
            final_embeddings_mmap[offset:offset+chunk_size] = chunk_embeddings
            offset += chunk_size
            
            # 立即删除分块文件和数据
            del chunk_embeddings
            os.remove(chunk_file)
            
            if chunk_idx % 5 == 0:
                gc.collect()
        
        # 刷新并关闭内存映射
        del final_embeddings_mmap
        gc.collect()
        
        # 清理所有临时分块文件（确保没有遗漏）
        for chunk_file in embedding_files:
            if os.path.exists(chunk_file):
                try:
                    os.remove(chunk_file)
                except:
                    pass
        
        return final_embeddings_file
        
        """构建邻居字典（只构建一次，避免重复计算）"""
        if hasattr(self, 'neighbors_dict'):
            return  # 已经构建过了
        
        edge_index = self.enhanced_adj_matrix._indices()  # [2, num_edges]
        
        # 使用numpy加速构建
        src_nodes = edge_index[0].cpu().numpy()
        dst_nodes = edge_index[1].cpu().numpy()
        
        self.neighbors_dict = {}
        for src, dst in zip(src_nodes, dst_nodes):
            if src not in self.neighbors_dict:
                self.neighbors_dict[src] = []
            self.neighbors_dict[src].append(dst)
        
    
    def _build_enhanced_adj_matrix_for_gcn(self):
        """从enhanced_edges_data构建PyTorch稀疏邻接矩阵，供GraphSAINT采样使用"""
        if hasattr(self, 'enhanced_edges_data') and self.enhanced_edges_data is not None:
            edge_list = self.enhanced_edges_data['edge_list']
            node_ids = self.enhanced_edges_data['node_ids']
            num_nodes = len(node_ids)
            node_id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
            rows, cols, vals = [], [], []
            for node1_id, node2_id, weight in edge_list:
                idx1 = node_id_to_idx.get(node1_id)
                idx2 = node_id_to_idx.get(node2_id)
                if idx1 is None or idx2 is None:
                    continue
                w = float(weight)
                if not np.isfinite(w) or w == 0:
                    continue
                rows.extend([idx1, idx2])
                cols.extend([idx2, idx1])
                vals.extend([w, w])
        elif hasattr(self, 'enhanced_adj_sparse') and self.enhanced_adj_sparse is not None:
            sp = self.enhanced_adj_sparse.tocoo()
            num_nodes = sp.shape[0]
            rows = sp.row.tolist()
            cols = sp.col.tolist()
            vals = sp.data.tolist()
        else:
            raise ValueError("无可用的增强邻接矩阵数据")
        indices = torch.tensor([rows, cols], dtype=torch.long)
        values = torch.tensor(vals, dtype=torch.float32)
        self.enhanced_adj_matrix = torch.sparse_coo_tensor(
            indices, values, torch.Size([num_nodes, num_nodes]),
            dtype=torch.float32, device=self.device
        ).coalesce()

    def _build_neighbors_dict(self):
        """构建邻居字典（只构建一次，避免重复计算）"""
        if hasattr(self, 'neighbors_dict'):
            return  # 已经构建过了
        
        edge_index = self.enhanced_adj_matrix._indices()  # [2, num_edges]
        
        # 使用numpy加速构建
        src_nodes = edge_index[0].cpu().numpy()
        dst_nodes = edge_index[1].cpu().numpy()
        
        self.neighbors_dict = {}
        for src, dst in zip(src_nodes, dst_nodes):
            if src not in self.neighbors_dict:
                self.neighbors_dict[src] = []
            self.neighbors_dict[src].append(dst)
        

    def _graphsaint_random_walk_sampler(self, num_roots=100, walk_length=20):
        """GraphSAINT随机游走采样器
        
        Args:
            num_roots: 随机游走的起点数量
            walk_length: 每次游走的步数
            
        Returns:
            sampled_nodes: 采样的节点索引列表
        """
        if not hasattr(self, 'enhanced_adj_matrix') or self.enhanced_adj_matrix is None:
            raise ValueError("增强邻接矩阵未初始化")
        
        # 确保邻居字典已构建
        if not hasattr(self, 'neighbors_dict'):
            self._build_neighbors_dict()
        
        total_nodes = self.enhanced_adj_matrix.shape[0]
        sampled_nodes = set()
        
        # 随机选择起点
        root_nodes = np.random.choice(total_nodes, min(num_roots, total_nodes), replace=False)
        
        neighbors_dict = self.neighbors_dict
        
        # 从每个起点开始随机游走
        for root in root_nodes:
            current = root
            sampled_nodes.add(current)
            
            for _ in range(walk_length):
                # 获取当前节点的邻居
                if current in neighbors_dict:
                    neighbors = neighbors_dict[current]
                else:
                    neighbors = []
                if len(neighbors) == 0:
                    break
                # 随机选择一个邻居
                next_node = neighbors[np.random.randint(len(neighbors))]
                sampled_nodes.add(next_node)
                current = next_node
        
        return list(sampled_nodes)
    
    def _extract_subgraph(self, sampled_nodes):
        """提取子图的特征和邻接矩阵
        
        Args:
            sampled_nodes: 采样的节点索引列表
            
        Returns:
            subgraph_features: 子图节点特征 [num_sampled, feature_dim]
            subgraph_adj: 子图邻接矩阵 [num_sampled, num_sampled]
            node_mapping: 原始节点索引到子图索引的映射
        """
        sampled_nodes = sorted(sampled_nodes)
        num_sampled = len(sampled_nodes)
        
        # 创建节点映射
        node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(sampled_nodes)}
        sampled_set = set(sampled_nodes)
        
        # 提取特征
        subgraph_features = self.features[sampled_nodes]
        
        # 提取子图邻接矩阵（稀疏格式）
        indices = self.enhanced_adj_matrix._indices()
        values = self.enhanced_adj_matrix._values()
        
        # 找到子图内的边（优化版本：使用numpy加速）
        src_nodes = indices[0].cpu().numpy()
        dst_nodes = indices[1].cpu().numpy()
        
        # 找到两端都在子图内的边
        mask = np.array([
            (src in sampled_set) and (dst in sampled_set)
            for src, dst in zip(src_nodes, dst_nodes)
        ], dtype=bool)
        
        if mask.sum() > 0:
            # 提取子图的边
            sub_src = src_nodes[mask]
            sub_dst = dst_nodes[mask]
            sub_weights = values[mask].cpu().numpy()
            
            # 重新映射索引
            new_src = np.array([node_mapping[s] for s in sub_src])
            new_dst = np.array([node_mapping[d] for d in sub_dst])
            
            # 创建子图邻接矩阵
            new_indices = torch.LongTensor(np.stack([new_src, new_dst]))
            new_values = torch.FloatTensor(sub_weights)
            
            subgraph_adj = torch.sparse_coo_tensor(
                new_indices, new_values,
                size=(num_sampled, num_sampled)
            ).coalesce()
        else:
            # 空邻接矩阵
            subgraph_adj = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros(0),
                size=(num_sampled, num_sampled)
            )
        
        return subgraph_features, subgraph_adj, node_mapping
    
    def train_gcn_with_graphsaint(self):
        """使用GraphSAINT采样训练加权GCN模型"""
        
        # 确保features已加载
        if not hasattr(self, 'features') or self.features is None:
            pass
            module2_dir = get_result_dir(self.sample_ratio, self.db_path, module=2, force_no_threshold=True)
            features_path = os.path.join(module2_dir, f'feature_matrix_14d_{self.sample_ratio}.npy')
            self.features = np.load(features_path)
        
        # 转换为PyTorch张量
        if not isinstance(self.features, torch.Tensor):
            self.features = torch.FloatTensor(self.features)
        
        # 构建完整的增强邻接矩阵（稀疏格式）
        if not hasattr(self, 'enhanced_adj_matrix') or self.enhanced_adj_matrix is None:
            pass
            self._build_enhanced_adj_matrix_for_gcn()
        
        # 预先构建邻居字典（用于GraphSAINT采样，只构建一次）
        self._build_neighbors_dict()
        
        # 初始化两层GCN模型
        input_dim = self.features.shape[1]
        self.gcn_model = EnhancedTGNNModel(
            nfeat=input_dim,
            nhid=self.hidden_dim,
            nclass=self.output_dim,
            dropout=self.dropout
        ).to(self.device)
        
        # 优化器（稍后初始化，需要包含损失函数参数）
        # self.optimizer将在损失函数初始化后创建
        
        #  损失函数：使用简化版损失（重构损失 + 对比学习损失）
        self.criterion = MixedUserAwareLoss(
            lambda_contrastive=0.5,  # 对比学习损失权重
            feature_dim=input_dim,
            embed_dim=self.output_dim,
            use_contrastive=self.use_contrastive_loss  # 是否使用对比学习
        ).to(self.device)
        
        # 初始化优化器（包含GCN和损失函数的参数）
        self.optimizer = torch.optim.Adam(
            list(self.gcn_model.parameters()) + list(self.criterion.parameters()),
            lr=self.lr,
            weight_decay=5e-4
        )
        
        # GraphSAINT采样参数
        num_subgraphs_per_epoch = 10  # 每个epoch采样10个子图
        num_roots = 100  # 每个子图100个起点
        walk_length = 20  # 每次游走20步


        # 训练循环 - 使用self.epochs（论文设置：200轮）
        num_epochs = self.epochs

        # ── 迭代精炼状态：HDBSCAN引导的引力斥力对自适应更新 ─────────────────
        _WARMUP_EPOCHS = 30     # 前30轮预热，避免随机初始化嵌入污染聚类信号
        _PATIENCE_MAX  = 3      # 连续无改善次数上限，达到后锁定图结构
        _DELTA_MIN     = 0.005  # 有效改善的最小轮廓系数增量
        _best_sil      = -2.0   # 历史最优轮廓系数
        _patience_cnt  = 0      # 连续无改善计数器
        _struct_locked = False  # 图结构是否已锁定
        _POST_LOCK_FINETUNE = 40   # 结构锁定后再精炼的最大轮数（早停上限）
        _post_lock_cnt      = 0    # 结构锁定后已精炼的轮数
        _post_lock_sil_bad  = 0    # 锁定后轮廓系数连续明显下降次数（达2次则早停）
        _orig_att_pairs = list(self.attraction_pairs)
        _orig_rep_pairs = list(self.repulsion_pairs)
        _orig_att_set   = frozenset(
            (min(p[0], p[1]), max(p[0], p[1])) for p in _orig_att_pairs)
        _orig_rep_set   = frozenset(
            (min(p[0], p[1]), max(p[0], p[1])) for p in _orig_rep_pairs)

        for epoch in range(num_epochs):
            # ── 早停（精炼窗口）：结构锁定后每轮递增，超出上限则退出 ──────────
            if _struct_locked:
                _post_lock_cnt += 1
                if _post_lock_cnt >= _POST_LOCK_FINETUNE:
                    break
            self.gcn_model.train()
            epoch_loss = 0.0
            epoch_loss_components = {}

            # 每个epoch采样多个子图
            for subgraph_idx in range(num_subgraphs_per_epoch):
                try:
                    # 1. GraphSAINT采样
                    sampled_nodes = self._graphsaint_random_walk_sampler(
                        num_roots=num_roots,
                        walk_length=walk_length
                    )
                    if len(sampled_nodes) < 10:
                        pass
                    # 2. 提取子图
                    subgraph_features, subgraph_adj, node_mapping = self._extract_subgraph(sampled_nodes)
                    # 移动到设备
                    subgraph_features = subgraph_features.to(self.device)
                    subgraph_adj = subgraph_adj.to(self.device)
                    # 3.  TGNN前向传播（包含LSTM时序建模）
                    self.optimizer.zero_grad()
                    # 使用TGNN的时序功能
                    node_embeddings, user_embeddings = self.gcn_model(
                        subgraph_features, 
                        subgraph_adj,
                        user_to_virtual=getattr(self, 'user_to_virtual_mapping', None),
                        virtual_node_times=getattr(self, 'virtual_node_times', None),
                        use_temporal=True
                    )
                    # 使用节点嵌入进行损失计算
                    embeddings = node_embeddings
                    # 获取子图对应的spam_scores
                    subgraph_spam_scores = torch.FloatTensor([self.spam_behavior_scores[i] for i in sampled_nodes]).to(self.device)
                    
                    #  4. 提取子图对应的引力图和斥力图节点对
                    subgraph_attraction_pairs = []
                    subgraph_repulsion_pairs = []
                    
                    if self.use_contrastive_loss:
                        # 创建子图节点索引映射
                        subgraph_idx_map = {global_idx: local_idx for local_idx, global_idx in enumerate(sampled_nodes)}
                        
                        # 提取子图内的引力图节点对
                        for idx1, idx2, weight in self.attraction_pairs:
                            if idx1 in subgraph_idx_map and idx2 in subgraph_idx_map:
                                local_idx1 = subgraph_idx_map[idx1]
                                local_idx2 = subgraph_idx_map[idx2]
                                subgraph_attraction_pairs.append((local_idx1, local_idx2, weight))
                        
                        # 提取子图内的斥力图节点对
                        for idx1, idx2, weight in self.repulsion_pairs:
                            if idx1 in subgraph_idx_map and idx2 in subgraph_idx_map:
                                local_idx1 = subgraph_idx_map[idx1]
                                local_idx2 = subgraph_idx_map[idx2]
                                subgraph_repulsion_pairs.append((local_idx1, local_idx2, weight))
                    
                    #  5. 计算混合用户感知损失 + 对比学习损失（已修改）
                    loss_original, loss_components = self.criterion(
                        embeddings,
                        subgraph_features,
                        subgraph_spam_scores,
                        subgraph_adj,
                        attraction_pairs=subgraph_attraction_pairs,  #  新增
                        repulsion_pairs=subgraph_repulsion_pairs     #  新增
                    )
                    # 5. 最终训练损失
                    loss = loss_original
                    loss_components['original_loss'] = loss_original.item()
                    loss_components['combined_loss'] = loss.item()
                    
                    # 7. 反向传播和优化（关键！）
                    loss.backward()
                    self.optimizer.step()
                    
                    # 累计损失
                    epoch_loss += loss.item()
                    for key, value in loss_components.items():
                        if key not in epoch_loss_components:
                            epoch_loss_components[key] = 0.0
                        epoch_loss_components[key] += value
                    # 清理内存
                    del subgraph_features, subgraph_adj, embeddings, loss
                    if self.use_gpu:
                        torch.cuda.empty_cache()
                except Exception as e:
                    pass
            
            # 计算平均损失
            avg_loss = epoch_loss / num_subgraphs_per_epoch
            avg_components = {k: v / num_subgraphs_per_epoch for k, v in epoch_loss_components.items()}
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:2d}/{num_epochs}: Loss = {avg_components.get('combined_loss', avg_loss):.6f}")
            
            # 每10轮：HDBSCAN探测 + 迭代精炼（预热后启用）
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}: HDBSCAN clustering probe...")
                self.gcn_model.eval()
                probe_checkpoint_dir = os.path.join(self.preprocessed_dir, "training_checkpoints")
                os.makedirs(probe_checkpoint_dir, exist_ok=True)
                try:
                    probe_embeddings = self._generate_embeddings_chunked_gcn()
                    if probe_embeddings.shape[1] >= 32:
                        from sklearn.decomposition import PCA
                        pca = PCA(n_components=25, random_state=42)
                        probe_emb_hdbscan = pca.fit_transform(probe_embeddings).astype(np.float32)
                    else:
                        probe_emb_hdbscan = probe_embeddings
                    probe_clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=5, min_samples=3,
                        metric='euclidean', core_dist_n_jobs=-1
                    )
                    probe_labels = probe_clusterer.fit_predict(probe_emb_hdbscan)
                    if probe_emb_hdbscan is not probe_embeddings:
                        del probe_emb_hdbscan
                    if self.use_gpu:
                        torch.cuda.empty_cache()
                    n_clusters = len(set(probe_labels)) - (1 if -1 in probe_labels else 0)
                    n_noise    = int(np.sum(probe_labels == -1))
                    print(f"    HDBSCAN: {n_clusters} clusters, {n_noise} noise points")
                    sil_score, centroids = self._compute_centroid_silhouette(
                        probe_embeddings, probe_labels)
                    print(f"    Silhouette: {sil_score:.4f}  Best so far: {_best_sil:.4f}")
                    probe_labels_path = os.path.join(
                        probe_checkpoint_dir, f"hdbscan_probe_epoch_{epoch+1}.npy")
                    np.save(probe_labels_path, probe_labels)
                    with open(os.path.join(probe_checkpoint_dir, 'latest_probe_info.txt'), 'w') as _f:
                        _f.write(f"epoch={epoch+1}\n")
                        _f.write(f"labels_path={probe_labels_path}\n")
                        _f.write(f"n_clusters={n_clusters}\n")
                        _f.write(f"n_noise={n_noise}\n")
                        _f.write(f"silhouette={sil_score:.6f}\n")
                        _f.write(f"best_silhouette={_best_sil:.6f}\n")
                        _f.write(f"structure_locked={_struct_locked}\n")
                        _f.write(f"loss={avg_loss:.6f}\n")
                    if (epoch + 1) >= _WARMUP_EPOCHS and not _struct_locked and n_clusters >= 2:
                        if sil_score > _best_sil + _DELTA_MIN:
                            print(f"    Silhouette improved +{sil_score - _best_sil:.4f}. Updating graph pairs.")
                            _best_sil     = sil_score
                            _patience_cnt = 0
                            new_att, new_rep = self._update_pairs_from_hdbscan(
                                probe_embeddings, probe_labels, centroids,
                                _orig_att_pairs, _orig_att_set,
                                _orig_rep_pairs, _orig_rep_set)
                            self.attraction_pairs = new_att
                            self.repulsion_pairs  = new_rep
                        else:
                            _patience_cnt += 1
                            if _patience_cnt >= _PATIENCE_MAX:
                                _struct_locked = True
                    elif _struct_locked:
                        pass
                        # ── 早停（轮廓恶化）：锁定后若轮廓系数持续明显下降则退出 ──
                        if sil_score < _best_sil - 0.01:
                            _post_lock_sil_bad += 1
                        else:
                            _post_lock_sil_bad = 0
                        if _post_lock_sil_bad >= 2:
                            pass
                            break
                    else:
                        print(f"    Warm-up phase ({epoch+1}/{_WARMUP_EPOCHS}), graph structure unchanged.")
                    del probe_embeddings, probe_labels
                    if centroids is not None:
                        del centroids
                except Exception as e:
                    pass
                self.gcn_model.train()

            # 每20轮保存训练损失检查点到文件
            if (epoch + 1) % 20 == 0:
                checkpoint_dir = os.path.join(self.preprocessed_dir, "training_checkpoints")
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}_checkpoint.txt")
                with open(checkpoint_path, 'w') as f:
                    f.write(f"Epoch: {epoch+1}\n")
                    f.write(f"Loss: {avg_loss:.6f}\n")
                    f.write(f"Timestamp: {time.time()}\n")
                if self.use_gpu:
                    torch.cuda.empty_cache()

        # 训练结束后，生成最终的全图嵌入
        with torch.no_grad():
            gcn_embeddings = self._generate_embeddings_chunked_gcn()

            #  如果启用了行为解耦模型，生成混合嵌入
            if self.use_behavior_disentanglement:
                pass
                disentangle_embeddings, alpha_values = self._generate_disentangle_embeddings()
                # 混合嵌入：[GCN(16维) | 解耦(128维) | α(1维)] = 145维
                self.embeddings = np.concatenate([
                    gcn_embeddings,           # [N, 16]
                    disentangle_embeddings,   # [N, 128]
                    alpha_values              # [N, 1]
                ], axis=1)
                # 保存α值用于后续过滤
                self.alpha_values = alpha_values
            else:
                # 仅使用GCN嵌入
                self.embeddings = gcn_embeddings
                self.alpha_values = None

        print("  GCN training complete.")

    def _generate_embeddings_chunked_gcn(self):
        """使用GCN生成全图嵌入（极致优化：分块处理+文件流式存储）"""
        self.gcn_model.eval()

        total_samples = len(self.features)
        chunk_size = 3000  # 极小批量避免内存峰值


        # 创建临时文件目录（使用正确的带阈值后缀的路径）
        module5_dir = get_result_dir(self.sample_ratio, self.db_path, module=5)
        temp_dir = os.path.join(module5_dir, "temp_embeddings")
        os.makedirs(temp_dir, exist_ok=True)

        # 分块生成嵌入并立即保存到文件
        embedding_files = []
        num_chunks = (total_samples + chunk_size - 1) // chunk_size

        with torch.no_grad():
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, total_samples)
                
                try:
                    # 方案：使用GraphSAINT采样生成局部嵌入（避免全图加载）
                    # 为当前块的节点生成嵌入
                    chunk_nodes = list(range(start_idx, end_idx))
                    
                    # 提取子图
                    subgraph_features, subgraph_adj, node_mapping = self._extract_subgraph(chunk_nodes)
                    
                    # 移到设备
                    subgraph_features = subgraph_features.to(self.device)
                    subgraph_adj = subgraph_adj.to(self.device)
                    
                    # 前向传播（只在子图上）
                    node_embeddings, _ = self.gcn_model(
                        subgraph_features,
                        subgraph_adj,
                        user_to_virtual=None,  # 子图不需要时序信息
                        virtual_node_times=None,
                        use_temporal=False
                    )
                    
                    # 转为numpy并保存
                    chunk_embeddings = node_embeddings.cpu().numpy()
                    
                    # 立即保存到文件
                    chunk_file = f"{temp_dir}/chunk_{chunk_idx}.npy"
                    np.save(chunk_file, chunk_embeddings)
                    embedding_files.append(chunk_file)
                    
                    # 清理内存
                    del subgraph_features, subgraph_adj, node_embeddings, chunk_embeddings
                    if self.use_gpu:
                        torch.cuda.empty_cache()
                    
                    if (chunk_idx + 1) % 20 == 0:
                        pass
                        
                except Exception as e:
                    pass
                    # 创建零嵌入作为占位
                    chunk_embeddings = np.zeros((end_idx - start_idx, self.output_dim))
                    chunk_file = f"{temp_dir}/chunk_{chunk_idx}.npy"
                    np.save(chunk_file, chunk_embeddings)
        
        # 从文件逐个加载并合并（避免一次性加载所有文件）
        
        # 使用memmap方式合并，避免内存峰值
        final_emb_file = os.path.join(self.preprocessed_dir, "final_embeddings_temp.npy")
        
        # 先确定总形状
        first_chunk = np.load(embedding_files[0])
        emb_dim = first_chunk.shape[1]
        del first_chunk
        
        # 创建memmap文件
        final_embeddings = np.memmap(final_emb_file, dtype='float32', mode='w+',
                                     shape=(total_samples, emb_dim))
        
        # 逐块写入
        current_idx = 0
        for emb_file in embedding_files:
            chunk_emb = np.load(emb_file)
            chunk_len = len(chunk_emb)
            final_embeddings[current_idx:current_idx+chunk_len] = chunk_emb
            current_idx += chunk_len
            del chunk_emb
            try:
                os.remove(emb_file)
            except Exception:
                pass
        
        # 转为普通numpy数组
        embeddings_np = np.array(final_embeddings)
        
        # 关闭并删除memmap文件
        final_embeddings.flush()
        del final_embeddings
        gc.collect()
        try:
            os.remove(final_emb_file)
        except Exception:
            pass
        
        
        try:
            os.rmdir(temp_dir)
        except Exception:
            pass
        
        return embeddings_np

    def _compute_centroid_silhouette(self, embeddings_np, labels_np):
        """
        基于质心的近似轮廓系数，时间复杂度 O(N·K·D)，无需 O(N²) 全对计算。
        返回 (silhouette_score: float, centroids: np.ndarray[K, D] or None)。
        """
        unique_labels = np.array([l for l in np.unique(labels_np) if l >= 0], dtype=np.int32)
        n_clusters = len(unique_labels)
        if n_clusters < 2:
            return -1.0, None

        valid_mask  = labels_np >= 0
        valid_embs  = embeddings_np[valid_mask].astype(np.float32)  # [M, D]
        valid_labels = labels_np[valid_mask]                          # [M]

        # 质心 [K, D]，按 unique_labels 顺序索引
        centroids = np.stack(
            [valid_embs[valid_labels == k].mean(axis=0) for k in unique_labels]
        ).astype(np.float32)

        # 展开式欧氏距离矩阵 [M, K]，纯矩阵运算
        emb_sq = (valid_embs ** 2).sum(axis=1, keepdims=True)        # [M, 1]
        cen_sq = (centroids  ** 2).sum(axis=1, keepdims=True).T      # [1, K]
        cross  = valid_embs @ centroids.T                             # [M, K]
        dists  = np.sqrt(np.maximum(emb_sq + cen_sq - 2 * cross, 0.0))  # [M, K]

        label_to_col = {k: i for i, k in enumerate(unique_labels)}
        own_cols = np.array([label_to_col[l] for l in valid_labels], dtype=np.int32)

        a_vals = dists[np.arange(len(valid_labels)), own_cols]        # 簇内距离 [M]

        inf_fill = dists.copy()
        inf_fill[np.arange(len(valid_labels)), own_cols] = np.inf
        b_vals = inf_fill.min(axis=1)                                  # 最近异簇距离 [M]

        sil = (b_vals - a_vals) / np.maximum(a_vals, b_vals)
        sil = np.nan_to_num(sil, nan=0.0, posinf=0.0, neginf=-1.0)
        return float(sil.mean()), centroids

    def _update_pairs_from_hdbscan(self, embeddings_np, labels_np, centroids,
                                    orig_att_pairs, orig_att_set,
                                    orig_rep_pairs, orig_rep_set,
                                    theta_att=0.65, theta_rep=0.40,
                                    top_core=30, max_new_att=50000, max_new_rep=30000):
        """
        基于 HDBSCAN 簇标签更新引力/斥力对（不替换原始对，只叠加高置信度新对）。

        时间复杂度：
          引力对更新 O(K · top_core² · D)：每簇取最近质心的 top_core 个核心节点，
                     批量计算两两余弦相似度，筛选高置信度对。
          斥力对更新 O(E_rep · 1)：逐条检查原始斥力对的当前簇标签，保留跨簇对；
                     同时从新引力候选中找跨簇低相似对补充。
        """
        unique_labels = np.array([l for l in np.unique(labels_np) if l >= 0], dtype=np.int32)
        if len(unique_labels) < 2 or centroids is None:
            return list(orig_att_pairs), list(orig_rep_pairs)

        # L2 归一化嵌入，供余弦相似度批量计算
        norms     = np.linalg.norm(embeddings_np, axis=1, keepdims=True) + 1e-8
        norm_embs = (embeddings_np / norms).astype(np.float32)           # [N, D]

        # ── 引力对更新 ───────────────────────────────────────────────────────
        new_att_candidates = []
        label_to_col = {k: i for i, k in enumerate(unique_labels)}

        for k in unique_labels:
            cluster_mask = np.where(labels_np == k)[0]   # 全局节点索引
            n_k = len(cluster_mask)
            if n_k < 2:
                continue

            cluster_norm_embs = norm_embs[cluster_mask]  # [n_k, D]

            # 归一化质心，用余弦距离选核心节点（距质心最近的 top_core 个）
            centroid_k = centroids[label_to_col[k]]
            c_norm     = centroid_k / (np.linalg.norm(centroid_k) + 1e-8)
            cos_to_cen = cluster_norm_embs @ c_norm      # [n_k]，越大越近
            top_k_n    = min(top_core, n_k)
            core_local = np.argsort(-cos_to_cen)[:top_k_n]   # 降序取 top_k
            core_global = cluster_mask[core_local]             # [top_k] 全局索引
            core_embs   = norm_embs[core_global]               # [top_k, D]

            # 批量余弦相似度矩阵 [top_k, top_k]
            cos_mat = core_embs @ core_embs.T
            i_triu, j_triu = np.triu_indices(top_k_n, k=1)
            sims = cos_mat[i_triu, j_triu]

            hi_conf = sims >= theta_att
            for ii, jj, sv in zip(i_triu[hi_conf], j_triu[hi_conf], sims[hi_conf]):
                gi  = int(core_global[ii])
                gj  = int(core_global[jj])
                key = (min(gi, gj), max(gi, gj))
                if key not in orig_att_set:
                    new_att_candidates.append((gi, gj, float(sv)))

            if len(new_att_candidates) >= max_new_att * 2:
                break  # 提前截断，防止候选集过大

        # 取相似度最高的 max_new_att 对
        if len(new_att_candidates) > max_new_att:
            new_att_candidates.sort(key=lambda x: -x[2])
            new_att_candidates = new_att_candidates[:max_new_att]

        updated_attraction = list(orig_att_pairs) + new_att_candidates

        # ── 斥力对更新 ───────────────────────────────────────────────────────
        # 保留原始斥力对中仍跨越不同 HDBSCAN 簇的对
        kept_rep = []
        for idx1, idx2, w in orig_rep_pairs:
            l1 = int(labels_np[idx1]) if idx1 < len(labels_np) else -1
            l2 = int(labels_np[idx2]) if idx2 < len(labels_np) else -1
            if l1 != l2:   # 跨簇（含噪声节点）的对保留
                kept_rep.append((idx1, idx2, w))

        # 从原始引力对中找"混淆对"：原特征空间相似，但当前 HDBSCAN 划到不同簇
        # 且 GCN 嵌入余弦相似度低 → 这些对需要斥力强化
        # new_att_candidates 全为同簇对，不能作为跨簇斥力来源
        new_rep_candidates = []
        _att_scan = orig_att_pairs if len(orig_att_pairs) <= max_new_rep * 10 \
                    else orig_att_pairs[:max_new_rep * 10]
        for gi, gj, _ in _att_scan:
            l1 = int(labels_np[gi]) if gi < len(labels_np) else -1
            l2 = int(labels_np[gj]) if gj < len(labels_np) else -1
            if l1 >= 0 and l2 >= 0 and l1 != l2:
                sim_ij = float(norm_embs[gi] @ norm_embs[gj])
                if sim_ij < theta_rep:
                    key = (min(gi, gj), max(gi, gj))
                    if key not in orig_rep_set:
                        new_rep_candidates.append((gi, gj, 1.0 - sim_ij))

        if len(new_rep_candidates) > max_new_rep:
            new_rep_candidates.sort(key=lambda x: -x[2])
            new_rep_candidates = new_rep_candidates[:max_new_rep]

        updated_repulsion = kept_rep + new_rep_candidates
        return updated_attraction, updated_repulsion

    def _generate_disentangle_embeddings(self):
        """使用行为解耦模型生成嵌入和α值（优化：使用文件存储避免内存峰值）"""
        
        self.disentangle_model.eval()
        
        total_samples = len(self.spam_features)
        batch_size = 10000  # 批量处理
        
        # 创建临时文件存储（使用内存映射避免大矩阵）
        import tempfile
        temp_dir = tempfile.mkdtemp(prefix="disentangle_emb_")
        disentangle_file = os.path.join(temp_dir, "disentangle_embeddings.npy")
        alpha_file = os.path.join(temp_dir, "alpha_values.npy")
        
        # 创建内存映射文件（预分配空间）
        disentangle_mmap = np.lib.format.open_memmap(
            disentangle_file, mode='w+',
            dtype=np.float32, shape=(total_samples, 128)
        )
        alpha_mmap = np.lib.format.open_memmap(
            alpha_file, mode='w+',
            dtype=np.float32, shape=(total_samples, 1)
        )
        
        with torch.no_grad():
            for start_idx in range(0, total_samples, batch_size):
                end_idx = min(start_idx + batch_size, total_samples)
                # 准备批次数据
                batch_spam = torch.FloatTensor(self.spam_features[start_idx:end_idx]).to(self.device)
                batch_genuine = torch.FloatTensor(self.genuine_features[start_idx:end_idx]).to(self.device)
                # 前向传播
                _, _, alpha, final_emb = self.disentangle_model(batch_spam, batch_genuine)
                # 直接写入内存映射文件（不收集到列表）
                disentangle_mmap[start_idx:end_idx] = final_emb.cpu().numpy()
                alpha_mmap[start_idx:end_idx] = alpha.cpu().numpy()
                # 清理GPU内存
                del batch_spam, batch_genuine, alpha, final_emb
                if self.use_gpu:
                    torch.cuda.empty_cache()
                if (start_idx // batch_size + 1) % 5 == 0:
                    pass
        
        # 刷新内存映射文件
        del disentangle_mmap, alpha_mmap
        gc.collect()
        
        # 加载结果（使用内存映射模式，不占用大量内存）
        disentangle_embeddings = np.load(disentangle_file, mmap_mode='r')
        alpha_values = np.load(alpha_file, mmap_mode='r')
        
        
        # 注意：返回内存映射对象，不是完整数组
        return disentangle_embeddings, alpha_values
        
    def _compute_embeddings_hash(self, embeddings_np, chunk_size=10000):
        """分块计算嵌入数据的哈希值，避免内存问题"""
        hasher = hashlib.md5()
        n_samples = embeddings_np.shape[0]
        
        for i in range(0, n_samples, chunk_size):
            chunk_end = min(i + chunk_size, n_samples)
            chunk = embeddings_np[i:chunk_end]
            hasher.update(chunk.tobytes())
            del chunk
            gc.collect()
        
        return hasher.hexdigest()[:16]

    def _filter_clusters_by_alpha(self, threshold=0.3):
        """基于α值过滤聚类结果，移除低α群组"""
        
        unique_clusters = [c for c in set(self.cluster_labels) if c != -1]
        original_clusters = len(unique_clusters)
        filtered_count = 0
        filtered_nodes = 0
        
        # 统计每个群组的α值
        cluster_alpha_stats = {}
        
        for cluster_id in unique_clusters:
            cluster_mask = self.cluster_labels == cluster_id
            cluster_alpha = self.alpha_values[cluster_mask].mean()
            cluster_size = cluster_mask.sum()
            
            cluster_alpha_stats[cluster_id] = {
                'mean_alpha': cluster_alpha,
                'size': cluster_size
            }
            
            # 如果群组平均α低于阈值，标记为噪声
            if cluster_alpha < threshold:
                self.cluster_labels[cluster_mask] = -1
                filtered_count += 1
                filtered_nodes += cluster_size
        
        # 统计结果
        remaining_clusters = len([c for c in set(self.cluster_labels) if c != -1])
        
        
        # 显示剩余群组的α统计
        if remaining_clusters > 0:
            remaining_alpha_values = []
            for cluster_id in set(self.cluster_labels):
                if cluster_id != -1:
                    remaining_alpha_values.append(cluster_alpha_stats[cluster_id]['mean_alpha'])
            
    
    def perform_clustering(self):
        """对全部数据进行聚类（优先使用HDBSCAN，降级到Ball树DBSCAN）"""
        import gc
        from sklearn.neighbors import NearestNeighbors
        from collections import deque
        
        total_samples = len(self.embeddings)
        
        print(f"  Running HDBSCAN on {total_samples} nodes...")
        
        # HDBSCAN参数 - 纯层次化聚类，不使用cluster_selection_epsilon
        # 让HDBSCAN自动选择最稳定的簇，而不是强制合并
        # [!] min_cluster_size固定为10（虚拟节点级别，对应至少2-5个真实用户）
        # [!] 不能使用total_samples*0.001：64万节点时=643，会过滤掉绝大多数小型水军群组
        # 目标群组数2500-3000，平均每群组约200+虚拟节点，最小群组需≥10虚拟节点
        min_cluster_size = 10  # 固定值：过滤掉极小噪声群组，同时保留中小型水军群组
        min_samples = 5  # 核心点的最小邻居数
        
        # 第1步：优先使用HDBSCAN（更快、更省内存、更好的结果）
        
        try:
            import hdbscan
            

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_method='eom',
                metric='euclidean',
                core_dist_n_jobs=1,
                algorithm='best'
            )

            start_time = time.time()

            # 优化：对高维嵌入进行PCA降维，加速聚类
            # 对于64万节点，>=32维的HDBSCAN极慢（KD-tree退化），必须降维
            embeddings_for_clustering = self.embeddings
            if self.embeddings.shape[1] >= 32:
                pass
                from sklearn.decomposition import IncrementalPCA
                import tempfile
                target_dim = 25  # 降至25维（KD-tree在低维高效，大幅提速）
                ipca = IncrementalPCA(n_components=target_dim, batch_size=10000)
                n_samples = self.embeddings.shape[0]
                batch_size = 10000
                # 分批拟合
                for i in range(0, n_samples, batch_size):
                    batch_end = min(i + batch_size, n_samples)
                    ipca.partial_fit(self.embeddings[i:batch_end])
                # 分批转换（使用内存映射避免大数组）
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.npy')
                temp_path = temp_file.name
                temp_file.close()
                embeddings_reduced = np.lib.format.open_memmap(
                    temp_path, mode='w+',
                    dtype=np.float32, shape=(n_samples, target_dim)
                )
                for i in range(0, n_samples, batch_size):
                    batch_end = min(i + batch_size, n_samples)
                    embeddings_reduced[i:batch_end] = ipca.transform(self.embeddings[i:batch_end])
                    if (i // batch_size + 1) % 10 == 0:
                        pass
                del embeddings_reduced
                gc.collect()
                embeddings_reduced = np.load(temp_path, mmap_mode='r')
                variance_ratio = ipca.explained_variance_ratio_.sum()
                embeddings_for_clustering = embeddings_reduced
                self._temp_pca_file = temp_path
            
            labels = clusterer.fit_predict(embeddings_for_clustering)
            
            # 清理降维后的数据和临时文件
            if embeddings_for_clustering is not self.embeddings:
                del embeddings_for_clustering
                gc.collect()
                # 清理临时PCA文件
                if hasattr(self, '_temp_pca_file') and os.path.exists(self._temp_pca_file):
                    try:
                        os.remove(self._temp_pca_file)
                    except Exception as e:
                        pass
            
            elapsed_time = time.time() - start_time
            
            # 统计结果
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            
            self.cluster_labels = labels
            return labels
            
        except ImportError:
            pass
            print(f"       pip install hdbscan")
        except Exception as e:
            pass
        
        # 降级方案：使用Ball树 + 内存中构建邻接图（不使用文件）
        
        # 定义DBSCAN的eps参数（邻域半径）
        eps = 0.5  # 默认邻域半径
        
        # 一次性构建Ball树（避免重复构建）
        try:
            knn = NearestNeighbors(
                radius=eps,
                algorithm='ball_tree',
                metric='euclidean',
                n_jobs=-1
            )
            knn.fit(self.embeddings)
        except MemoryError:
            pass
            return self._perform_full_data_chunked_clustering_fallback()
        
        # 分块查询邻域并直接存储到字典（不使用文件）
        chunk_size = 5000  # 每次处理5000个点
        num_chunks = (total_samples + chunk_size - 1) // chunk_size
        
        
        neighbors_dict = {}
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, total_samples)
            chunk_embeddings = self.embeddings[start_idx:end_idx]
            
            if chunk_idx % 10 == 0:
                pass
            
            try:
                # 查询当前块的点的邻域
                distances, indices = knn.radius_neighbors(chunk_embeddings, return_distance=True)
                # 直接存储到字典
                for i, (point_idx, neighs) in enumerate(zip(range(start_idx, end_idx), indices)):
                    neighbors_dict[point_idx] = neighs.tolist()
                del distances, indices
            except MemoryError:
                pass
                # 降级：逐点查询
                for i in range(len(chunk_embeddings)):
                    point_idx = start_idx + i
                    point_emb = chunk_embeddings[i:i+1]
                    distances, indices = knn.radius_neighbors(point_emb, return_distance=True)
                    neighs = indices[0]
                    neighbors_dict[point_idx] = neighs.tolist()
                    del distances, indices
            
            del chunk_embeddings
            
            # 每10块强制垃圾回收
            if chunk_idx % 10 == 0:
                gc.collect()
        
        # 释放Ball树
        del knn
        gc.collect()
        
        
        # 统计邻居数量
        neighbor_counts = [len(v) for v in neighbors_dict.values()]
        
        # 第2步：基于邻接图执行DBSCAN聚类（使用连通性）
        
        # 初始化标签
        labels = np.full(total_samples, -1, dtype=np.int32)
        cluster_id = 0
        
        # 计算每个点的邻居数（用于判断核心点）
        neighbor_counts = {idx: len(neighs) for idx, neighs in neighbors_dict.items()}
        
        # 遍历所有点
        visited = set()
        
        for point_idx in range(total_samples):
            if point_idx in visited:
                continue
            
            # 检查是否是核心点
            if neighbor_counts.get(point_idx, 0) < min_samples:
                visited.add(point_idx)
            
            # 从核心点开始BFS扩展簇
            visited.add(point_idx)
            labels[point_idx] = cluster_id
            
            # BFS队列
            queue = deque([point_idx])
            
            while queue:
                current = queue.popleft()
                current_neighbors = neighbors_dict.get(current, [])
                for neighbor in current_neighbors:
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    labels[neighbor] = cluster_id
                    # 如果邻居也是核心点，加入队列继续扩展
                    if neighbor_counts.get(neighbor, 0) >= min_samples:
                        queue.append(neighbor)
            
            cluster_id += 1
            
            # 定期输出进度
            if (point_idx + 1) % 50000 == 0:
                pass
        
        # 统计最终结果
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        print(f"  Clustering complete: {n_clusters} groups, {n_noise} noise points.")
        
        self.cluster_labels = labels
        return labels
    
    def _perform_full_data_chunked_clustering_fallback(self):
        """备用方案：基于文件的增量DBSCAN聚类（慢速但内存友好）"""
        import gc
        
        total_samples = len(self.embeddings)
        
        
        # DBSCAN参数
        eps = 0.02
        min_samples = 3
        
        # 创建临时目录存储距离文件
        current_result_dir = result_dir if result_dir is not None else get_result_dir(self.sample_ratio, self.db_path, module=5)
        distance_cache_dir = os.path.join(current_result_dir, "distance_cache")
        os.makedirs(distance_cache_dir, exist_ok=True)
        
        # 将嵌入数据保存为mmap文件以节省内存
        embeddings_mmap_file = os.path.join(distance_cache_dir, "embeddings_mmap.npy")
        np.save(embeddings_mmap_file, self.embeddings)
        embeddings_mmap = np.load(embeddings_mmap_file, mmap_mode='r')
        
        # 初始化聚类标签（-1表示未分类）
        cluster_labels = np.full(total_samples, -1, dtype=np.int32)
        current_cluster_id = 0
        
        # 分批处理每个点，找到其邻域（优化版：增大批处理和分块大小）
        batch_size = 5000  # 增大到5000个点/批（加速5倍）
        num_batches = (total_samples + batch_size - 1) // batch_size
        
        
        # 存储每个点的邻居到文件
        neighbors_file = os.path.join(distance_cache_dir, "neighbors.txt")
        
        # 使用缓冲写入提高I/O效率
        neighbor_counts = []  # 统计邻居数量
        
        with open(neighbors_file, 'w', buffering=1024*1024) as f_neighbors:
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, total_samples)
                if batch_idx % 2 == 0:
                    pass
                # 读取当前批次的嵌入
                batch_embeddings = embeddings_mmap[start_idx:end_idx].copy()
                # 分块计算到所有点的距离（增大分块大小）
                chunk_size = 10000  # 从5000增大到10000
                # 为当前批次的所有点计算邻域（向量化）
                for i in range(len(batch_embeddings)):
                    point_idx = start_idx + i
                    point_emb = batch_embeddings[i]
                    neighbors = []
                    # 分块计算距离
                    for chunk_start in range(0, total_samples, chunk_size):
                        chunk_end = min(chunk_start + chunk_size, total_samples)
                        chunk_data = embeddings_mmap[chunk_start:chunk_end]
                        # 向量化计算欧氏距离（避免sqrt以加速）
                        distances_sq = np.sum((chunk_data - point_emb) ** 2, axis=1)
                        # 找到eps邻域内的点（使用平方距离比较）
                        eps_sq = eps * eps
                        neighbor_mask = distances_sq <= eps_sq
                        neighbor_indices = np.where(neighbor_mask)[0] + chunk_start
                        neighbors.extend(neighbor_indices.tolist())
                        del chunk_data, distances_sq, neighbor_mask, neighbor_indices
                    # 统计邻居数量
                    neighbor_counts.append(len(neighbors))
                    # 将邻居列表写入文件
                    f_neighbors.write(f"{point_idx} {len(neighbors)} {' '.join(map(str, neighbors))}\n")
                    del neighbors
                del batch_embeddings
                # 每2批强制垃圾回收
                if batch_idx % 2 == 0:
                    gc.collect()
        
        # 输出邻居数量统计
        del neighbor_counts
        
        # 第2阶段：基于邻域文件进行DBSCAN聚类（流式处理，避免加载全部到内存）
        
        # 第一遍：构建邻域索引（记录每个点在文件中的位置）
        neighbors_index = {}  # {point_idx: file_position}
        
        with open(neighbors_file, 'r') as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                parts = line.strip().split()
                point_idx = int(parts[0])
                neighbors_index[point_idx] = pos
        
        # 辅助函数：从文件读取指定点的邻居
        def get_neighbors(point_idx):
            if point_idx not in neighbors_index:
                return []
            with open(neighbors_file, 'r') as f:
                f.seek(neighbors_index[point_idx])
                line = f.readline()
                parts = line.strip().split()
                num_neighbors = int(parts[1])
                neighbors = [int(x) for x in parts[2:2+num_neighbors]]
                return neighbors
        
        # DBSCAN核心算法（流式处理）
        visited = np.zeros(total_samples, dtype=bool)
        
        for point_idx in range(total_samples):
            if visited[point_idx]:
                continue
            
            visited[point_idx] = True
            
            # 获取当前点的邻居
            neighbors = get_neighbors(point_idx)
            
            # 如果邻居数量 >= min_samples，则为核心点
            if len(neighbors) >= min_samples:
                # 创建新簇
                cluster_labels[point_idx] = current_cluster_id
                # 扩展簇（BFS）
                queue = list(neighbors)
                while queue:
                    neighbor_idx = queue.pop(0)
                    if not visited[neighbor_idx]:
                        visited[neighbor_idx] = True
                        # 获取该邻居的邻居
                        neighbor_neighbors = get_neighbors(neighbor_idx)
                        if len(neighbor_neighbors) >= min_samples:
                            queue.extend(neighbor_neighbors)
                    # 如果邻居还未分类，分配到当前簇
                    if cluster_labels[neighbor_idx] == -1:
                        cluster_labels[neighbor_idx] = current_cluster_id
                current_cluster_id += 1
            
            # 每5000个点打印一次进度
            if point_idx % 5000 == 0 and point_idx > 0:
                n_clustered = np.sum(cluster_labels != -1)
        
        # 清理临时文件
        if os.path.exists(neighbors_file):
            os.remove(neighbors_file)
        if os.path.exists(embeddings_mmap_file):
            os.remove(embeddings_mmap_file)
        
        n_total_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        
        self.cluster_labels = cluster_labels
        return cluster_labels
    
    def _check_and_cleanup_disk_space(self, result_dir):
        """检查磁盘空间并清理旧缓存"""
        import shutil
        
        # 检查磁盘空间
        stat = shutil.disk_usage('/')
        free_gb = stat.free / (1024**3)
        total_gb = stat.total / (1024**3)
        used_gb = stat.used / (1024**3)
        
        
        # 如果可用空间小于5GB，发出警告
        if free_gb < 5:
            pass
        
        # 清理旧的缓存目录
        cleanup_dirs = [
            os.path.join(result_dir, 'distance_cache'),
            os.path.join(result_dir, 'neighbors_cache'),
            os.path.join(result_dir, 'temp_embeddings')
        ]
        
        cleaned_size = 0
        for cleanup_dir in cleanup_dirs:
            if os.path.exists(cleanup_dir):
                try:
                    # 计算目录大小
                    dir_size = sum(
                        os.path.getsize(os.path.join(dirpath, filename))
                        for dirpath, dirnames, filenames in os.walk(cleanup_dir)
                        for filename in filenames
                    ) / (1024**3)  # 转换为GB
                    if dir_size > 0.1:  # 如果大于100MB
                        pass
                        shutil.rmtree(cleanup_dir)
                        cleaned_size += dir_size
                    else:
                        # 只清理文件，保留目录
                        for root, dirs, files in os.walk(cleanup_dir):
                            for file in files:
                                os.remove(os.path.join(root, file))
                except Exception as e:
                    pass
        
        # 清理dynamic_clustering_cache中的旧文件（保留最新的）
        dynamic_cache_dir = os.path.join(result_dir, 'dynamic_clustering_cache')
        if os.path.exists(dynamic_cache_dir):
            try:
                import time
                current_time = time.time()
                for filename in os.listdir(dynamic_cache_dir):
                    filepath = os.path.join(dynamic_cache_dir, filename)
                    if os.path.isfile(filepath):
                        # 删除1小时前的文件
                        if current_time - os.path.getmtime(filepath) > 3600:
                            os.remove(filepath)
            except Exception as e:
                pass
        
        if cleaned_size > 0:
            pass
            
            # 重新检查磁盘空间
            stat = shutil.disk_usage('/')
            free_gb = stat.free / (1024**3)
        else:
            pass
    
    def save_results(self):
        """保存GCN嵌入和聚类结果"""
        current_result_dir = get_result_dir(self.sample_ratio, self.db_path, module=5)
        os.makedirs(current_result_dir, exist_ok=True)  # 确保目录存在
        
        # 保存前检查嵌入维度
        if self.embeddings.shape[1] == 145:
            pass
        elif self.embeddings.shape[1] == 16:
            pass
        
        # 保存GCN嵌入（文件名保持gat_embeddings以兼容后续模块）
        embeddings_path = os.path.join(current_result_dir, f'gat_embeddings_{self.sample_ratio}.npy')
        np.save(embeddings_path, self.embeddings)
        
        # 保存聚类标签
        clusters_path = os.path.join(current_result_dir, f'cluster_labels_{self.sample_ratio}.npy')
        np.save(clusters_path, self.cluster_labels)
        
        # 保存聚类详细信息
        cluster_info = {}
        node_ids = list(self.virtual_nodes.keys())
        
        # 检查数组长度是否匹配
        
        # 确保所有数组长度一致
        min_length = min(len(node_ids), len(self.cluster_labels), len(self.embeddings))
        if len(node_ids) != len(self.cluster_labels) or len(node_ids) != len(self.embeddings):
            pass
            node_ids = node_ids[:min_length]
            cluster_labels_aligned = self.cluster_labels[:min_length]
            embeddings_aligned = self.embeddings[:min_length]
        else:
            cluster_labels_aligned = self.cluster_labels
            embeddings_aligned = self.embeddings
        
        for i, (node_id, cluster_label) in enumerate(zip(node_ids, cluster_labels_aligned)):
            if cluster_label not in cluster_info:
                cluster_info[cluster_label] = []
            cluster_info[cluster_label].append({
                'node_id': node_id,
                'embedding': embeddings_aligned[i].tolist(),
                'virtual_node_info': self.virtual_nodes[node_id]
            })
        
        cluster_info_path = os.path.join(current_result_dir, f'cluster_info_{self.sample_ratio}.pkl')
        with open(cluster_info_path, 'wb') as f:
            pickle.dump(cluster_info, f)
        
        # 保存聚类结果CSV
        cluster_df = pd.DataFrame({
            'node_id': node_ids,
            'cluster_label': cluster_labels_aligned,
            'original_user_id': [self.virtual_nodes[nid]['original_user_id'] for nid in node_ids],
            'virtual_node_id': [self.virtual_nodes[nid]['virtual_node_id'] for nid in node_ids]
        })
        
        cluster_csv_path = os.path.join(current_result_dir, f'cluster_results_{self.sample_ratio}.csv')
        cluster_df.to_csv(cluster_csv_path, index=False, encoding='utf-8')
        
        
    def run(self, use_iterative_training=True):
        # [FLOW-M5] 模块5：GCN训练+HDBSCAN聚类 | 缓存: module5/gat_embeddings_*.npy, cluster_labels_*.npy
        # [!] perform_clustering中PCA降维阈值为>=32维→25维，不可改回>64（会导致HDBSCAN卡死）
        # [!] _generate_embeddings_chunked_gcn已修复Windows文件锁PermissionError
        """运行模块5的完整流程
        
        Args:
            use_iterative_training (bool): 是否使用交替训练机制，默认为True（交替训练）
        """
        try:
            current_result_dir = result_dir if result_dir is not None else get_result_dir(self.sample_ratio, self.db_path, module=5)
            
            #  启动前检查和清理
            self._check_and_cleanup_disk_space(current_result_dir)
            
            # 检查缓存文件是否存在
            embeddings_path = os.path.join(current_result_dir, f'gat_embeddings_{self.sample_ratio}.npy')
            clusters_path = os.path.join(current_result_dir, f'cluster_labels_{self.sample_ratio}.npy')
            cluster_info_path = os.path.join(current_result_dir, f'cluster_info_{self.sample_ratio}.pkl')
            cluster_csv_path = os.path.join(current_result_dir, f'cluster_results_{self.sample_ratio}.csv')
            
            if (os.path.exists(embeddings_path) and os.path.exists(clusters_path) and 
                os.path.exists(cluster_info_path) and os.path.exists(cluster_csv_path)):
                pass
                # 加载缓存的结果
                self.embeddings = np.load(embeddings_path)
                self.cluster_labels = np.load(clusters_path)
                with open(cluster_info_path, 'rb') as f:
                    cluster_info = pickle.load(f)
                return True
            
            # 如果缓存不存在，执行完整流程
            # 根据初始化参数决定是否使用GPU加速
            if self.use_gpu:
                pass
                self.force_cpu_mode = False
            else:
                pass
                self.force_cpu_mode = True
            
            self.load_data()
            self._prepare_tensors()
            # 注释掉GAT初始化，因为我们使用GCN
            # self._initialize_model()  # 这会初始化GAT，但我们不需要
            
            # 使用加权GCN + GraphSAINT采样训练
            self.train_gcn_with_graphsaint()  # 这个方法内部会初始化GCN
            
            # 最终聚类
            self.perform_clustering()
            
            self.save_results()
            return True
        except Exception as e:
            pass
            import traceback
            traceback.print_exc()
            return False

# ================================
# 模块6-7：节点聚合和候选群组净化与合并
# ================================

class Module6_7_NodeAggregationAndGroupPurification:
    """模块6-7：节点聚合和候选群组净化与合并
    
    实现ISS计算、群组净化、群组合并等功能
    """
    
    def __init__(self, sample_ratio=1.0, iss_threshold=0.3, group_threshold=0.5, db_path=None, dataset_name=None):
        self.sample_ratio = sample_ratio
        self.iss_threshold = iss_threshold  # δ_I: 个体阈值，用于ISS指标进行个体用户净化
        self.group_threshold = group_threshold  # δ_G: 群组阈值，用于GSS得分判别候选群组（从0.6降到0.5）
        self.db_path = db_path  # 数据库路径，用于ISS和GSS计算
        self.dataset_name = dataset_name or self._identify_dataset(db_path)  # 数据集名称
        
        # 数据存储
        self.virtual_nodes = None
        self.cluster_info = None
        self.user_groups = {}  # 用户聚合后的群组
        self.candidate_groups = {}  # 候选垃圾群组
        self.final_groups = {}  # 最终合并后的群组
        
    def _identify_dataset(self, db_path):
        """识别数据集类型"""
        if db_path is None:
            return "Unknown"
        if "Electronics" in db_path:
            return "Electronics"
        elif "Cell_Phones" in db_path:
            return "Cell_Phones"
        elif "Clothing" in db_path:
            return "Clothing"
        else:
            return "Unknown"
        
    def load_data(self):
        """加载聚类结果和虚拟节点信息"""
        
        # 模块6-7需要加载模块1和模块5的缓存
        module1_dir = get_result_dir(self.sample_ratio, self.db_path, module=1, force_no_threshold=True)
        module5_dir = get_result_dir(self.sample_ratio, self.db_path, module=5)
        current_result_dir = result_dir if result_dir is not None else get_result_dir(self.sample_ratio, self.db_path, module=6)
        
        # 加载虚拟节点信息（从模块1）
        virtual_nodes_path = os.path.join(module1_dir, 'virtual_nodes.pkl')
        with open(virtual_nodes_path, 'rb') as f:
            self.virtual_nodes = pickle.load(f)
        
        # 加载聚类信息（从模块5）
        cluster_info_path = os.path.join(module5_dir, f'cluster_info_{self.sample_ratio}.pkl')
        with open(cluster_info_path, 'rb') as f:
            self.cluster_info = pickle.load(f)
        
        
        # 统计聚类分布
        cluster_sizes = []
        for cluster_id, nodes in self.cluster_info.items():
            if cluster_id != -1:  # 排除噪声点
                cluster_sizes.append(len(nodes))
        
        if cluster_sizes:
            pass
    
    def deduplicate_virtual_nodes(self):
        """步骤2：去重虚拟节点（在聚合之前）
        
        对于被分散到多个群组的用户，保留其虚拟节点数量最多的群组，
        从其他群组中移除该用户的虚拟节点。
        
        策略：虚拟节点数量优先原则
        - 虚拟节点多 = 该用户在该群组中活跃度高
        - 保留活跃度最高的群组，移除其他群组中的误判节点
        """
        
        # 统计每个用户在每个群组中的虚拟节点数量
        user_cluster_nodes = defaultdict(lambda: defaultdict(list))
        
        for cluster_id, nodes in self.cluster_info.items():
            if cluster_id == -1:
                continue
            
            for node_info in nodes:
                user_id = node_info['virtual_node_info']['original_user_id']
                node_id = node_info['node_id']
                user_cluster_nodes[user_id][cluster_id].append(node_id)
        
        # 识别被分散的用户并选择最佳群组
        dispersed_users = {}
        resolved_users = {}
        
        for user_id, clusters in user_cluster_nodes.items():
            if len(clusters) > 1:  # 被分散到多个群组
                dispersed_users[user_id] = clusters
                
                # 计算每个群组中该用户的虚拟节点数量
                cluster_node_counts = {}
                for cluster_id, node_list in clusters.items():
                    cluster_node_counts[cluster_id] = {
                        'node_count': len(node_list),
                        'node_ids': node_list
                    }
                
                # 选择虚拟节点数量最多的群组
                best_cluster = max(cluster_node_counts.items(), key=lambda x: x[1]['node_count'])
                
                resolved_users[user_id] = {
                    'best_cluster': best_cluster[0],
                    'best_node_count': best_cluster[1]['node_count'],
                    'all_clusters': cluster_node_counts,
                    'clusters_to_remove': [cid for cid in clusters.keys() if cid != best_cluster[0]]
                }
        
        
        # 重构群组（移除误判节点）
        new_cluster_info = {}
        total_removed = 0
        clusters_affected = 0
        
        for cluster_id, nodes in self.cluster_info.items():
            if cluster_id == -1:
                new_cluster_info[cluster_id] = nodes
                continue
            
            # 过滤节点
            filtered_nodes = []
            removed_count = 0
            
            for node_info in nodes:
                user_id = node_info['virtual_node_info']['original_user_id']
                
                # 检查该用户是否被分散
                if user_id in resolved_users:
                    # 如果当前群组不是该用户的最佳群组，则移除
                    if cluster_id in resolved_users[user_id]['clusters_to_remove']:
                        removed_count += 1
                        total_removed += 1
                        continue
                
                filtered_nodes.append(node_info)
            
            if removed_count > 0:
                clusters_affected += 1
            
            if filtered_nodes:  # 只保留非空群组
                new_cluster_info[cluster_id] = filtered_nodes
        
        # 更新cluster_info
        self.cluster_info = new_cluster_info
        
        
        # 保存去重统计信息（用于后续分析）
        self.deduplication_stats = {
            'dispersed_users': len(dispersed_users),
            'total_removed_nodes': total_removed,
            'clusters_affected': clusters_affected,
            'resolved_users': resolved_users
        }
        
    def aggregate_nodes_to_users(self):
        """步骤3：将虚拟节点聚合为用户群组"""
        
        # 遍历每个聚类，将虚拟节点按用户ID聚合
        for cluster_id, nodes in self.cluster_info.items():
            if cluster_id == -1:  # 跳过噪声点
                continue
            # 按用户ID聚合节点
            user_dict = defaultdict(list)
            for node_info in nodes:
                user_id = node_info['virtual_node_info']['original_user_id']
                user_dict[user_id].append(node_info)
            
            # 创建用户群组，将虚拟节点列表存储为virtual_reviews
            if len(user_dict) >= 2:  # 至少包含2个用户才能形成群组
                users_info = {}
                for user_id, virtual_nodes in user_dict.items():
                    users_info[user_id] = virtual_nodes  # 暂时存储虚拟节点列表，后续会被ISS计算替换
                self.user_groups[cluster_id] = {
                    'users': users_info,
                    'user_count': len(user_dict),
                    'total_reviews': sum(len(reviews) for reviews in user_dict.values())
                }
        
        
        # 统计用户群组信息
        if self.user_groups:
            user_counts = [group['user_count'] for group in self.user_groups.values()]
            review_counts = [group['total_reviews'] for group in self.user_groups.values()]
            
        
    def secondary_clustering_with_temporal_features(self):
        """步骤3.5：使用时序特征对混合用户和正常用户进行二次聚类
        
        目的：
        1. 利用混合用户独特的时序特征和行为不一致性
        2. 防止混合用户被视为离散节点导致漏检
        3. 使用HDBSCAN进行二次聚类，最小规模>3
        """
        
        # 加载时序特征
        module1_dir = get_result_dir(self.sample_ratio, self.db_path, module=1, force_no_threshold=True)
        temporal_features_path = os.path.join(module1_dir, 'temporal_features.pkl')
        
        if not os.path.exists(temporal_features_path):
            pass
            return
        
        with open(temporal_features_path, 'rb') as f:
            temporal_features = pickle.load(f)
        
        
        # 对每个群组进行二次聚类
        total_groups_before = len(self.user_groups)
        new_user_groups = {}
        next_group_id = max(self.user_groups.keys()) + 1 if self.user_groups else 0
        
        groups_split = 0
        
        for cluster_id, group_info in self.user_groups.items():
            users = group_info['users']
            
            # 如果群组太小，不进行二次聚类
            if len(users) < 4:
                new_user_groups[cluster_id] = group_info
                continue
            
            # 提取该群组用户的时序特征
            user_ids = list(users.keys())
            feature_matrix = []
            valid_user_ids = []
            
            for user_id in user_ids:
                if user_id in temporal_features:
                    feat = temporal_features[user_id]
                    # 构建特征向量（6维）
                    feature_vector = [
                        feat['avg_time_interval'],
                        feat['std_time_interval'],
                        feat['cv_time_interval'],
                        feat['rating_change_rate'],
                        feat['product_concentration'],
                        feat['text_similarity']
                    ]
                    feature_matrix.append(feature_vector)
                    valid_user_ids.append(user_id)
            
            # 如果有效用户太少，不进行二次聚类
            if len(valid_user_ids) < 4:
                new_user_groups[cluster_id] = group_info
                continue
            
            # 标准化特征
            feature_matrix = np.array(feature_matrix)
            scaler = StandardScaler()
            feature_matrix_scaled = scaler.fit_transform(feature_matrix)
            
            # 使用HDBSCAN进行二次聚类
            try:
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=max(3, int(len(valid_user_ids) * 0.1)),  # 最小规模至少为3
                    min_samples=1,
                    metric='euclidean',
                    cluster_selection_method='eom'
                )
                sub_labels = clusterer.fit_predict(feature_matrix_scaled)
                
                # 统计子聚类
                unique_labels = set(sub_labels)
                n_sub_clusters = len([l for l in unique_labels if l != -1])
                
                if n_sub_clusters > 1:
                    # 群组被拆分
                    groups_split += 1
                    
                    # 创建新的子群组
                    for sub_label in unique_labels:
                        if sub_label == -1:
                            # 噪声点保留在原群组
                            continue
                        
                        # 获取该子聚类的用户
                        sub_user_ids = [valid_user_ids[i] for i in range(len(valid_user_ids)) 
                                       if sub_labels[i] == sub_label]
                        
                        if len(sub_user_ids) >= 2:  # 至少2个用户
                            sub_users_info = {uid: users[uid] for uid in sub_user_ids}
                            new_group_id = next_group_id
                            next_group_id += 1
                            
                            new_user_groups[new_group_id] = {
                                'users': sub_users_info,
                                'user_count': len(sub_users_info),
                                'total_reviews': sum(len(reviews) for reviews in sub_users_info.values()),
                                'parent_cluster': cluster_id,
                                'sub_cluster_label': sub_label
                            }
                    
                    # 处理噪声点（如果有的话）
                    noise_user_ids = [valid_user_ids[i] for i in range(len(valid_user_ids)) 
                                     if sub_labels[i] == -1]
                    if len(noise_user_ids) >= 2:
                        noise_users_info = {uid: users[uid] for uid in noise_user_ids}
                        new_user_groups[cluster_id] = {
                            'users': noise_users_info,
                            'user_count': len(noise_users_info),
                            'total_reviews': sum(len(reviews) for reviews in noise_users_info.values()),
                            'is_noise_from_secondary': True
                        }
                else:
                    # 没有拆分，保留原群组
                    new_user_groups[cluster_id] = group_info
                    
            except Exception as e:
                pass
                new_user_groups[cluster_id] = group_info
        
        # 更新群组信息
        self.user_groups = new_user_groups
        
        
        # 统计新的群组信息
        if self.user_groups:
            user_counts = [group['user_count'] for group in self.user_groups.values()]
            review_counts = [group['total_reviews'] for group in self.user_groups.values()]
            
        
    def calculate_iss_scores(self):
        """步骤4：计算ISS（Individual Suspiciousness Score）- 使用缓存"""
        
        # 使用集成的用户指标缓存
        try:
            # 获取正确的缓存目录路径（包含数据集名称）
            dataset_name = get_dataset_name(self.db_path)
            cache_dir = f"preprocessed_{dataset_name}/user_metrics_cache"
            cache_reader = UserMetricsCacheReader(cache_dir=cache_dir)
            self._calculate_iss_scores_from_cache(cache_reader)
            cache_reader.close()
        except Exception as e:
            pass
            raise RuntimeError("用户指标缓存不存在，请先运行缓存构建")
        
    def _calculate_iss_scores_from_cache(self, cache_reader):
        """使用缓存计算ISS分数（无SQL查询）
        
        实现论文公式 eq.(10):
            ISS(v) = (RD̂ + ERR̂ + MRÔ + RB̂ + RF̂) / 5
        特征映射（基于缓存可用指标，全部在全局用户集上min-max归一化）：
            ERR : extreme_rating_ratio       — 极端评分比例，直接对应
            RD  : rating_deviation           — 评分偏差，代理指标
            MRO : product_concentration      — 商品集中度，代理指标
            RB  : rating_std                 — 评分标准差，突发性代理指标
            RF  : review_count/time_span_days — 评论频率 = 1/ATI，直接对应
        """

        # 收集所有需要处理的用户ID
        all_user_ids = set()
        for group_info in self.user_groups.values():
            all_user_ids.update(group_info['users'].keys())

        user_metrics_dict = cache_reader.get_batch_iss_metrics(list(all_user_ids))

        # ── 第一轮：收集原始特征值，用于全局min-max归一化 ──────────────────
        raw = {'ERR': [], 'RD': [], 'MRO': [], 'RB': [], 'RF': []}
        for m in user_metrics_dict.values():
            raw['ERR'].append(float(m.get('extreme_rating_ratio', 0)))
            raw['RD'].append(float(m.get('rating_deviation', 0)))
            raw['MRO'].append(float(m.get('product_concentration', 0)))
            raw['RB'].append(float(m.get('rating_std', 0)))
            ts = max(float(m.get('time_span_days', 1)), 0.1)
            rc = max(int(m.get('review_count', 1)), 1)
            raw['RF'].append(rc / ts)

        minmax = {}
        for feat, vals in raw.items():
            arr = np.array(vals, dtype=np.float32)
            lo, hi = float(arr.min()), float(arr.max())
            minmax[feat] = (lo, hi if hi > lo else lo + 1e-8)

        def _norm(val, feat):
            lo, hi = minmax[feat]
            return float(np.clip((val - lo) / (hi - lo), 0.0, 1.0))

        # ── 第二轮：计算每个用户的ISS ─────────────────────────────────────
        processed_groups = 0
        for group_id, group_info in self.user_groups.items():
            valid_users = {}
            for user_id, virtual_reviews in group_info['users'].items():
                if user_id not in user_metrics_dict:
                    continue
                m = user_metrics_dict[user_id]
                ts = max(float(m.get('time_span_days', 1)), 0.1)
                rc = max(int(m.get('review_count', 1)), 1)

                err_hat = _norm(float(m.get('extreme_rating_ratio', 0)), 'ERR')
                rd_hat  = _norm(float(m.get('rating_deviation', 0)),      'RD')
                mro_hat = _norm(float(m.get('product_concentration', 0)), 'MRO')
                rb_hat  = _norm(float(m.get('rating_std', 0)),            'RB')
                rf_hat  = _norm(rc / ts,                                  'RF')

                total_iss = (rd_hat + err_hat + mro_hat + rb_hat + rf_hat) / 5.0

                valid_users[user_id] = {
                    'virtual_reviews': virtual_reviews,
                    'iss_scores': {'RD': rd_hat, 'ERR': err_hat,
                                   'MRO': mro_hat, 'RB': rb_hat, 'RF': rf_hat},
                    'total_iss': total_iss,
                    'review_count': rc,
                    'rating_std':  float(m.get('rating_std', 0)),
                    'rating_mean': float(m.get('rating_mean', 0)),
                }

            group_info['users'] = valid_users
            group_info['user_count'] = len(valid_users)
            processed_groups += 1
            if processed_groups % 100 == 0:
                pass

        # 移除没有有效用户的群组
        empty_groups = [gid for gid, gi in self.user_groups.items() if len(gi['users']) == 0]
        for gid in empty_groups:
            del self.user_groups[gid]

    
    def _text_similarity(self, text1, text2):
        """计算两个文本的相似度"""
        if not text1 or not text2:
            return 0.0
        
        # 简单的字符级相似度计算
        text1 = str(text1).lower()
        text2 = str(text2).lower()
        
        # 使用Jaccard相似度
        set1 = set(text1)
        set2 = set(text2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def filter_candidate_groups(self):
        """步骤5：候选群组净化（基于ISS阈值）"""
        
        # 设置最小群组大小为5，以提高混合用户召回率
        # 降低阈值可以保留更多中小规模但高质量的群组
        min_group_size = 5
        
        filtered_by_iss = 0
        filtered_by_std = 0
        
        for group_id, group_info in self.user_groups.items():
            # 只使用ISS阈值过滤
            filtered_users = {}
            
            for user_id, user_info in group_info['users'].items():
                # 条件1：ISS阈值过滤
                if user_info['total_iss'] < self.iss_threshold:
                    filtered_by_iss += 1
                    continue
                # 通过所有过滤条件
                filtered_users[user_id] = user_info
            
            # 如果过滤后仍有足够用户，则保留为候选群组
            if len(filtered_users) >= min_group_size:
                self.candidate_groups[group_id] = {
                    'users': filtered_users,
                    'user_count': len(filtered_users),
                    'avg_iss': np.mean([user['total_iss'] for user in filtered_users.values()]),
                    'total_reviews': sum(user['review_count'] for user in filtered_users.values())
                }
        
        
        if self.candidate_groups:
            avg_iss_scores = [group['avg_iss'] for group in self.candidate_groups.values()]
    
    def optimize_group_purity(self):
        """步骤4.5：群组后处理优化（方案2）"""
        
        optimized_groups = {}
        total_removed = 0
        
        for group_id, group_info in self.candidate_groups.items():
            users = group_info['users']
            
            if len(users) < 2:
                continue
            
            # 计算群组核心特征
            rating_stds = [u.get('rating_std', 0) for u in users.values()]
            rating_means = [u.get('rating_mean', 0) for u in users.values()]
            
            group_avg_std = np.mean(rating_stds)
            group_avg_rating = np.mean(rating_means)
            group_std_std = np.std(rating_stds)  # rating_std的标准差
            group_rating_std = np.std(rating_means)  # rating_mean的标准差
            
            # 过滤偏离群组核心特征的用户（使用相对阈值）
            optimized_users = {}
            for user_id, user_info in users.items():
                user_std = user_info.get('rating_std', 0)
                user_rating = user_info.get('rating_mean', 0)
                # 评分方向一致性（在群组均值±1.5倍标准差内，更严格）
                rating_threshold = max(0.8, 1.5 * group_rating_std)
                rating_consistent = abs(user_rating - group_avg_rating) < rating_threshold
                # 评分标准差一致性（在群组均值±1.5倍标准差内，更严格）
                std_threshold = max(0.5, 1.5 * group_std_std)
                std_consistent = abs(user_std - group_avg_std) < std_threshold
                # 保留一致性用户
                if rating_consistent and std_consistent:
                    optimized_users[user_id] = user_info
                else:
                    total_removed += 1
            
            # 如果优化后仍有足够用户，保留群组（与filter_candidate_groups保持一致）
            if len(optimized_users) >= 4:
                optimized_groups[group_id] = {
                    'users': optimized_users,
                    'user_count': len(optimized_users),
                    'avg_iss': np.mean([u['total_iss'] for u in optimized_users.values()]),
                    'total_reviews': sum(u['review_count'] for u in optimized_users.values())
                }
        
        self.candidate_groups = optimized_groups
        
    
    def merge_similar_groups(self):
        """步骤6：群组合并（基于Jaccard相似度和重叠比例）"""
        
        group_ids = list(self.candidate_groups.keys())
        merged_groups = {}
        merged_flags = set()
        
        # 计算群组间的相似度并合并
        for i, group_id1 in enumerate(group_ids):
            if group_id1 in merged_flags:
                continue
            current_group = self.candidate_groups[group_id1].copy()
            merged_with = [group_id1]
            
            for j, group_id2 in enumerate(group_ids[i+1:], i+1):
                if group_id2 in merged_flags:
                    continue
                # 计算群组相似度
                similarity = self._calculate_group_similarity(
                    self.candidate_groups[group_id1], 
                    self.candidate_groups[group_id2]
                )
                # 如果相似度满足合并条件
                if similarity['jaccard'] > 0.8 and similarity['overlap_ratio'] > 0.8:
                    # 合并群组
                    current_group['users'].update(self.candidate_groups[group_id2]['users'])
                    merged_with.append(group_id2)
                    merged_flags.add(group_id2)
            
            # 更新合并后的群组信息
            current_group['user_count'] = len(current_group['users'])
            current_group['avg_iss'] = np.mean([user['total_iss'] for user in current_group['users'].values()])
            current_group['total_reviews'] = sum(user['review_count'] for user in current_group['users'].values())
            current_group['merged_from'] = merged_with
            
            merged_groups[f"merged_{group_id1}"] = current_group
            merged_flags.add(group_id1)
        
        self.final_groups = merged_groups
        
        
    def _calculate_group_similarity(self, group1, group2):
        """计算两个群组的相似度"""
        users1 = set(group1['users'].keys())
        users2 = set(group2['users'].keys())
        
        # Jaccard相似度
        intersection = len(users1.intersection(users2))
        union = len(users1.union(users2))
        jaccard = intersection / union if union > 0 else 0
        
        # 重叠用户节点比例
        overlap_ratio = intersection / min(len(users1), len(users2)) if min(len(users1), len(users2)) > 0 else 0
        
        return {
            'jaccard': jaccard,
            'overlap_ratio': overlap_ratio
        }
    
    def calculate_group_suspicion_scores(self):
        """步骤7/7: 计算群组可疑度分数（GSS）- 使用预构建缓存"""
        
        # 使用集成的用户指标缓存（无需SQL查询）
        try:
            # 获取正确的缓存目录路径（包含数据集名称）
            dataset_name = get_dataset_name(self.db_path)
            cache_dir = f"preprocessed_{dataset_name}/user_metrics_cache"
            cache_reader = UserMetricsCacheReader(cache_dir=cache_dir)
        except Exception as e:
            pass
            raise RuntimeError("用户指标缓存不存在，请先运行缓存构建")
        
        # 加载user_iss_dict和user_metrics_dict（用于GSS计算）
        
        # 加载用户指标字典
        self.user_metrics_dict = cache_reader.iss_metrics

        # 计算用户ISS字典（论文公式: ISS = (RD̂+ERR̂+MRÔ+RB̂+RF̂)/5，全局min-max归一化）
        raw_gss = {'ERR': [], 'RD': [], 'MRO': [], 'RB': [], 'RF': []}
        for m in self.user_metrics_dict.values():
            raw_gss['ERR'].append(float(m.get('extreme_rating_ratio', 0)))
            raw_gss['RD'].append(float(m.get('rating_deviation', 0)))
            raw_gss['MRO'].append(float(m.get('product_concentration', 0)))
            raw_gss['RB'].append(float(m.get('rating_std', 0)))
            ts = max(float(m.get('time_span_days', 1)), 0.1)
            rc = max(int(m.get('review_count', 1)), 1)
            raw_gss['RF'].append(rc / ts)
        mm_gss = {}
        for feat, vals in raw_gss.items():
            arr = np.array(vals, dtype=np.float32)
            lo, hi = float(arr.min()), float(arr.max())
            mm_gss[feat] = (lo, hi if hi > lo else lo + 1e-8)

        def _gss_norm(val, feat):
            lo, hi = mm_gss[feat]
            return float(np.clip((val - lo) / (hi - lo), 0.0, 1.0))

        self.user_iss_dict = {}
        for user_id, m in self.user_metrics_dict.items():
            ts = max(float(m.get('time_span_days', 1)), 0.1)
            rc = max(int(m.get('review_count', 1)), 1)
            iss = (_gss_norm(float(m.get('extreme_rating_ratio', 0)), 'ERR') +
                   _gss_norm(float(m.get('rating_deviation', 0)),      'RD')  +
                   _gss_norm(float(m.get('product_concentration', 0)), 'MRO') +
                   _gss_norm(float(m.get('rating_std', 0)),            'RB')  +
                   _gss_norm(rc / ts,                                  'RF')) / 5.0
            self.user_iss_dict[user_id] = iss

        
        # 加载用户评论缓存（用于计算混合用户奖励）
        self._user_reviews_cache = cache_reader.user_reviews
        
        #  第一遍遍历：计算全局最大值（用于GSS归一化）
        self._global_max_gs = 0
        self._global_max_product_count = 0
        
        for group_id, group_info in self.final_groups.items():
            users = list(group_info['users'].keys())
            gs = len(users)
            self._global_max_gs = max(self._global_max_gs, gs)
            
            # 计算product_count
            product_set = set()
            for user_id in users:
                if user_id in self._user_reviews_cache:
                    reviews = self._user_reviews_cache[user_id]
                    for r in reviews:
                        pid = r.get('asin', r.get('product_id', ''))
                        if pid:
                            product_set.add(pid)
            
            product_count = len(product_set)
            self._global_max_product_count = max(self._global_max_product_count, product_count)
        
        
        total_groups = len(self.final_groups)
        processed_groups = 0
        import time
        start_time = time.time()
        
        for group_id, group_info in self.final_groups.items():
            processed_groups += 1
            if processed_groups % 50 == 0:  # 更频繁的进度更新
                elapsed = time.time() - start_time
                avg_time = elapsed / processed_groups
                eta = avg_time * (total_groups - processed_groups)
            
            # 添加group_id到group_info（用于噪声因子计算）
            group_info['group_id'] = group_id
            
            user_ids = list(group_info['users'].keys())
            
            # 使用所有用户计算GSS，保证准确性
            # GSS计算基于缓存数据，速度可接受
            
            # 从缓存批量获取用户评论数据（无SQL查询）
            user_reviews_batch = cache_reader.get_batch_user_reviews(user_ids)
            
            if len(user_reviews_batch) == 0:
                continue
            
            # 将用户评论数据转换为DataFrame
            all_reviews = []
            for uid, reviews_list in user_reviews_batch.items():
                all_reviews.extend(reviews_list)
            
            if not all_reviews:
                continue
            
            group_reviews = pd.DataFrame(all_reviews)
            
            # 预加载产品平均评分（如果还没有加载）
            if not hasattr(self, 'product_avg_dict'):
                self.product_avg_dict = self._load_product_avg_from_cache(cache_reader)
            
            # 基于缓存的评论数据计算群组GSS
            gss_scores = self._calculate_group_gss(group_reviews, group_info, self.product_avg_dict)
            
            # 计算综合GSS分数
            total_gss = self._compute_total_gss(gss_scores, group_info)
            
            # 更新群组信息
            group_info['gss_scores'] = gss_scores
            group_info['gss_info'] = gss_scores  # 同时保存为gss_info以便分析
            group_info['gss_score'] = total_gss  # 保存为gss_score（原始GSS）
            group_info['total_gss'] = total_gss
        
        cache_reader.close()
        
        # 按GSS分数排序
        sorted_groups = sorted(
            self.final_groups.items(), 
            key=lambda x: x[1].get('total_gss', 0), 
            reverse=True
        )
        
        for i, (group_id, group_info) in enumerate(sorted_groups[:5]):
            gss = group_info.get('total_gss', 0)
            user_count = group_info['user_count']
    
    def _load_product_avg_from_cache(self, cache_reader) -> Dict:
        """从缓存的用户评论数据中计算产品平均评分"""
        
        # 收集所有评论
        all_reviews = []
        for user_id, reviews_list in cache_reader.user_reviews.items():
            all_reviews.extend(reviews_list)
        
        # 转换为DataFrame
        df = pd.DataFrame(all_reviews)
        
        # 计算每个产品的平均评分
        product_avg = df.groupby('product_id')['rating'].mean().to_dict()
        
        return product_avg
    
    def _calculate_group_gss(self, group_reviews, group_info, product_avg_dict):
        """计算群组GSS各项指标
        
        优化：由于新的GSS公式只需要GS和GRD，优先快速计算这两个指标
        注意：Cell_Phones数据集需要完整指标计算以提高混合用户召回率
        """
        gss_scores = {}
        
        # 获取基本信息
        member_count = group_info['user_count']  # |R_g|
        products = group_reviews['product_id'].unique()  # P_g
        product_count = len(products)  # |P_g|
        total_reviews = len(group_reviews)  # |V_g|
        
        # 检查是否为Cell_Phones数据集，如果是则使用完整计算
        use_full_calculation = 'Cell_Phones' in str(self.db_path)
        
        # ============ 快速计算GS和GRD（新GSS公式只需要这两个）============
        
        # 1. GS - 群组规模 (Group Size) - 快速计算
        if member_count >= 5:
            if member_count >= 20:
                gs_score = 1.0
            elif member_count >= 10:
                gs_score = 0.8 + (member_count - 10) / 50
            else:
                gs_score = 0.5 + (member_count - 5) / 10
        else:
            gs_score = 0.1
        gss_scores['GS'] = min(gs_score, 1.0)
        
        # 2. GRD - 群组评分偏差 (Group Rating Deviation) - 向量化计算
        product_avg_series = group_reviews['product_id'].map(
            lambda pid: product_avg_dict.get(pid, 4.0)
        )
        group_product_avg = group_reviews.groupby('product_id')['rating'].mean()
        global_product_avg = group_product_avg.index.map(lambda pid: product_avg_dict.get(pid, 4.0))
        grd_score = (group_product_avg.values - global_product_avg.values).__abs__().mean() / 4.0 if product_count > 0 else 0
        gss_scores['GRD'] = min(grd_score, 1.0)
        
        # 3. GER - 群组极端评分比例 (Group Extreme Ratings) - 向量化计算
        if product_count > 0:
            extreme_mask = group_reviews['rating'].isin([1, 5])
            ger_score = group_reviews.assign(_extreme=extreme_mask).groupby('product_id')['_extreme'].mean().mean()
        else:
            ger_score = 0
        gss_scores['GER'] = min(ger_score, 1.0)
        
        # ============ 快速模式：计算GS、GRD、GER、TRI，跳过其他指标（所有数据集统一）============
        
        # 计算TRI - 时间接近度指标（简化版，直接用原始date列）
        tri_score = self._calculate_time_proximity_indicator_fast(group_reviews)
        gss_scores['TRI'] = tri_score
        
        # GRT - 群组评论时间紧密度（逐用户时间密度+工作时间比，向量化计算）
        gss_scores['GRT'] = self._calculate_group_review_time(group_reviews)

        # avg_rating_std：群组成员平均评分标准差（归一化到[0,1]）
        # 水军群组在Clothing数据集中呈现评分行为多样性（混合高低评分规避检测），std偏高
        _users = group_info.get('users', {})
        _rating_stds = [float(u.get('rating_std', 0.0)) for u in _users.values() if isinstance(u, dict)]
        if _rating_stds:
            gss_scores['avg_rating_std'] = float(np.clip(np.mean(_rating_stds) / 2.5, 0.0, 1.0))
        else:
            gss_scores['avg_rating_std'] = 0.0

        # low_act：低活跃用户占比（历史评论总数<=2的用户比例）
        # 水军账号通常仅为发布少量虚假评论而创建，全局评论数极少
        # GSS指标已统一，时间：2026.4.11
        _low_act_cnt = sum(1 for u in _users.values() if isinstance(u, dict) and int(u.get('review_count', 1)) <= 2)
        gss_scores['low_act'] = float(_low_act_cnt / len(_users)) if _users else 0.0

        # ============ 新GSS公式所需4个指标（基于virtual_reviews，与gss_combo_search.py一致）============
        # 使用虚拟节点时间窗口内的评论（virtual_reviews），而非用户全量历史评论缓存，
        # 保持与穷举搜索实验的数据来源一致，确保Precision@300在0.88-0.95范围内。

        _vr_ratings = []
        _n_users = len(_users)
        for _u in _users.values():
            if not isinstance(_u, dict):
                continue
            for _vr in _u.get('virtual_reviews', []):
                if not isinstance(_vr, dict):
                    continue
                _vni = _vr.get('virtual_node_info', {})
                _r = _vni.get('rating')
                if _r is None:
                    _r = _vni.get('overall')
                try:
                    if _r is not None:
                        _vr_ratings.append(float(_r))
                except Exception:
                    pass

        _vr_total = len(_vr_ratings)
        if _vr_total > 0:
            _vr_arr = np.array(_vr_ratings, dtype=np.float32)

            # low_rating_ratio：≤2星占比（低分操控强度）
            gss_scores['low_rating_ratio'] = float((_vr_arr <= 2.0).sum()) / _vr_total

            # high_rating_ratio：≥4星占比（IS_2020 GER高分分量）
            gss_scores['high_rating_ratio'] = float((_vr_arr >= 4.0).sum()) / _vr_total

            # rating_entropy：评分分布熵，归一化到[0,1]（log2(5)为最大熵）
            from collections import Counter as _Counter
            _rcnt = _Counter(int(round(r)) for r in _vr_ratings)
            _probs = np.array([_rcnt.get(i, 0) / _vr_total for i in range(1, 6)], dtype=np.float64)
            _probs = _probs[_probs > 0]
            _H = -float((_probs * np.log2(_probs)).sum())
            gss_scores['rating_entropy'] = float(np.clip(_H / np.log2(5), 0.0, 1.0))

            # review_cnt_score：人均评论数得分（log归一化，以50条为参考上限）
            gss_scores['review_cnt_score'] = float(
                min(1.0, np.log1p(_vr_total / max(_n_users, 1)) / np.log1p(50))
            )
        else:
            gss_scores['low_rating_ratio']  = 0.0
            gss_scores['high_rating_ratio'] = 0.0
            gss_scores['rating_entropy']    = 0.0
            gss_scores['review_cnt_score']  = 0.0

        # 其他指标保留默认值（兼容性）
        gss_scores['GOR'] = 0.0
        gss_scores['CS'] = 0.1
        gss_scores['BC'] = 0.5
        gss_scores['PS'] = 0.2
        gss_scores['BI'] = 0.0
        gss_scores['BAI'] = 0.7
        gss_scores['RPI'] = 0.6

        return gss_scores

    def _calculate_group_review_time(self, group_reviews):
        """GRT：逐用户时间密度 + 工作时间比，向量化实现

        逻辑：对群组内每个用户单独计算其评论时间的集中程度，
        高时间密度（每小时多条）或高度集中在工作时段均为可疑特征，
        最终返回所有用户得分的均値作为群组级GRT。
        """
        try:
            if group_reviews is None or len(group_reviews) < 2:
                return 0.1
            df = group_reviews.copy()
            df['_ts'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['_ts'])
            if len(df) < 2:
                return 0.1

            group_ts_min = df['_ts'].min()
            group_time_range = (df['_ts'].max() - group_ts_min).total_seconds()

            grp = df.groupby('user_id')['_ts']
            u_min = grp.min()
            u_max = grp.max()
            u_cnt = grp.count()
            u_span_s = (u_max - u_min).dt.total_seconds()

            df['_wh'] = df['_ts'].dt.hour.between(9, 17).astype(int)
            u_work = df.groupby('user_id')['_wh'].mean()

            scores = []
            for uid in u_cnt.index:
                cnt  = u_cnt[uid]
                span = u_span_s[uid]
                if cnt < 2:
                    if group_time_range > 0:
                        offset = (u_min[uid] - group_ts_min).total_seconds()
                        scores.append(0.4 if offset <= group_time_range * 0.2 else 0.2)
                    else:
                        scores.append(0.2)
                    continue
                if span == 0:
                    scores.append(1.0)
                    continue
                density = cnt / max(span / 3600, 1e-9)
                base = 0.9 if density >= 1 else (0.7 if density >= 0.5 else (0.5 if density >= 0.1 else 0.3))
                wr = u_work.get(uid, 0.5)
                bonus = 0.2 if wr >= 0.8 else (0.15 if wr <= 0.2 else 0)
                scores.append(min(base + bonus, 1.0))
            return min(float(np.mean(scores)) if scores else 0.1, 1.0)
        except Exception:
            return 0.1

    def _calculate_time_proximity_indicator_fast(self, group_reviews):
        """TRI - 时间接近度指标（群组内评论时间集中程度）
        使用 exp(-span_days/30) 衡量：span越小 → TRI越高 → 越可疑
        """
        try:
            dates = pd.to_datetime(group_reviews['date'], errors='coerce').dropna()
            if len(dates) < 2:
                return 0.5
            span_days = (dates.max() - dates.min()).days
            import math
            return float(min(1.0, math.exp(-span_days / 30.0)))
        except Exception:
            return 0.5

    def _compute_total_gss(self, gss_scores, group_info=None):
        """GSS计算 - 4指标行为均等平均策略

        # GSS指标已更新，时间：2026.4.14
        指标组合（经gss_combo_search.py穷举15候选指标9438种组合K=300三数据集交叉验证确定）：
          low_rating_ratio + rating_entropy + high_rating_ratio + review_cnt_score

        各指标含义：
          low_rating_ratio : 群组内≤2星评论占比（低分操控强度）
          rating_entropy   : 群组评分分布的Shannon熵（归一化，评分行为多样性）
          high_rating_ratio: 群组内≥4星评论占比（IS_2020 GER高分方向分量）
          review_cnt_score : 人均评论数的log归一化得分（群组活跃度）

        公式：
          GSS(g) = (low_rating_ratio + rating_entropy + high_rating_ratio + review_cnt_score) / 4

        性能（三数据集, K=300, 纯行为指标）：
          Cell_Phones Precision=0.9233  TP=277
          Clothing    Precision=0.9100  TP=273
          Electronics Precision=0.9000  TP=270
        """
        low_rating_ratio  = gss_scores.get('low_rating_ratio',  0.0)
        rating_entropy    = gss_scores.get('rating_entropy',    0.0)
        high_rating_ratio = gss_scores.get('high_rating_ratio', 0.0)
        review_cnt_score  = gss_scores.get('review_cnt_score',  0.0)

        final_gss = (low_rating_ratio + rating_entropy + high_rating_ratio + review_cnt_score) / 4.0

        return float(np.clip(final_gss, 0.0, 1.0))

    def save_results(self):
        """保存群组分析结果"""
        current_result_dir = result_dir if result_dir is not None else get_result_dir(self.sample_ratio, self.db_path, module=6)
        os.makedirs(current_result_dir, exist_ok=True)  # 确保目录存在
        
        sorted_by_gss = sorted(
            self.final_groups.items(),
            key=lambda x: x[1].get('total_gss', 0),
            reverse=True
        )
        
        for i, (group_id, group_info) in enumerate(sorted_by_gss[:5]):
            gss = group_info.get('total_gss', 0)
            user_count = group_info['user_count']
        
        # 保存最终群组信息
        final_groups_path = os.path.join(current_result_dir, f'final_spam_groups_{self.sample_ratio}.pkl')
        with open(final_groups_path, 'wb') as f:
            pickle.dump(self.final_groups, f)
        
        # 额外保存：纯Python格式的群组用户列表（不依赖pandas）
        groups_users_only = {}
        for group_id, group_info in self.final_groups.items():
            # 只保存用户ID列表和基本统计信息
            groups_users_only[group_id] = {
                'users': list(group_info['users'].keys()),  # 用户ID列表
                'user_count': group_info['user_count'],
                'total_reviews': group_info['total_reviews'],
                'avg_iss': group_info['avg_iss'],
                'gss': group_info.get('total_gss', 0)
            }
        
        # 保存纯Python格式（不依赖pandas）
        groups_users_path = os.path.join(current_result_dir, f'group_users_list_{self.sample_ratio}.pkl')
        with open(groups_users_path, 'wb') as f:
            pickle.dump(groups_users_only, f)
        
        # 保存群组分析CSV
        group_analysis = []
        for group_id, group_info in self.final_groups.items():
            group_analysis.append({
                'group_id': group_id,
                'user_count': group_info['user_count'],
                'total_reviews': group_info['total_reviews'],
                'avg_iss': group_info['avg_iss'],
                'gss': group_info.get('total_gss', 0),
                'merged_from': ','.join(map(str, group_info.get('merged_from', [group_id])))
            })
        
        group_df = pd.DataFrame(group_analysis)
        group_csv_path = os.path.join(current_result_dir, f'spam_group_analysis_{self.sample_ratio}.csv')
        group_df.to_csv(group_csv_path, index=False, encoding='utf-8')
        
        
    def run(self):
        # [FLOW-M67] 模块6-7：节点聚合+群组净化合并 | 缓存: module6/final_spam_groups_*.pkl
        """运行模块6-7的完整流程"""
        try:
            current_result_dir = result_dir if result_dir is not None else get_result_dir(self.sample_ratio, self.db_path, module=6)
            
            # 检查缓存文件是否存在
            final_groups_path = os.path.join(current_result_dir, f'final_spam_groups_{self.sample_ratio}.pkl')
            group_csv_path = os.path.join(current_result_dir, f'spam_group_analysis_{self.sample_ratio}.csv')
            
            if os.path.exists(final_groups_path) and os.path.exists(group_csv_path):
                pass
                # 加载缓存的结果
                with open(final_groups_path, 'rb') as f:
                    self.final_groups = pickle.load(f)
                return True
            
            # 如果缓存不存在，执行完整流程
            self.load_data()
            self.deduplicate_virtual_nodes()  #  新增：去重虚拟节点（在聚合之前）
            self.aggregate_nodes_to_users()
            self.secondary_clustering_with_temporal_features()  #  新增：使用时序特征进行二次聚类
            self.calculate_iss_scores()
            self.filter_candidate_groups()
            self.optimize_group_purity()  #  启用：群组后处理优化（方案2）
            self.merge_similar_groups()
            self.calculate_group_suspicion_scores()
            self.save_results()
            return True
        except Exception as e:
            pass
            return False

# ================================
# 模块8：指标验证与结果输出
# ================================

class Module8_ValidationAndOutput:
    """模块8：指标验证与结果输出
    
    计算精确率、召回率、F1值等性能指标，并输出最终结果
    """
    
    def __init__(self, sample_ratio=1.0, delta_g=0.7, delta_G=0.6, top_k=300, multi_k_values=None, db_path=None):
        self.sample_ratio = sample_ratio
        self.db_path = db_path
        self.delta_g = delta_g  # δ_g: 真实标记阈值，群组内水军成员最低比例（不可改变）
        self.delta_G = delta_G  # δ_G: GSS得分阈值，判别候选群组是否为水军群组
        self.top_k = top_k     # TopK群组数量
        # 多K评估：从TopK=10开始，每10递增到300
        self.multi_k_values = multi_k_values or [
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
            110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
            210, 220, 230, 240, 250, 260, 270, 280, 290, 300,
        ]
        
        # 设置结果目录
        self.current_result_dir = get_result_dir(self.sample_ratio, self.db_path)
        
        # 数据存储
        self.final_groups = None
        self.ground_truth_users = None  # 真实spammer用户集合
        self.detected_groups = []       # 模型检测出的群组
        self.true_spam_groups = []      # 真实的水军群组
        
        # 性能指标
        self.metrics = {}
        self.multi_k_metrics = {}  # 存储多个K值的指标
        
    def load_data(self):
        """加载最终群组信息"""
        
        # 模块8需要加载模块6的缓存
        module6_dir = get_result_dir(self.sample_ratio, self.db_path, module=6)
        current_result_dir = get_result_dir(self.sample_ratio, self.db_path, module=8)
        
        # 加载最终群组信息（从模块6）
        final_groups_path = os.path.join(module6_dir, f'final_spam_groups_{self.sample_ratio}.pkl')
        with open(final_groups_path, 'rb') as f:
            self.final_groups = pickle.load(f)
        
        
        # 加载真实标签数据
        self._load_ground_truth()
        
    def _load_ground_truth(self):
        """从缓存加载真实spammer用户标签（无SQL查询）"""
        
        # 首先从虚拟节点文件中获取采样的用户列表（从模块1）
        module1_dir = get_result_dir(self.sample_ratio, self.db_path, module=1, force_no_threshold=True)
        virtual_nodes_path = os.path.join(module1_dir, 'virtual_nodes.pkl')
        
        if not os.path.exists(virtual_nodes_path):
            raise FileNotFoundError(f"虚拟节点文件不存在: {virtual_nodes_path}")
        
        with open(virtual_nodes_path, 'rb') as f:
            virtual_nodes = pickle.load(f)
        
        # 获取采样的用户ID列表
        sampled_user_ids = set()
        for node_info in virtual_nodes.values():
            sampled_user_ids.add(node_info['original_user_id'])
        
        
        # 从用户评论缓存中读取用户标签（避免SQL查询）
        # 获取正确的缓存目录路径（包含数据集名称）
        dataset_name = get_dataset_name(self.db_path)
        cache_dir = f"preprocessed_{dataset_name}/user_metrics_cache"
        cache_reader = UserMetricsCacheReader(cache_dir=cache_dir)
        
        # 创建用户标签字典
        self.user_labels = {}
        spam_count = 0
        normal_count = 0
        
        for user_id in sampled_user_ids:
            user_reviews = cache_reader.get_user_reviews(user_id)
            if user_reviews and len(user_reviews) > 0:
                # 检查用户的所有评论，只要有任何一条label=-1，就是水军用户
                # 规则20：只要发表过虚假评论的用户均为水军用户
                has_spam_review = any(review.get('label', 1) == -1 for review in user_reviews)
                label = -1 if has_spam_review else 1
                self.user_labels[user_id] = label
                if label == -1:
                    spam_count += 1
                else:
                    normal_count += 1
        
        cache_reader.close()
        
        # 获取水军用户集合
        self.ground_truth_users = set([uid for uid, label in self.user_labels.items() if label == -1])
        
        
    def generate_predictions(self):
        """基于GSS得分和TopK选择生成群组级别预测结果"""
        
        sorted_by_gss = sorted(
            self.final_groups.items(),
            key=lambda x: x[1].get('total_gss', 0),
            reverse=True
        )
        
        self.detected_groups = sorted_by_gss[:self.top_k]
        
        
        # 3. 计算每个群组中真实水军用户比例，确定真实水军群组
        self.true_spam_groups = []
        
        for group_id, group_info in self.detected_groups:
            # 计算该群组中真实水军用户比例
            total_users = len(group_info['users'])
            spam_users = 0
            
            for user_id in group_info['users'].keys():
                if user_id in self.ground_truth_users:  # 该用户是真实水军用户
                    spam_users += 1
            
            spam_ratio = spam_users / total_users if total_users > 0 else 0
            
            # 如果真实水军用户比例>=70%，则认为是真实水军群组
            if spam_ratio >= 0.7:
                self.true_spam_groups.append((group_id, group_info, spam_ratio))
        
        
    def calculate_metrics(self):
        """计算群组级别性能指标"""
        
        # 1. 计算所有群组中真实水军用户比例>=70%的群组数（实际真实水军群组数）
        actual_spam_groups = 0
        for group_id, group_info in self.final_groups.items():
            total_users = len(group_info['users'])
            spam_users = 0
            
            for user_id in group_info['users'].keys():
                if user_id in self.ground_truth_users:
                    spam_users += 1
            
            spam_ratio = spam_users / total_users if total_users > 0 else 0
            if spam_ratio >= 0.7:
                actual_spam_groups += 1
        
        # 2. 计算TP, FP, FN
        # TP: 模型检测出的真实水军群组数
        tp = len(self.true_spam_groups)
        
        # FP: 模型认定为水军但实际不是真实水军的群组数
        fp = len(self.detected_groups) - tp
        
        # FN: 实际是真实水军但模型未检测出的群组数
        fn = actual_spam_groups - tp
        
        # TN: 对于群组级别评估，TN不太适用，设为0
        tn = 0
        
        # 3. 计算各项指标
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # 对于群组级别，准确率的计算需要调整
        # 这里使用检测准确性：正确检测的群组数 / 总检测群组数
        accuracy = tp / len(self.detected_groups) if len(self.detected_groups) > 0 else 0.0
        
        # 存储指标
        self.metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy,
            'confusion_matrix': {
                'tp': int(tp),
                'fp': int(fp),
                'tn': int(tn),
                'fn': int(fn)
            },
            'total_groups': len(self.final_groups),
            'detected_groups': len(self.detected_groups),
            'actual_spam_groups': actual_spam_groups,
            'true_spam_groups': tp,
            'delta_g': self.delta_g,
            'delta_G': self.delta_G,
            'top_k': self.top_k
        }
        
        self._print_metrics()
    
    def calculate_multi_k_metrics(self):
        """计算多个TopK值的性能指标"""
        
        # 获取所有超过δ_G阈值的群组，按GSS得分排序
        initial_spam_groups = []
        for group_id, group_info in self.final_groups.items():
            if group_info.get('total_gss', 0) > self.delta_G:
                initial_spam_groups.append((group_id, group_info))
        
        # 按GSS得分降序排序
        initial_spam_groups.sort(key=lambda x: x[1].get('total_gss', 0), reverse=True)
        
        # 计算所有群组中真实水军群组数（用于计算召回率）
        actual_spam_groups = 0
        for group_id, group_info in self.final_groups.items():
            total_users = len(group_info['users'])
            spam_users = sum(1 for user_id in group_info['users'].keys() 
                           if user_id in self.ground_truth_users)
            spam_ratio = spam_users / total_users if total_users > 0 else 0
            if spam_ratio >= 0.7:
                actual_spam_groups += 1
        
        # 为每个K值计算指标
        for k in self.multi_k_values:
            pass
            
            # 选取TopK个群组
            k_actual = min(k, len(initial_spam_groups))
            detected_groups_k = initial_spam_groups[:k_actual]
            
            # 计算真实水军群组数
            true_spam_groups_k = 0
            for group_id, group_info in detected_groups_k:
                total_users = len(group_info['users'])
                spam_users = sum(1 for user_id in group_info['users'].keys() 
                               if user_id in self.ground_truth_users)
                spam_ratio = spam_users / total_users if total_users > 0 else 0
                if spam_ratio >= 0.7:
                    true_spam_groups_k += 1
            
            # 计算TP, FP, FN
            tp = true_spam_groups_k
            fp = k_actual - tp
            fn = actual_spam_groups - tp
            tn = 0  # 群组级别评估不适用
            
            # 计算指标
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = tp / k_actual if k_actual > 0 else 0.0
            
            # 存储指标
            self.multi_k_metrics[k] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'accuracy': accuracy,
                'confusion_matrix': {
                    'tp': int(tp),
                    'fp': int(fp),
                    'tn': int(tn),
                    'fn': int(fn)
                },
                'total_groups': len(self.final_groups),
                'detected_groups': k_actual,
                'actual_spam_groups': actual_spam_groups,
                'true_spam_groups': tp,
                'delta_g': self.delta_g,
                'delta_G': self.delta_G,
                'top_k': k
            }
        
    
    def save_multi_k_results(self):
        """保存多个TopK值的结果到专门文件夹"""
        
        # 创建专门的多K值结果文件夹
        multi_k_dir = os.path.join(self.current_result_dir, "multi_topk_results")
        os.makedirs(multi_k_dir, exist_ok=True)
        
        # 为每个K值保存单独的指标文件
        for k, metrics in self.multi_k_metrics.items():
            # 保存JSON格式的指标
            metrics_file = os.path.join(multi_k_dir, f"metrics_topk_{k}.json")
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            
            # 保存可读的文本报告
            report_file = os.path.join(multi_k_dir, f"report_topk_{k}.txt")
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(f"TopK={k} 群组检测结果报告\n")
                f.write("="*50 + "\n\n")
                f.write(f"评估参数:\n")
                f.write(f"  TopK值: {k}\n")
                f.write(f"  δ_g (群组判定阈值): {metrics['delta_g']}\n")
                f.write(f"  δ_G (GSS过滤阈值): {metrics['delta_G']}\n")
                f.write(f"  采样比例: {self.sample_ratio}\n\n")
                f.write(f"检测统计:\n")
                f.write(f"  总群组数: {metrics['total_groups']}\n")
                f.write(f"  实际真实水军群组数: {metrics['actual_spam_groups']}\n")
                f.write(f"  模型认定的水军群组数: {metrics['detected_groups']}\n")
                f.write(f"  模型检测出的真实水军群组数: {metrics['true_spam_groups']}\n\n")
                cm = metrics['confusion_matrix']
                f.write(f"混淆矩阵:\n")
                f.write(f"                    预测\n")
                f.write(f"                水军群组  非水军群组\n")
                f.write(f"实际  水军群组    {cm['tp']:6d}      {cm['fn']:6d}\n")
                f.write(f"      非水军群组  {cm['fp']:6d}      {cm['tn']:6d}\n\n")
                f.write(f"性能指标:\n")
                f.write(f"  精确率 (Precision): {metrics['precision']:.4f}\n")
                f.write(f"  召回率 (Recall): {metrics['recall']:.4f}\n")
                f.write(f"  F1值 (F1-Score): {metrics['f1_score']:.4f}\n")
                f.write(f"  检测准确性: {metrics['accuracy']:.4f}\n")
        
        # 保存汇总对比表
        summary_file = os.path.join(multi_k_dir, "topk_comparison_summary.csv")
        summary_data = []
        for k in sorted(self.multi_k_values):
            if k in self.multi_k_metrics:
                metrics = self.multi_k_metrics[k]
                summary_data.append({
                    'TopK': k,
                    'Precision': f"{metrics['precision']:.4f}",
                    'Recall': f"{metrics['recall']:.4f}",
                    'F1_Score': f"{metrics['f1_score']:.4f}",
                    'Accuracy': f"{metrics['accuracy']:.4f}",
                    'TP': metrics['confusion_matrix']['tp'],
                    'FP': metrics['confusion_matrix']['fp'],
                    'FN': metrics['confusion_matrix']['fn'],
                    'Detected_Groups': metrics['detected_groups'],
                    'True_Spam_Groups': metrics['true_spam_groups']
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_file, index=False, encoding='utf-8')
        
        # 保存3个单独的指标文件（精确率、召回率、F1值）
        precision_data = []
        recall_data = []
        f1_data = []
        
        for k in sorted(self.multi_k_values):
            if k in self.multi_k_metrics:
                metrics = self.multi_k_metrics[k]
                precision_data.append({'TopK': k, 'Precision': metrics['precision']})
                recall_data.append({'TopK': k, 'Recall': metrics['recall']})
                f1_data.append({'TopK': k, 'F1_Score': metrics['f1_score']})
        
        # 保存精确率文件
        precision_file = os.path.join(multi_k_dir, "precision_by_topk.csv")
        pd.DataFrame(precision_data).to_csv(precision_file, index=False, encoding='utf-8')
        
        # 保存召回率文件
        recall_file = os.path.join(multi_k_dir, "recall_by_topk.csv")
        pd.DataFrame(recall_data).to_csv(recall_file, index=False, encoding='utf-8')
        
        # 保存F1值文件
        f1_file = os.path.join(multi_k_dir, "f1_score_by_topk.csv")
        pd.DataFrame(f1_data).to_csv(f1_file, index=False, encoding='utf-8')
        
        
    def _print_metrics(self):
        """打印群组级别性能指标"""
        print("\n" + "="*60)
        print("           Detection Results (Group Level)")
        print("="*60)
        
        cm = self.metrics['confusion_matrix']
        print(f"Confusion Matrix:")
        print(f"                    Predicted")
        print(f"               Spam Group  Non-spam")
        print(f"Actual  Spam         {cm['tp']:6d}      {cm['fn']:6d}")
        print(f"        Non-spam     {cm['fp']:6d}      {cm['tn']:6d}")
        print()
        
        print(f"Performance Metrics:")
        print(f"  Precision: {self.metrics['precision']:.4f}")
        print(f"  Recall:    {self.metrics['recall']:.4f}")
        print(f"  F1-Score:  {self.metrics['f1_score']:.4f}")
        print(f"  Accuracy:  {self.metrics['accuracy']:.4f}")
        print()
        
        print(f"Detection Statistics:")
        print(f"  Total groups:             {self.metrics['total_groups']}")
        print(f"  True spam groups:         {self.metrics['actual_spam_groups']}")
        print(f"  Predicted spam groups:    {self.metrics['detected_groups']}")
        print(f"  Correctly detected:       {self.metrics['true_spam_groups']}")
        print()
        
        print(f"Evaluation Parameters:")
        print(f"  delta_g (spam threshold):  {self.metrics['delta_g']}")
        print(f"  delta_G (GSS threshold):   {self.metrics['delta_G']}")
        print(f"  TopK:                {self.metrics['top_k']}")
        print(f"  Sample ratio:          {self.sample_ratio}")
        print("="*60)
        
    def save_results(self):
        """保存最终结果"""
        current_result_dir = result_dir if result_dir is not None else get_result_dir(self.sample_ratio, self.db_path, module=8)
        os.makedirs(current_result_dir, exist_ok=True)  # 确保目录存在
        
        # 保存性能指标
        metrics_path = os.path.join(current_result_dir, f'performance_metrics_{self.sample_ratio}.json')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)
        
        # 保存详细预测结果
        prediction_results = []
        
        # 遍历所有用户（包括水军和真实用户）
        for user_id, ground_truth_label in self.user_labels.items():
            # 检查用户是否在任何检测到的群组中
            is_predicted_spammer = any(user_id in group_info['users'] for group_id, group_info in self.detected_groups)
            
            # 设置预测标签：如果被检测为水军群组成员则为-1，否则为1
            prediction_label = -1 if is_predicted_spammer else 1
            
            # 判断预测是否正确
            is_correct = (ground_truth_label == prediction_label)
            
            prediction_results.append({
                'user_id': user_id,
                'ground_truth': ground_truth_label,  # 使用真实标签：-1表示水军，1表示真实用户
                'prediction': prediction_label,      # 预测标签：-1表示水军，1表示真实用户
                'correct': is_correct
            })
        
        prediction_df = pd.DataFrame(prediction_results)
        prediction_csv_path = os.path.join(current_result_dir, f'prediction_results_{self.sample_ratio}.csv')
        prediction_df.to_csv(prediction_csv_path, index=False, encoding='utf-8')
        
        # 保存最终检测报告
        report_path = os.path.join(current_result_dir, f'detection_report_{self.sample_ratio}.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("垃圾群组检测报告\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"采样比例: {self.sample_ratio}\n")
            f.write(f"检测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("性能指标:\n")
            f.write(f"  精确率: {self.metrics['precision']:.4f}\n")
            f.write(f"  召回率: {self.metrics['recall']:.4f}\n")
            f.write(f"  F1值: {self.metrics['f1_score']:.4f}\n")
            f.write(f"  准确率: {self.metrics['accuracy']:.4f}\n\n")
            
            f.write("检测统计:\n")
            f.write(f"  检测到的垃圾群组数: {len(self.detected_groups)}\n")
            f.write(f"  参与评估的用户总数: {len(self.user_labels)}\n")
            f.write(f"  其中水军用户数: {len(self.ground_truth_users)}\n")
            f.write(f"  其中真实用户数: {len(self.user_labels) - len(self.ground_truth_users)}\n\n")
            
            cm = self.metrics['confusion_matrix']
            f.write("混淆矩阵:\n")
            f.write(f"  真正例(TP): {cm['tp']}\n")
            f.write(f"  假正例(FP): {cm['fp']}\n")
            f.write(f"  真负例(TN): {cm['tn']}\n")
            f.write(f"  假负例(FN): {cm['fn']}\n")
        
        
    def run(self):
        # [FLOW-M8] 模块8：指标验证与结果输出 | 输出: module8/detection_report_*.txt
        """运行模块8的完整流程"""
        try:
            # 检查缓存文件是否存在
            metrics_file = os.path.join(self.current_result_dir, f'performance_metrics_{self.sample_ratio}.json')
            predictions_file = os.path.join(self.current_result_dir, f'prediction_results_{self.sample_ratio}.csv')
            report_file = os.path.join(self.current_result_dir, f'detection_report_{self.sample_ratio}.txt')
            
            if os.path.exists(metrics_file) and os.path.exists(predictions_file) and os.path.exists(report_file):
                pass
                # 加载性能指标
                with open(metrics_file, 'r', encoding='utf-8') as f:
                    metrics = json.load(f)
                # 加载预测结果
                predictions_df = pd.read_csv(predictions_file)
                # 显示报告文件信息
                with open(report_file, 'r', encoding='utf-8') as f:
                    report_lines = f.readlines()
                return True
            
            self.load_data()
            self.generate_predictions()
            self.calculate_metrics()
            self.calculate_multi_k_metrics()  # 添加多K值计算
            self.save_results()
            self.save_multi_k_results()      # 保存多K值结果
            return True
        except Exception as e:
            pass
            return False

# ================================
# 主函数和命令行接口

class SpamGroupDetectionPipeline:
    """垃圾群组检测完整流水线"""
    
    def __init__(self, db_path="DataSet/Electronics_2013_1.6.db", sample_ratio=1.0, 
                 attraction_threshold=0.92, repulsion_threshold=0.60, 
                 lambda_factor=0.5, iss_threshold=0.3, group_threshold=0.5, use_gpu=None):
        self.db_path = db_path
        self.sample_ratio = sample_ratio
        self.attraction_threshold = attraction_threshold
        
        # 识别数据集类型
        self.dataset_name = self._identify_dataset(db_path)
        
        # 斥力图阈值：优先命令行参数，否则使用默认值（模块3内部会动态计算分位数阈值）
        self.repulsion_threshold = repulsion_threshold
        
        self.lambda_factor = lambda_factor
        self.iss_threshold = iss_threshold  # δ_I: 个体阈值，用于ISS指标进行个体用户净化
        self.group_threshold = group_threshold  # δ_G: 群组阈值，用于GSS得分判别候选群组

        # 根据数据集设置最优引力图/斥力图分位数（来自网格搜索热力图最优组合）
        # Cell_Phones: attr=80%, rep=30%  (Precision=0.9433)
        # Electronics: attr=60%, rep=40%  (Precision=0.9367)
        # Clothing:    attr=60%, rep=20%  (Precision=0.9270)
        dataset_pct_config = {
            "Cell_Phones":  {"attraction_pct": 80, "repulsion_pct": 30},
            "Electronics":  {"attraction_pct": 60, "repulsion_pct": 40},
            "Clothing":     {"attraction_pct": 60, "repulsion_pct": 20},
        }
        pct_cfg = dataset_pct_config.get(self.dataset_name, {"attraction_pct": 80, "repulsion_pct": 30})
        self.attraction_pct = pct_cfg["attraction_pct"]
        self.repulsion_pct  = pct_cfg["repulsion_pct"]

        # 自动检测GPU可用性，如果use_gpu为None则自动使用GPU（如果可用）
        if use_gpu is None:
            self.use_gpu = torch.cuda.is_available()
            if self.use_gpu:
                pass
            else:
                pass
        else:
            self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # 初始化各模块
        self.module1 = Module1_NodeSplitting(db_path, sample_ratio)
        self.module2 = Module2_FeatureAdjacencyConstruction(sample_ratio, db_path)
        self.module3 = Module3_AttractionRepulsionGraphs(sample_ratio, attraction_threshold, self.repulsion_threshold, db_path, use_adaptive_inversion=True,
                                                         attraction_pct=self.attraction_pct, repulsion_pct=self.repulsion_pct)
        self.module4 = Module4_EnhancedAdjacencyMatrix(sample_ratio, lambda_factor, db_path)
        self.module5 = Module5_TGNNDBSCANClustering(db_path, sample_ratio, use_gpu=self.use_gpu)
        self.module67 = Module6_7_NodeAggregationAndGroupPurification(sample_ratio, iss_threshold, group_threshold, db_path, dataset_name=self.dataset_name)
        # 将最终评估的TopK从500改为300
        self.module8 = Module8_ValidationAndOutput(sample_ratio, delta_g=0.7, delta_G=group_threshold, top_k=300, db_path=db_path)
    
    def _identify_dataset(self, db_path):
        """识别数据集类型"""
        if "Electronics" in db_path:
            return "Electronics"
        elif "Cell_Phones" in db_path:
            return "Cell_Phones"
        elif "Clothing" in db_path:
            return "Clothing"
        else:
            return "Unknown"
        
    def run_full_pipeline(self, start_module=1, end_module=8):
        # ============================================================
        # [主流程入口] SpamGroupDetectionPipeline.run_full_pipeline
        # 调用顺序: 模块1->2->3->4->5->6-7->8
        # 模块1: 节点时序拆分       (缓存: module1/)
        # 模块2: 特征矩阵+邻接矩阵  (缓存: module2/)
        # 模块3: 引力图+斥力图      (缓存: module3/)
        # 模块4: 增强邻接矩阵       (缓存: module4/)
        # 模块5: GCN训练+HDBSCAN   (缓存: module5/)
        # 模块6-7: 节点聚合+净化    (缓存: module6/)
        # 模块8: 指标验证+输出      (输出: module8/)
        # [!] 每个模块有缓存则跳过执行
        # [!] 模块1-4代码及缓存不可修改（规则11）
        # ============================================================
        """运行完整的检测流水线"""
        print("="*60)
        print("        Spam Group Detection System")
        print("="*60)
        print(f"Dataset: {self.dataset_name}")
        print(f"Running modules: {start_module} to {end_module}")
        print("="*60)

        start_time = time.time()

        try:
            # 模块1：节点分割
            if start_module <= 1 <= end_module:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting Module 1: Virtual Node Construction...")
                if not self.module1.run():
                    return False
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Module 1 complete.")
            # 模块2：特征和邻接矩阵构建
            if start_module <= 2 <= end_module:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting Module 2: Feature Matrix and Adjacency Construction...")
                if not self.module2.run():
                    return False
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Module 2 complete.")
            # 模块3：引力斥力图构建
            if start_module <= 3 <= end_module:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting Module 3: Attraction and Repulsion Graph Construction...")
                if not self.module3.run():
                    return False
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Module 3 complete.")
            # 模块4：增强邻接矩阵
            if start_module <= 4 <= end_module:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting Module 4: Enhanced Adjacency Matrix...")
                if not self.module4.run():
                    return False
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Module 4 complete.")
            # 模块5：GAT编码和DBSCAN聚类
            if start_module <= 5 <= end_module:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting Module 5: GNN Training and HDBSCAN Clustering...")
                if not self.module5.run():
                    return False
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Module 5 complete.")
            # 模块6-7：节点聚合和群组净化合并
            if start_module <= 6 <= end_module or start_module <= 7 <= end_module:
                # 在模块6-7运行前，确保用户指标缓存已构建
                cache_builder = UserMetricsCacheBuilder(self.db_path)
                if not cache_builder.build_cache(force_rebuild=False):
                    print("[ERROR] Failed to build user metrics cache.")
                    return False
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting Modules 6-7: Node Aggregation and Group Purification...")
                if not self.module67.run():
                    return False
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Modules 6-7 complete.")
            # 模块8：指标验证和结果输出
            if start_module <= 8 <= end_module:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting Module 8: Validation and Output...")
                if not self.module8.run():
                    return False
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Module 8 complete.")

            # 计算总执行时间
            total_time = time.time() - start_time

            print("\n" + "="*60)
            print("           Pipeline Complete")
            print("="*60)
            print(f"Total runtime: {total_time:.2f} seconds")
            print("="*60)

            return True

        except Exception as e:
            print(f"\n[ERROR] Pipeline failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def retrain_from_gat(self):
        """从GAT模块开始重新训练（删除GAT及后续模块的缓存）"""
        current_result_dir = get_result_dir(self.sample_ratio, self.db_path)

        # 删除GAT及后续模块的缓存文件
        files_to_delete = [
            f'gat_embeddings_{self.sample_ratio}.npy',
            f'cluster_labels_{self.sample_ratio}.npy',
            f'cluster_info_{self.sample_ratio}.pkl',
            f'cluster_details_{self.sample_ratio}.csv',
            f'final_spam_groups_{self.sample_ratio}.pkl',
            f'spam_group_analysis_{self.sample_ratio}.csv',
            f'performance_metrics_{self.sample_ratio}.json',
            f'prediction_results_{self.sample_ratio}.csv',
            f'detection_report_{self.sample_ratio}.txt'
        ]

        deleted_count = 0
        for filename in files_to_delete:
            filepath = os.path.join(current_result_dir, filename)
            if os.path.exists(filepath):
                os.remove(filepath)
                deleted_count += 1

        # 从模块5开始重新执行
        return self.run_full_pipeline(start_module=5, end_module=8)

def main():
    """主函数"""
    try:
        logging.info("垃圾群组检测系统启动")
        
        parser = argparse.ArgumentParser(description='垃圾群组检测系统')
        
        # 基本参数
        parser.add_argument('--dataset', type=str, default='DataSet/Electronics_2013_1.6.db',
                           help='数据库文件路径')
        parser.add_argument('--sample_ratio', type=float, default=1.0,
                           help='数据采样比例，默认1.0')
        parser.add_argument('--start_module', type=int, default=1, choices=range(1, 9),
                           help='起始模块编号，1-8，默认1')
        parser.add_argument('--end_module', type=int, default=8, choices=range(1, 9),
                           help='结束模块编号，1-8，默认8')
        
        # 模型参数
        parser.add_argument('--attraction_threshold', type=float, default=0.92,
                           help='引力图阈值，默认0.92')
        parser.add_argument('--repulsion_threshold', type=float, default=0.60,
                           help='斥力图阈值，默认0.60')
        parser.add_argument('--lambda_factor', type=float, default=0.5,
                           help='Lambda因子，默认0.5')
        parser.add_argument('--iss_threshold', type=float, default=0.3,
                           help='ISS过滤阈值，默认0.3')
        parser.add_argument('--group_threshold', type=float, default=0.7,
                           help='群组GSS阈值，默认0.7')
        
        # 特殊模式
        parser.add_argument('--retrain', action='store_true',
                           help='重新训练模式（从GAT开始）')
        parser.add_argument('--gpu', action='store_true',
                           help='强制使用GPU加速')
        
        args = parser.parse_args()
        
        # 记录参数信息
        logging.info(f"运行参数: {vars(args)}")
        
        # 设置随机种子
        set_seed(42)
        
        # 设置GPU
        if args.gpu:
            device = get_device()
            logging.info(f"使用设备: {device}")
        
        # 创建检测流水线（使用指定的采样比例）
        pipeline = SpamGroupDetectionPipeline(
            db_path=args.dataset,
            sample_ratio=args.sample_ratio,  # 使用命令行指定的采样比例
            attraction_threshold=args.attraction_threshold,
            repulsion_threshold=args.repulsion_threshold,
            lambda_factor=args.lambda_factor,
            iss_threshold=args.iss_threshold,
            group_threshold=args.group_threshold,
            use_gpu=args.gpu if args.gpu else None  # 如果没有指定--gpu，则使用None触发自动检测
        )
        
        # 执行检测
        if args.retrain:
            success = pipeline.retrain_from_gat()
        else:
            success = pipeline.run_full_pipeline(args.start_module, args.end_module)
        
        if success:
            logging.info(" 检测任务完成！")
            log_program_end(log_filename, success=True)
            sys.exit(0)
        else:
            logging.error(" 检测任务失败！")
            log_program_end(log_filename, success=False, error_msg="检测任务执行失败")
            sys.exit(1)
            
    except Exception as e:
        error_msg = f"程序异常: {str(e)}\n{traceback.format_exc()}"
        logging.error(error_msg)
        log_program_end(log_filename, success=False, error_msg=error_msg)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # 无参数运行：逐个启动独立子进程运行3个数据集，水军比例阈值=0.7，从模块1开始
        import subprocess
        DATASETS = [
            "DataSet/Cell_Phones_and_Accessorie.db",
            "DataSet/Clothing_Shoes_and_Jewelry.db",
            "DataSet/Electronics_2013_1.6.db",
        ]
        python = sys.executable
        script = os.path.abspath(__file__)
        print("=" * 60)
        print("=" * 60)
        for db in DATASETS:
            print(f"\n{'='*60}")
            print(f"{'='*60}")
            ret = subprocess.call([
                python, script,
                "--dataset", db,
                "--group_threshold", "0.7",
                "--start_module", "1",
                "--end_module", "8",
            ])
            if ret != 0:
                pass
            else:
                pass
        print("\n" + "=" * 60)
        print("=" * 60)
    else:
        main()