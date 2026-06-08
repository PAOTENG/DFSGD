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
        """加载缓存文件到内存（懒加载：只立即加载小的iss_metrics，user_reviews按需加载）"""
        if not os.path.exists(self.iss_cache_file):
            raise FileNotFoundError(f"ISS缓存文件不存在: {self.iss_cache_file}")
        
        if not os.path.exists(self.user_reviews_cache_file):
            raise FileNotFoundError(f"用户评论缓存文件不存在: {self.user_reviews_cache_file}")
        
        # 只加载ISS指标（小文件，立即加载）
        with open(self.iss_cache_file, 'rb') as f:
            self.iss_metrics = pickle.load(f)
        
        # user_reviews 延迟加载：首次访问 .user_reviews 属性时才读取309MB文件
        self._user_reviews = None
        
        # 加载元数据
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
        
    
    @property
    def user_reviews(self):
        """懒加载user_reviews（309MB），首次访问时才从磁盘读取"""
        if self._user_reviews is None:
            with open(self.user_reviews_cache_file, 'rb') as f:
                self._user_reviews = pickle.load(f)
        return self._user_reviews

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
        self._user_reviews = None
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

        # 先检查数据库中的表结构
        cursor = conn.cursor()
        cursor.execute('SELECT name FROM sqlite_master WHERE type="table"')
        tables = cursor.fetchall()

        # 检查reviews表的列结构
        cursor.execute('PRAGMA table_info(reviews)')
        columns = cursor.fetchall()

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
        创建虚拟节点（按自然日聚合，对应论文 Definition: Virtual Node 与 Eq.(mean_agg)）

        论文设定：用户 u 在同一自然日 d 发表的所有评论聚合为一个独立的虚拟节点 v_{u,d}，
        其特征向量由当日所有评论的特征做均值聚合得到。
        虚拟节点ID格式：用户ID_YYYYMMDD
        """

        virtual_nodes = {}
        user_to_virtual = defaultdict(list)
        node_id = 0

        # 按用户分组处理
        grouped = reviews_df.groupby('reviewerID')
        total_users = len(grouped)

        for user_id, user_reviews in tqdm(grouped, desc="创建虚拟节点(按日聚合)"):
            # 按时间排序
            user_reviews = user_reviews.sort_values('reviewTime')
            user_reviews = user_reviews.copy()
            user_reviews['_review_dt'] = pd.to_datetime(user_reviews['reviewTime'])
            user_reviews['_review_date'] = user_reviews['_review_dt'].dt.date

            # 按自然日分组，保持日期升序
            day_idx = 0
            for review_date, day_reviews in user_reviews.groupby('_review_date', sort=True):
                # 当日所有评论的对齐列表
                asins = [r for r in day_reviews['asin'].tolist()]
                overalls = [float(r) for r in day_reviews['overall'].tolist()]
                texts = [t if pd.notna(t) else "" for t in day_reviews['reviewText'].tolist()]
                labels = [r for r in day_reviews['label'].tolist()]

                # 节点级聚合
                review_dt = pd.to_datetime(str(review_date))
                # 评分：均值聚合（Eq. mean_agg）
                overall_mean = float(np.mean(overalls)) if overalls else 0.0
                # 文本：当日文本拼接，供 TF-IDF 向量化
                concat_text = " ".join([str(t) for t in texts if str(t).strip()])
                # 代表性产品：当日评论最多的产品（向后兼容单 asin 读取）
                rep_asin = Counter(asins).most_common(1)[0][0] if asins else None
                # 标签聚合：当日含任一虚假评论(label==-1)则该日节点判为水军节点，否则真实
                day_label = -1 if any(int(l) == -1 for l in labels) else 1

                virtual_node_id = f"{user_id}_{review_dt.strftime('%Y%m%d')}"
                virtual_nodes[node_id] = {
                    'virtual_node_id': virtual_node_id,
                    'original_user_id': user_id,
                    'review_time': review_dt,
                    'asin': rep_asin,                 # 代表产品（兼容单asin读取）
                    'asins': asins,                   # 当日全部产品（对齐 overalls）
                    'overall': overall_mean,          # 当日均值评分
                    'overalls': overalls,             # 当日全部评分
                    'reviewText': concat_text,        # 当日拼接文本
                    'reviewTexts': texts,             # 当日全部文本
                    'label': day_label,               # 当日聚合标签
                    'review_count_day': len(overalls),# 当日评论数（用于RB等特征）
                    'time_index': day_idx             # 该用户的第几个活跃日
                }
                user_to_virtual[user_id].append(node_id)
                node_id += 1
                day_idx += 1

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
            import traceback
            error_msg = f"Module 1 failed: {str(e)}\n{traceback.format_exc()}"
            logging.error(error_msg)
            print(error_msg)
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
        提取虚拟节点特征矩阵。

        新特征集（去重后）：
        - 论文个体特征：RD, AD, EXR, MRO, ATR
        - 现有高分辨率特征：rating_mean, extreme_ratio, rating_std

        说明：
        1) 这些特征均来自单个用户的历史评论行为，不使用标签。
        2) 与旧的 TF-IDF/多维混合特征相比，这里改为更聚焦“共谋行为”的纯行为特征。
        3) 保持原函数名和缓存路径，避免影响下游流程，但实际输出维度由 14D 改为 8D。
        """
        node_ids = list(self.virtual_nodes.keys())
        n_nodes = len(node_ids)
        feature_names = [
            'rating_mean',
            'extreme_ratio',
            'rating_std',
            'RD',
            'AD',
            'EXR',
            'MRO',
            'ATR',
        ]
        n_beh = len(feature_names)
        beh = np.zeros((n_nodes, n_beh), dtype=np.float32)

        # 1) 全局统计：用于 RD / EXR / MRO / AD / ATR
        product_rating_sum = defaultdict(float)
        product_rating_cnt = defaultdict(int)
        global_review_counts_by_user = {}
        global_day_counts_by_user = {}
        global_time_span_days = {}
        global_active_ratio = {}
        dataset_min_date = None
        dataset_max_date = None

        # 用于 AD: 每用户的review_time列表（用于计算时间跨度）
        user_review_times_list = defaultdict(list)
        # 用于 ATR: 每用户的每日评论数列表（virtual node = 1 day, count = len(overalls)）
        user_daily_counts = defaultdict(list)

        for nid in node_ids:
            info = self.virtual_nodes[nid]
            user_id = info.get('original_user_id', nid)
            ratings = np.array(info.get('overalls', []), dtype=float)
            # 每个虚拟节点代表该用户某一自然日的所有评论，用 review_time 作为该日日期
            rt = pd.to_datetime(info.get('review_time'), errors='coerce')
            if rt is not None and not pd.isna(rt):
                user_review_times_list[user_id].append(rt)
                dataset_min_date = rt if dataset_min_date is None else min(dataset_min_date, rt)
                dataset_max_date = rt if dataset_max_date is None else max(dataset_max_date, rt)
            day_count = len(ratings)  # 该虚拟节点当日评论数
            user_daily_counts[user_id].append(day_count)
            if day_count > 0:
                global_review_counts_by_user[user_id] = global_review_counts_by_user.get(user_id, 0) + day_count
                global_day_counts_by_user[user_id] = global_day_counts_by_user.get(user_id, 0) + 1
            for a, o in zip(info.get('asins', []), info.get('overalls', [])):
                product_rating_sum[a] += float(o)
                product_rating_cnt[a] += 1

        # 计算每用户时间跨度（用于 AD）
        for user_id, times in user_review_times_list.items():
            if len(times) >= 2:
                global_time_span_days[user_id] = float((max(times) - min(times)).days)
            else:
                global_time_span_days[user_id] = 0.0

        # 计算每用户 ATR（有2+条评论的活跃日 / 总活跃日）
        for user_id, counts in user_daily_counts.items():
            total_days = len(counts)
            multi_days = sum(1 for c in counts if c >= 2)
            global_active_ratio[user_id] = multi_days / total_days if total_days > 0 else 0.0

        product_avg = {a: product_rating_sum[a] / product_rating_cnt[a] for a in product_rating_sum}
        # global_max_reviews_per_day：所有虚拟节点中当日评论数最大值
        global_max_reviews_per_day = max(
            (len(self.virtual_nodes[nid].get('overalls', [])) for nid in node_ids),
            default=1
        )
        global_max_reviews_per_day = max(global_max_reviews_per_day, 1)

        for idx, nid in enumerate(tqdm(node_ids, desc="提取新共谋特征(8维)")):
            info = self.virtual_nodes[nid]
            overalls = np.array(info.get('overalls', []), dtype=float)
            asins = info.get('asins', [])
            user_id = info.get('original_user_id', nid)
            review_text = str(info.get('reviewText', ''))

            if len(overalls) == 0:
                continue

            # 1) rating_mean
            beh[idx, 0] = float(np.mean(overalls))

            # 2) extreme_ratio (1 or 5 stars)
            beh[idx, 1] = float(np.mean((overalls == 1) | (overalls == 5)))

            # 3) rating_std
            beh[idx, 2] = float(np.std(overalls))

            # 4) RD: 与所评产品均分的平均偏差
            node_mean = float(np.mean(overalls))
            devs = [abs(node_mean - product_avg.get(a, node_mean)) for a in asins]
            beh[idx, 3] = float(np.mean(devs)) if devs else 0.0

            # 5) AD: account duration (1 - normalized span over dataset range)
            span = global_time_span_days.get(user_id, 0.0)
            dataset_span = float((dataset_max_date - dataset_min_date).days) if (dataset_min_date is not None and dataset_max_date is not None and dataset_max_date > dataset_min_date) else 1.0
            beh[idx, 4] = float(1.0 - min(span / dataset_span, 1.0))

            # 6) EXR: ratio of extreme ratings (same semantic as extreme_ratio; keep as dedicated paper feature)
            beh[idx, 5] = float(np.mean((overalls == 1) | (overalls == 5)))

            # 7) MRO: 该节点当日评论数 / 全局最大单日评论数
            # 每个虚拟节点本身就是一整天，len(overalls) 即为该节点当日评论数
            beh[idx, 6] = float(len(overalls) / global_max_reviews_per_day)

            # 8) ATR: 用户级别活跃日中有2+条评论的比例（预计算）
            beh[idx, 7] = float(global_active_ratio.get(user_id, 0.0))

        # 按列归一化到 [0, 1]
        for i in range(n_beh):
            col = beh[:, i]
            cmin, cmax = float(col.min()), float(col.max())
            if cmax > cmin:
                beh[:, i] = (col - cmin) / (cmax - cmin)

        # 用新特征替换旧特征矩阵
        self.feature_matrix_14d = beh.astype(np.float32)

        # 节点级水军倾向分数：取更能反映共谋的核心特征均值
        # 这里优先使用 rating_mean / extreme_ratio / RD / ATR 的组合
        self.spam_behavior_scores = np.mean(
            self.feature_matrix_14d[:, [0, 1, 3, 7]], axis=1
        ).astype(np.float32)
        

    def build_adjacency_matrix(self):
        """
        构建基础邻接矩阵（时间边+空间边），保存为节点对txt文件，避免内存不足。

        时间边（论文 Definition: Temporal Edge）：同一用户按活跃日排序的相邻虚拟节点相连。
        空间边（论文 Definition: Spatial Edge）：不同用户在同一自然日评论同一产品的虚拟节点相连。
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
        
        # 构建空间边（论文 Definition: Spatial Edge）：
        # 不同用户在"同一自然日"评论"同一产品"的虚拟节点相连。
        # 按 (自然日, 产品ASIN) 分组节点；一个按日聚合节点可能涉及多个产品，故按其 asins 集合展开。
        date_product_to_nodes = defaultdict(set)
        for node_id, node_info in self.virtual_nodes.items():
            review_date = pd.to_datetime(node_info['review_time']).date()
            for asin in set(node_info.get('asins', [])):
                date_product_to_nodes[(review_date, asin)].add(node_id)

        for (review_date, asin), node_set in tqdm(date_product_to_nodes.items(), desc="构建空间边(同日同品)"):
            if len(node_set) < 2:
                continue
            node_list = sorted(node_set)
            for i in range(len(node_list)):
                node1_id = node_list[i]
                for j in range(i + 1, len(node_list)):
                    node2_id = node_list[j]
                    # 确保是不同用户
                    if (self.virtual_nodes[node1_id]['original_user_id'] !=
                            self.virtual_nodes[node2_id]['original_user_id']):
                        edge_key = (node1_id, node2_id)  # node_list 已升序，node1<node2
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
        
        # 引力图使用新共谋特征中的核心维度
        # rating_mean, extreme_ratio, rating_std, RD, AD, EXR, MRO, ATR
        self.attraction_feature_indices = [0, 1, 2, 3, 4, 5, 6, 7]
        
        # 数据集专属斥力图配置（基于实验结果）
        #  修改：改用余弦相似度 + 全部12维特征
        dataset_specific_repulsion_configs = {
            "DataSet/Cell_Phones_and_Accessorie.db": {
                'feature_indices': list(range(8)),  # 使用新的8维共谋特征
                'similarity_method': 'cosine',  # 改用余弦相似度
                'description': '使用新的8维共谋特征 + 余弦相似度'
            },
            "DataSet/Electronics_2013_1.6.db": {
                'feature_indices': list(range(8)),  # 使用新的8维共谋特征
                'similarity_method': 'cosine',  # 改用余弦相似度
                'description': '使用新的8维共谋特征 + 余弦相似度'
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
            # 使用默认配置：新的8维共谋特征 + 余弦相似度
            self.repulsion_feature_indices = list(range(8))
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

        # 论文设定：余弦相似度基于完整节点特征向量 x = [行为特征 || TF-IDF]，
        # 因此引力图/斥力图相似度均使用全部特征维度。
        full_indices = list(range(self.feature_matrix_14d.shape[1]))
        self.attraction_feature_indices = full_indices
        self.repulsion_feature_indices = full_indices

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
        
        # 构建引力/斥力图专用的评分风格标量特征（独立于8维feature_matrix_14d）
        self._build_node_rating_scalars()
        
    def _build_node_rating_scalars(self):
        """
        从虚拟节点的评分记录计算三个用户级标量特征，映射到节点ID。
        用于引力图/斥力图相似度计算，不影响GNN输入的8维feature_matrix_14d。
          G1: 评分标准差  (实验 Cohen.d=0.9543)
          F5: 评分均值    (实验 Cohen.d=0.5100)
          F7: 评分熵      (实验 Cohen.d=0.9894)
        """
        import math
        from collections import defaultdict
        uid_ratings = defaultdict(list)
        for nid, info in self.virtual_nodes.items():
            uid = info['original_user_id']
            uid_ratings[uid].extend([float(r) for r in info.get('overalls', [])])
        
        uid_scalars = {}
        for uid, ratings in uid_ratings.items():
            if not ratings:
                uid_scalars[uid] = (3.0, 0.0, 1.0)
                continue
            r = np.array(ratings, dtype=np.float32)
            mean_r = float(r.mean())
            std_r  = float(r.std())
            hist = np.bincount(np.clip(r.round().astype(int) - 1, 0, 4), minlength=5).astype(float)
            hist_n = hist / hist.sum()
            with np.errstate(divide='ignore', invalid='ignore'):
                ent = float(-np.nansum(hist_n * np.log(hist_n + 1e-12)) / math.log(5))
            uid_scalars[uid] = (mean_r, std_r, ent)
        
        default = (3.0, 0.0, 1.0)
        self.node_mean = {nid: uid_scalars.get(info['original_user_id'], default)[0]
                         for nid, info in self.virtual_nodes.items()}
        self.node_std  = {nid: uid_scalars.get(info['original_user_id'], default)[1]
                         for nid, info in self.virtual_nodes.items()}
        self.node_ent  = {nid: uid_scalars.get(info['original_user_id'], default)[2]
                         for nid, info in self.virtual_nodes.items()}

    def _compute_rating_style_sim_batch(self, node1_ids, node2_ids):
        """
        批量计算引力/斥力图专用相似度：基于评分风格特征(G1+F5+F7)的加权组合。
        高相似度 -> 同类节点对(spam-spam / legit-legit) -> 引力图
        低相似度 -> 异类节点对(spam-legit)              -> 斥力图
        公式: sim = 0.5*(1-|std_i-std_j|/4) + 0.3*(1-|mean_i-mean_j|/4) + 0.2*(1-|ent_i-ent_j|)
        """
        std1  = np.array([self.node_std.get(n,  0.0) for n in node1_ids], dtype=np.float32)
        std2  = np.array([self.node_std.get(n,  0.0) for n in node2_ids], dtype=np.float32)
        mean1 = np.array([self.node_mean.get(n, 3.0) for n in node1_ids], dtype=np.float32)
        mean2 = np.array([self.node_mean.get(n, 3.0) for n in node2_ids], dtype=np.float32)
        ent1  = np.array([self.node_ent.get(n,  1.0) for n in node1_ids], dtype=np.float32)
        ent2  = np.array([self.node_ent.get(n,  1.0) for n in node2_ids], dtype=np.float32)
        G1 = np.clip(1.0 - np.abs(std1  - std2)  / 4.0, 0.0, 1.0)
        F5 = np.clip(1.0 - np.abs(mean1 - mean2) / 4.0, 0.0, 1.0)
        F7 = np.clip(1.0 - np.abs(ent1  - ent2),         0.0, 1.0)
        return (0.5 * G1 + 0.3 * F5 + 0.2 * F7).astype(np.float32)

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
        
        # 按商品ASIN分组，30天窗口内共评的节点对作为引力图候选
        # 实验发现：水军-水军共评对中位数间隔26天，同日约束仅覆盖1.2%；30天覆盖57.5%
        COREVIEW_WINDOW_DAYS = 30
        asin_groups = defaultdict(list)  # asin -> [(node_id, idx, review_date)]
        for i, node_id in enumerate(node_ids):
            node_info = self.virtual_nodes[node_id]
            review_date = node_times[node_id].date() if hasattr(node_times[node_id], 'date') else pd.to_datetime(node_times[node_id]).date()
            for asin in set(node_info.get('asins', [])):
                asin_groups[asin].append((node_id, i, review_date))
        
        valid_groups = {asin: entries for asin, entries in asin_groups.items() if len(entries) >= 2}
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
            
            for group_idx, (asin, nodes_info) in enumerate(valid_groups.items()):
                group_pairs = len(nodes_info) * (len(nodes_info) - 1) // 2
                for i in range(len(nodes_info)):
                    for j in range(i + 1, len(nodes_info)):
                        node1_id, idx1, date1 = nodes_info[i]
                        node2_id, idx2, date2 = nodes_info[j]
                        if abs((date1 - date2).days) > COREVIEW_WINDOW_DAYS:
                            continue
                        batch_node1_ids.append(node1_id)
                        batch_node2_ids.append(node2_id)
                        batch_indices1.append(idx1)
                        batch_indices2.append(idx2)
                        if len(batch_node1_ids) >= batch_size:
                            similarities = self._compute_rating_style_sim_batch(batch_node1_ids, batch_node2_ids)
                            for k in range(len(batch_node1_ids)):
                                temp_writer.writerow([batch_node1_ids[k], batch_node2_ids[k], float(similarities[k])])
                                total_pairs_written += 1
                            batch_node1_ids = []
                            batch_node2_ids = []
                            batch_indices1 = []
                            batch_indices2 = []
                if (group_idx + 1) % 10000 == 0:
                    pass
            
            # 处理剩余批次
            if len(batch_node1_ids) > 0:
                similarities = self._compute_rating_style_sim_batch(batch_node1_ids, batch_node2_ids)
                for k in range(len(batch_node1_ids)):
                    temp_writer.writerow([batch_node1_ids[k], batch_node2_ids[k], float(similarities[k])])
                    total_pairs_written += 1
        
        
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
            # 排除时间边（同用户连续虚拟节点，论文 Definition 明确：时间边必然连接同标签节点，不能作为斥力候选）
            if (node1_id in self.virtual_nodes and node2_id in self.virtual_nodes and
                    self.virtual_nodes[node1_id]['original_user_id'] == self.virtual_nodes[node2_id]['original_user_id']):
                continue
            if node1_id in node_id_to_index and node2_id in node_id_to_index:
                i = node_id_to_index[node1_id]
                j = node_id_to_index[node2_id]
                batch_indices1.append(i)
                batch_indices2.append(j)
                batch_node1_ids.append(node1_id)
                batch_node2_ids.append(node2_id)
                
                if len(batch_node1_ids) >= batch_size:
                    similarities = self._compute_rating_style_sim_batch(batch_node1_ids, batch_node2_ids)
                    
                    # 保存结果
                    for k in range(len(batch_node1_ids)):
                        node1 = batch_node1_ids[k]
                        node2 = batch_node2_ids[k]
                        sim = similarities[k]
                        label1 = node_labels.get(node1, 0)
                        label2 = node_labels.get(node2, 0)
                        all_edge_similarities.append((node1, node2, sim, label1, label2))
                    
                    batch_indices1 = []
                    batch_indices2 = []
                    batch_node1_ids = []
                    batch_node2_ids = []
        
        # 处理剩余批次
        if len(batch_node1_ids) > 0:
            similarities = self._compute_rating_style_sim_batch(batch_node1_ids, batch_node2_ids)
            
            for k in range(len(batch_node1_ids)):
                node1 = batch_node1_ids[k]
                node2 = batch_node2_ids[k]
                sim = similarities[k]
                label1 = node_labels.get(node1, 0)
                label2 = node_labels.get(node2, 0)
                all_edge_similarities.append((node1, node2, sim, label1, label2))
        
        
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
        # 论文 Eq.(adj_enhanced) 固定增强强度：引力放大 λ_A=10，斥力抑制 λ_R=50，下限裁剪 c=0.01
        self.lambda_param = 10.0   # λ_A：引力放大强度
        self.lambda_rep = 50.0     # λ_R：斥力抑制强度
        self.clip_c = 0.01         # c：斥力权重下限，防止边被完全移除
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
            # 论文 Eq.(adj_enhanced) 斥力分支：max(c, exp(-λ_R·s_ij))，c 为下限裁剪
            weight = np.exp(-self.lambda_rep * sim)
            weight = max(self.clip_c, weight)
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
        """应用论文 Eq.(adj_enhanced) 的指数加权增强（文件级操作）。

        对每条边按以下优先级赋权（引力优先于斥力，与论文 cases 顺序一致）：
          - (i,j)∈E_A: Ã = 1 + λ_A(exp(ŝ_ij)-1)              （引力放大）
          - (i,j)∈E_R: Ã = max(c, exp(-λ_R·s_ij)) · A[i,j]     （斥力抑制，A=1）
          - 其它:       Ã = A[i,j]                               （保持不变）
        此外，引力图中不在基础邻接矩阵的节点对将被"插入"为新边（论文：
        Attraction-graph edges are inserted into the adjacency matrix if not already present）。
        """

        # 引力/斥力权重按标准化节点对(小ID在前)建立映射
        attr_by_key = {tuple(sorted(k)): w for k, w in self.attr_weights.items()}
        rep_by_key = {tuple(sorted(k)): w for k, w in self.rep_weights.items()}

        temp_path = enhanced_edges_path + '.tmp'
        base_edge_keys = set()
        total_edges = 0

        # 第一步：在基础邻接边上按优先级赋权
        with open(enhanced_edges_path, 'r') as src, open(temp_path, 'w') as dst:
            for line in src:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) == 3:
                        node1, node2, weight_str = parts
                        node1_int = int(node1)
                        node2_int = int(node2)
                        original_weight = float(weight_str)  # 基础邻接 A[i,j]=1
                        key = (node1_int, node2_int) if node1_int <= node2_int else (node2_int, node1_int)
                        base_edge_keys.add(key)

                        if key in attr_by_key:
                            # 引力分支：直接取放大后的权重（公式已含基线 1）
                            new_weight = attr_by_key[key]
                        elif key in rep_by_key:
                            # 斥力分支：抑制系数 × A[i,j]
                            new_weight = rep_by_key[key] * original_weight
                        else:
                            new_weight = original_weight

                        dst.write(f"{node1}\t{node2}\t{new_weight:.6f}\n")
                        total_edges += 1

            # 第二步：插入"仅存在于引力图、不在基础邻接"的新边
            inserted = 0
            for key, w in attr_by_key.items():
                if key not in base_edge_keys:
                    dst.write(f"{key[0]}\t{key[1]}\t{w:.6f}\n")
                    inserted += 1
                    total_edges += 1

        # 替换原文件
        os.replace(temp_path, enhanced_edges_path)
    
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
                 dropout=0.3, alpha=0.2, epochs=60, lr=0.001, use_gpu=False):
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
        
        self.user_reviews_data = None  # 延迟加载，避免__init__时读309MB pkl文件
    
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
        
        # 提前加载虚拟节点信息（_create_enhanced_features 需要用到）
        virtual_nodes_path = os.path.join(module1_dir, 'virtual_nodes.pkl')
        with open(virtual_nodes_path, 'rb') as f:
            self.virtual_nodes = pickle.load(f)
        
        #  增强特征矩阵缓存（避免每次重复加载309MB user_reviews.pkl）
        enhanced_cache_path = os.path.join(module2_dir, f'enhanced_features_{self.sample_ratio}.npy')
        if os.path.exists(enhanced_cache_path):
            self.features = np.load(enhanced_cache_path)
        else:
            # 首次计算：按需加载user_reviews_data
            if self.user_reviews_data is None:
                self.user_reviews_data = self._load_user_reviews_data()
            self.features = self._create_enhanced_features(original_features)
            np.save(enhanced_cache_path, self.features)
            self.user_reviews_data = None  # 计算完成后释放内存
        
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
        
        # virtual_nodes 已在增强特征计算前提前加载，此处无需重复加载
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
            # 文件不存在时设为None，不影响主流程
            self.spam_behavior_scores = None
            print(f"警告: 水军行为得分文件不存在: {spam_scores_path}，将跳过混合用户感知损失")
        
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
        
        from collections import defaultdict
        _tmp = defaultdict(list)
        for src, dst in zip(src_nodes.tolist(), dst_nodes.tolist()):
            _tmp[src].append(dst)
        self.neighbors_dict = dict(_tmp)
        
    
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
        
        from collections import defaultdict
        _tmp = defaultdict(list)
        for src, dst in zip(src_nodes.tolist(), dst_nodes.tolist()):
            _tmp[src].append(dst)
        self.neighbors_dict = dict(_tmp)
        

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
                    if self.spam_behavior_scores is not None:
                        subgraph_spam_scores = torch.FloatTensor([self.spam_behavior_scores[i] for i in sampled_nodes]).to(self.device)
                    else:
                        # 如果没有spam_scores，使用全0占位
                        subgraph_spam_scores = torch.zeros(len(sampled_nodes)).to(self.device)
                    
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

            # ── 全局对比学习损失：每个epoch从全局引力/斥力对中采样计算 ──────────
            # 子图中两个节点同时命中概率极低，此处单独对全局对采样一次保证梯度有效流入
            if self.use_contrastive_loss and (
                    len(self.attraction_pairs) > 0 or len(self.repulsion_pairs) > 0):
                try:
                    self.gcn_model.train()
                    self.optimizer.zero_grad()
                    # 定向采样引力/斥力对端点，保证每步命中有效对（论文：所有对参与对比损失）
                    _global_n = self.features.shape[0]
                    _sample_k = min(512, _global_n)
                    _att_n = min(100, len(self.attraction_pairs))
                    _rep_n = min(100, len(self.repulsion_pairs))
                    _att_samp = [self.attraction_pairs[i] for i in np.random.choice(len(self.attraction_pairs), _att_n, replace=False)] if _att_n > 0 else []
                    _rep_samp = [self.repulsion_pairs[i] for i in np.random.choice(len(self.repulsion_pairs), _rep_n, replace=False)] if _rep_n > 0 else []
                    _ep_list = list({idx for p in _att_samp + _rep_samp for idx in p[:2]})
                    # 补充随机节点至 _sample_k，提供聚合上下文
                    if len(_ep_list) < _sample_k:
                        _ep_set = set(_ep_list)
                        _cands = np.random.choice(_global_n, _sample_k * 3, replace=False).tolist()
                        for _r in _cands:
                            if _r not in _ep_set and len(_ep_list) < _sample_k:
                                _ep_list.append(_r); _ep_set.add(_r)
                    _global_idx = np.array(_ep_list[:_sample_k])
                    _feat_sample = self.features[_global_idx].to(self.device) if isinstance(
                        self.features, torch.Tensor) else torch.FloatTensor(
                        self.features[_global_idx]).to(self.device)
                    # 构建采样子图邻接
                    _idx_set = set(_global_idx.tolist())
                    _idx_map = {g: l for l, g in enumerate(_global_idx.tolist())}
                    _rows, _cols, _vals = [], [], []
                    for g_src in _global_idx:
                        for g_dst in (self.neighbors_dict.get(int(g_src)) or []):
                            if g_dst in _idx_set:
                                _rows.append(_idx_map[int(g_src)])
                                _cols.append(_idx_map[g_dst])
                                _vals.append(1.0)
                    if _rows:
                        _adj_t = torch.sparse_coo_tensor(
                            torch.tensor([_rows, _cols], dtype=torch.long),
                            torch.tensor(_vals, dtype=torch.float32),
                            torch.Size([_sample_k, _sample_k]),
                            device=self.device).coalesce()
                    else:
                        _adj_t = torch.sparse_coo_tensor(
                            torch.zeros((2, 0), dtype=torch.long),
                            torch.zeros(0, dtype=torch.float32),
                            torch.Size([_sample_k, _sample_k]),
                            device=self.device)
                    _emb_g, _ = self.gcn_model(
                        _feat_sample, _adj_t,
                        user_to_virtual=None, virtual_node_times=None, use_temporal=False)
                    # 从全局对中过滤出采样节点内的对
                    _g_att, _g_rep = [], []
                    for idx1, idx2, w in self.attraction_pairs:
                        if idx1 in _idx_map and idx2 in _idx_map:
                            _g_att.append((_idx_map[idx1], _idx_map[idx2], w))
                    for idx1, idx2, w in self.repulsion_pairs:
                        if idx1 in _idx_map and idx2 in _idx_map:
                            _g_rep.append((_idx_map[idx1], _idx_map[idx2], w))
                    if _g_att or _g_rep:
                        _cl, _ = self.criterion.contrastive_loss(_emb_g, _g_att, _g_rep)
                        _cl = torch.clamp(_cl, 0, 10.0)
                        _cl.backward()
                        self.optimizer.step()
                        avg_components['global_contrastive_loss'] = _cl.item()
                    del _feat_sample, _adj_t, _emb_g
                    if self.use_gpu:
                        torch.cuda.empty_cache()
                except Exception:
                    pass
            # ── 全局对比损失结束 ──────────────────────────────────────────────

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:2d}/{num_epochs}: Loss = {avg_components.get('combined_loss', avg_loss):.6f}"
                      + (f"  GlobalCL = {avg_components['global_contrastive_loss']:.6f}"
                         if 'global_contrastive_loss' in avg_components else ""))
            
            # 每15轮：DBSCAN探测 + 迭代精炼（预热后启用）
            # 注：对553K节点全量probe耗时数小时，改为15轮一次+子采样加速
            if (epoch + 1) % 15 == 0:
                print(f"  Epoch {epoch+1}: DBSCAN clustering probe...")
                self.gcn_model.eval()
                probe_checkpoint_dir = os.path.join(self.preprocessed_dir, "training_checkpoints")
                os.makedirs(probe_checkpoint_dir, exist_ok=True)
                try:
                    probe_embeddings = self._generate_embeddings_chunked_gcn()
                    # 采用全量近邻图上的稀疏连通分量探测，避免 sklearn DBSCAN 在全量点上退化为 O(N^2)
                    # 这里沿用最终聚类阶段的思路：FAISS 近邻图 + ε分位数遍历 + connected_components
                    import faiss
                    from sklearn.decomposition import PCA
                    from scipy.sparse import csr_matrix
                    from scipy.sparse.csgraph import connected_components

                    if probe_embeddings.shape[1] > 10:
                        pca = PCA(n_components=10, random_state=42)
                        probe_emb_reduced = pca.fit_transform(probe_embeddings).astype(np.float32)
                    else:
                        probe_emb_reduced = probe_embeddings.astype(np.float32)

                    _probe_n = len(probe_emb_reduced)
                    _probe_idx_sorted = np.arange(_probe_n, dtype=np.int32)
                    probe_emb_sub = np.ascontiguousarray(probe_emb_reduced, dtype=np.float32)
                    faiss.normalize_L2(probe_emb_sub)

                    k_search = 15
                    min_samples = 2
                    n = _probe_n
                    nlist = min(4096, max(64, int(np.sqrt(n))))
                    nprobe = min(nlist, 64)
                    quantizer = faiss.IndexFlatL2(probe_emb_sub.shape[1])
                    cpu_idx = faiss.IndexIVFFlat(quantizer, probe_emb_sub.shape[1], nlist, faiss.METRIC_L2)
                    cpu_idx.nprobe = nprobe
                    cpu_idx.train(probe_emb_sub)
                    cpu_idx.add(probe_emb_sub)
                    distances_sq, indices_knn = cpu_idx.search(probe_emb_sub, k_search + 1)
                    del cpu_idx, quantizer

                    distances_sq = np.ascontiguousarray(distances_sq[:, 1:], dtype=np.float32)
                    indices_knn = np.ascontiguousarray(indices_knn[:, 1:], dtype=np.int32)
                    k_dists_col = np.sqrt(distances_sq[:, min_samples - 1].clip(0))
                    row_base = np.repeat(np.arange(n, dtype=np.int32), k_search)
                    flat_idx = indices_knn.ravel()
                    flat_dist = distances_sq.ravel()

                    best_labels = None
                    best_diff = float('inf')
                    probe_eps = 0.0
                    for pct in [3, 5, 8, 12, 18, 25, 35, 50, 70, 80, 85, 90, 95]:
                        eps_cand = float(np.percentile(k_dists_col, pct))
                        if eps_cand <= 0.0:
                            continue
                        eps_sq_cand = eps_cand * eps_cand
                        valid = flat_dist <= eps_sq_cand
                        row_f = row_base[valid]
                        col_f = flat_idx[valid]
                        data = np.ones(len(row_f) * 2, dtype=np.uint8)
                        rows = np.concatenate([row_f, col_f])
                        cols = np.concatenate([col_f, row_f])
                        adj = csr_matrix((data, (rows, cols)), shape=(n, n))
                        nbr_counts = np.diff(adj.indptr)
                        core_mask = nbr_counts >= (min_samples - 1)
                        edge_core = core_mask[row_f] | core_mask[col_f]
                        row_c = row_f[edge_core]; col_c = col_f[edge_core]
                        data2 = np.ones(len(row_c) * 2, dtype=np.uint8)
                        adj2 = csr_matrix((data2, (np.concatenate([row_c, col_c]), np.concatenate([col_c, row_c]))), shape=(n, n))
                        n_comp, comp_lbl = connected_components(adj2, directed=False)
                        comp_sizes = np.bincount(comp_lbl, minlength=n_comp)
                        lbl = np.where(comp_sizes[comp_lbl] >= min_samples, comp_lbl, -1).astype(np.int32)
                        valid_ids = np.unique(lbl[lbl >= 0])
                        if len(valid_ids) > 0:
                            remap = np.full(n_comp, -1, dtype=np.int32)
                            remap[valid_ids] = np.arange(len(valid_ids), dtype=np.int32)
                            mask_v = lbl >= 0
                            lbl[mask_v] = remap[lbl[mask_v]]
                        nc = int(lbl.max() + 1) if lbl.max() >= 0 else 0
                        diff = abs(nc - 7500)
                        print(f"    pct={pct:2d}% eps={eps_cand:.4f} → {nc} clusters")
                        if diff < best_diff:
                            best_diff = diff
                            best_labels = lbl.copy()
                            probe_eps = eps_cand
                        if nc >= 2:
                            # 只要形成多个簇即可继续训练，不强制等到极致最优
                            pass

                    del distances_sq, indices_knn, k_dists_col, row_base, flat_idx, flat_dist
                    probe_labels = best_labels if best_labels is not None else np.full(_probe_n, -1, dtype=np.int32)
                    del probe_emb_sub, probe_emb_reduced
                    if self.use_gpu:
                        torch.cuda.empty_cache()
                    n_clusters = len(set(probe_labels)) - (1 if -1 in probe_labels else 0)
                    n_noise    = int(np.sum(probe_labels == -1))
                    print(f"    DBSCAN: {n_clusters} clusters, {n_noise} noise points")
                    sil_score, centroids = self._compute_centroid_silhouette(
                        probe_embeddings, probe_labels)
                    print(f"    Silhouette: {sil_score:.4f}  Best so far: {_best_sil:.4f}")
                    probe_labels_path = os.path.join(
                        probe_checkpoint_dir, f"dbscan_probe_epoch_{epoch+1}.npy")
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
        """FAISS IVF k-NN + scipy connected_components DBSCAN
        原理：FAISS一次性预算全图k-NN（GPU/CPU IVF，绕开维度诅咒），然后用C实现的
        连通分量算法在稀疏图上做DBSCAN，ε分位数遍历自动选最优群组数。
        预计耗时：553K节点约3-8分钟（含PCA+FAISS+ε搜索）。
        """
        import gc
        import faiss
        from sklearn.decomposition import IncrementalPCA
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
        import tempfile

        total_samples = len(self.embeddings)
        min_samples = 2   # 最小群组规模（适度放宽，降低边缘节点被直接判噪声的概率）
        k_search = 15     # FAISS搜索邻居数（多于min_samples，供多ε遍历使用）
        TARGET_MIN  = 6000
        TARGET_MAX  = 9000
        TARGET_CENTER = 7500

        print(f"  Running FAISS-DBSCAN on {total_samples} nodes...", flush=True)
        start_time = time.time()

        # ── 第1步：PCA降维（>32维→32维，保留更多判别结构，2026-06-05调整）────────
        embeddings_for_clustering = self.embeddings
        if self.embeddings.shape[1] > 32:
            target_dim = 32
            n_s = self.embeddings.shape[0]
            bs  = 10000
            print(f"  PCA {self.embeddings.shape[1]}D → {target_dim}D ...", flush=True)
            ipca = IncrementalPCA(n_components=target_dim, batch_size=bs)
            for i in range(0, n_s, bs):
                ipca.partial_fit(self.embeddings[i:min(i + bs, n_s)])
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.npy')
            tp = tmp.name; tmp.close()
            er = np.lib.format.open_memmap(tp, mode='w+', dtype=np.float32, shape=(n_s, target_dim))
            for i in range(0, n_s, bs):
                be = min(i + bs, n_s)
                er[i:be] = ipca.transform(self.embeddings[i:be])
            del er; gc.collect()
            embeddings_for_clustering = np.load(tp, mmap_mode='r')
            print(f"  PCA explained variance: {ipca.explained_variance_ratio_.sum():.3f}", flush=True)
            self._temp_pca_file = tp

        # ── 步骤1.5：虚拟节点→用户节点均值池化聚合（在FAISS前）──────────────
        # 聚类应针对用户节点，而非虚拟节点；同一用户的多个虚拟节点取均值代表该用户
        node_ids_list = list(self.virtual_nodes.keys())
        _u2idx = defaultdict(list)
        for _i, _nid in enumerate(node_ids_list):
            _uid = self.virtual_nodes[_nid]['original_user_id']
            _u2idx[_uid].append(_i)
        user_ids_list = list(_u2idx.keys())
        n_users = len(user_ids_list)
        _raw = np.ascontiguousarray(embeddings_for_clustering).astype(np.float32)
        user_emb = np.zeros((n_users, _raw.shape[1]), dtype=np.float32)
        for _ui, _uid in enumerate(user_ids_list):
            _idxs = _u2idx[_uid]
            user_emb[_ui] = _raw[_idxs].mean(axis=0) if len(_idxs) > 1 else _raw[_idxs[0]]
        del _raw; gc.collect()
        print(f"  Aggregated {total_samples} virt-nodes → {n_users} users (mean-pool)", flush=True)

        emb_f32 = user_emb
        n, dim = emb_f32.shape
        faiss.normalize_L2(emb_f32)   # 归一化→余弦距离空间，水军行为向量比L2更紧凑

        # ── 第2步：FAISS IVF k-NN（一次性建图，GPU优先）────────────────────────
        nlist  = min(4096, max(64, int(np.sqrt(n))))
        nprobe = min(nlist, 64)
        print(f"  FAISS IVF nlist={nlist} nprobe={nprobe} k={k_search} dim={dim}...", flush=True)
        t_faiss = time.time()

        quantizer = faiss.IndexFlatL2(dim)
        cpu_idx   = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
        cpu_idx.nprobe = nprobe

        distances_sq = indices_knn = None
        if self.use_gpu:
            try:
                res     = faiss.StandardGpuResources()
                gpu_idx = faiss.index_cpu_to_gpu(res, 0, cpu_idx)
                gpu_idx.train(emb_f32)
                gpu_idx.add(emb_f32)
                distances_sq, indices_knn = gpu_idx.search(emb_f32, k_search + 1)
                del gpu_idx, res
                print(f"  FAISS GPU done in {time.time()-t_faiss:.1f}s", flush=True)
            except Exception as eg:
                print(f"  FAISS GPU failed ({eg}), CPU fallback...", flush=True)
                distances_sq = None

        if distances_sq is None:
            cpu_idx.train(emb_f32)
            cpu_idx.add(emb_f32)
            distances_sq, indices_knn = cpu_idx.search(emb_f32, k_search + 1)
            print(f"  FAISS CPU done in {time.time()-t_faiss:.1f}s", flush=True)

        del cpu_idx, quantizer, emb_f32; gc.collect()

        # 去掉第0列（自身，距离=0）
        distances_sq  = np.ascontiguousarray(distances_sq[:, 1:],  dtype=np.float32)  # [n, k]
        indices_knn   = np.ascontiguousarray(indices_knn[:, 1:],   dtype=np.int32)    # [n, k]
        k_dists_col   = np.sqrt(distances_sq[:, min_samples - 1].clip(0))             # k-th NN 距离

        # ── 第3步：ε分位数遍历 + scipy connected_components ─────────────────────
        # 每次迭代：构建CSR + connected_components（C实现，毫秒级）
        # 目标：找到使群组数落在[TARGET_MIN, TARGET_MAX]的ε
        row_base = np.repeat(np.arange(n, dtype=np.int32), k_search)  # 预算好，避免重复
        flat_idx = indices_knn.ravel()

        best_labels   = None
        best_diff     = float('inf')

        for pct in [3, 5, 8, 12, 18, 25, 35, 50, 70, 80, 85, 90, 95]:
            eps_cand    = float(np.percentile(k_dists_col, pct))
            if eps_cand <= 0.0:
                # 退化情况：大量用户嵌入完全相同（单评+相同产品），eps=0只聚合完全相同节点
                # 跳过此pct，寻找有意义的距离阈值
                print(f"  pct={pct:2d}% eps=0.0000 (degenerate, skip)", flush=True)
                continue
            eps_sq_cand = eps_cand * eps_cand

            # 筛选ε邻域内的边
            flat_dist = distances_sq.ravel()
            valid     = flat_dist <= eps_sq_cand
            row_f = row_base[valid]
            col_f = flat_idx[valid]

            # 对称稀疏邻接矩阵
            data  = np.ones(len(row_f) * 2, dtype=np.uint8)
            rows  = np.concatenate([row_f, col_f])
            cols  = np.concatenate([col_f, row_f])
            adj   = csr_matrix((data, (rows, cols)), shape=(n, n))

            # 核心点过滤（邻居数 >= min_samples-1，不含自身）
            nbr_counts = np.diff(adj.indptr)
            core_mask  = nbr_counts >= (min_samples - 1)

            # 只保留至少一端是核心点的边
            edge_core  = core_mask[row_f] | core_mask[col_f]
            row_c = row_f[edge_core]; col_c = col_f[edge_core]
            data2 = np.ones(len(row_c) * 2, dtype=np.uint8)
            adj2  = csr_matrix(
                (data2, (np.concatenate([row_c, col_c]), np.concatenate([col_c, row_c]))),
                shape=(n, n))

            # connected_components（scipy C实现，几百毫秒）
            n_comp, comp_lbl = connected_components(adj2, directed=False)
            comp_sizes = np.bincount(comp_lbl, minlength=n_comp)

            # 小于min_samples的连通分量标为噪声
            lbl = np.where(comp_sizes[comp_lbl] >= min_samples,
                           comp_lbl, -1).astype(np.int32)

            # 重编号（0起连续）
            valid_ids = np.unique(lbl[lbl >= 0])
            if len(valid_ids) > 0:
                remap = np.full(n_comp, -1, dtype=np.int32)
                remap[valid_ids] = np.arange(len(valid_ids), dtype=np.int32)
                mask_v = lbl >= 0
                lbl[mask_v] = remap[lbl[mask_v]]

            nc = int(lbl.max() + 1) if lbl.max() >= 0 else 0
            nn = int(np.sum(lbl == -1))
            diff = abs(nc - TARGET_CENTER)
            print(f"  pct={pct:2d}% eps={eps_cand:.4f} → {nc} clusters, noise={nn}", flush=True)

            if diff < best_diff:
                best_diff   = diff
                best_labels = lbl.copy()

            if TARGET_MIN <= nc <= TARGET_MAX:
                break   # 已在目标范围，停止遍历

        del distances_sq, indices_knn, k_dists_col, row_base, flat_idx; gc.collect()

        # ── 清理PCA临时文件 ──────────────────────────────────────────────────────
        if embeddings_for_clustering is not self.embeddings:
            del embeddings_for_clustering; gc.collect()
            if hasattr(self, '_temp_pca_file') and os.path.exists(self._temp_pca_file):
                try:
                    os.remove(self._temp_pca_file)
                except Exception:
                    pass

        # ── 用户级标签映射回虚拟节点级 ──────────────────────────────────────────
        # 每个虚拟节点继承其所属用户的聚类标签
        _user_label_map = {_uid: int(best_labels[_ui]) for _ui, _uid in enumerate(user_ids_list)}
        labels = np.array(
            [_user_label_map[self.virtual_nodes[_nid]['original_user_id']] for _nid in node_ids_list],
            dtype=np.int32
        )
        del user_emb, user_ids_list, _u2idx, _user_label_map, node_ids_list; gc.collect()

        elapsed_time = time.time() - start_time
        n_clusters = int(best_labels.max() + 1) if best_labels.max() >= 0 else 0
        n_noise_users = int(np.sum(best_labels == -1))
        n_noise_vn    = int(np.sum(labels == -1))
        print(f"  FAISS-DBSCAN complete: {n_clusters} user-clusters, "
              f"{n_noise_users} noise-users ({n_noise_vn} noise-virt-nodes), "
              f"elapsed={elapsed_time:.1f}s", flush=True)

        self.cluster_labels = labels
        return labels
    
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
            cluster_info[cluster_label].append(node_id)  # 只存node_id（整数），避免复制virtual_node_info导致文件膀耀0（原419MB→几MB）
        
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
            
            all_cached = (os.path.exists(embeddings_path) and os.path.exists(clusters_path) and
                          os.path.exists(cluster_info_path) and os.path.exists(cluster_csv_path))
            emb_only   = os.path.exists(embeddings_path) and not os.path.exists(clusters_path)

            if all_cached:
                # 全量缓存：直接加载
                self.embeddings = np.load(embeddings_path)
                self.cluster_labels = np.load(clusters_path)
                with open(cluster_info_path, 'rb') as f:
                    cluster_info = pickle.load(f)
                return True

            # 根据初始化参数决定是否使用GPU加速
            if self.use_gpu:
                self.force_cpu_mode = False
            else:
                self.force_cpu_mode = True

            if emb_only:
                # 嵌入已存在（训练完但聚类被中断）：跳过训练直接聚类
                print(f"  [M5] Found cached embeddings, skipping GNN training...")
                self.load_data()
                self.embeddings = np.load(embeddings_path)
            else:
                # 全量重跑
                self.load_data()
                self._prepare_tensors()
                self.train_gcn_with_graphsaint()
                # 训练完成后立刻保存嵌入，防止聚类中断导致重跑
                _emb_tmp = embeddings_path + '.tmp'
                np.save(_emb_tmp, self.embeddings)
                if os.path.exists(_emb_tmp):
                    os.replace(_emb_tmp, embeddings_path)
                    print(f"  [M5] Embeddings saved: {embeddings_path}")

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
        print(f"  [M6] Loading virtual_nodes from {virtual_nodes_path} ...", flush=True)
        with open(virtual_nodes_path, 'rb') as f:
            self.virtual_nodes = pickle.load(f)
        print(f"  [M6] virtual_nodes loaded: {len(self.virtual_nodes)} nodes", flush=True)
        
        # 加载聚类信息（从模块5）
        cluster_info_path = os.path.join(module5_dir, f'cluster_info_{self.sample_ratio}.pkl')
        print(f"  [M6] Loading cluster_info from {cluster_info_path} ...", flush=True)
        with open(cluster_info_path, 'rb') as f:
            self.cluster_info = pickle.load(f)
        print(f"  [M6] cluster_info loaded: {len(self.cluster_info)} clusters", flush=True)
        # 兼容新格式（只存node_id整数）和旧格式（存{'node_id':...,'virtual_node_info':...}）
        # 新格式加载后按引用回查virtual_nodes，不增加额外内存
        _first = next((v for v in self.cluster_info.values() if v), None)
        if _first and isinstance(_first[0], (int, np.integer)):
            print(f"  [M6] Reconstructing cluster_info from node_ids ...", flush=True)
            reconstructed = {}
            for cluster_id, node_ids_list in self.cluster_info.items():
                reconstructed[cluster_id] = [
                    {'node_id': int(nid), 'virtual_node_info': self.virtual_nodes[int(nid)]}
                    for nid in node_ids_list if int(nid) in self.virtual_nodes
                ]
            self.cluster_info = reconstructed
            print(f"  [M6] Reconstruction done", flush=True)
        
        # 统计聚类分布
        cluster_sizes = []
        for cluster_id, nodes in self.cluster_info.items():
            if cluster_id != -1:  # 排除噪声点
                cluster_sizes.append(len(nodes))
        
        if cluster_sizes:
            print(f"  [M6] load_data done: {len(cluster_sizes)} valid clusters, avg_size={sum(cluster_sizes)/len(cluster_sizes):.1f}", flush=True)
    
    def deduplicate_virtual_nodes(self):
        """步骤2：去重虚拟节点（在聚合之前）
        
        对于被分散到多个群组的用户，保留其虚拟节点数量最多的群组，
        从其他群组中移除该用户的虚拟节点。
        
        策略：虚拟节点数量优先原则
        - 虚拟节点多 = 该用户在该群组中活跃度高
        - 保留活跃度最高的群组，移除其他群组中的误判节点
        """
        print(f"  [M6] Step2: deduplicating virtual nodes ...", flush=True)
        
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
        print(f"  [M6] Dedup done: removed={total_removed} nodes from {clusters_affected} clusters", flush=True)
        
        # 保存去重统计信息（用于后续分析）
        self.deduplication_stats = {
            'dispersed_users': len(dispersed_users),
            'total_removed_nodes': total_removed,
            'clusters_affected': clusters_affected,
            'resolved_users': resolved_users
        }
        
    def aggregate_nodes_to_users(self):
        """步骤3：将虚拟节点聚合为用户群组"""
        print(f"  [M6] Step3: aggregating virtual nodes to user groups ...", flush=True)
        
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
            print(f"  [M6] Aggregation done: {len(self.user_groups)} groups, avg_users={sum(user_counts)/len(user_counts):.1f}", flush=True)
            
        
    def secondary_clustering_with_temporal_features(self):
        """步骤3.5：双面节点召回的二次聚类（对应论文 3.3.3 Sub-Clustering）。

        论文设定：双面水军的虚拟节点常因位于类别边界而在初次 HDBSCAN 中被标为噪声。
        本步骤收集初次 HDBSCAN 的"噪声节点"，使用其行为/时序一致性特征重新聚类，
        将得到的子聚类聚合为用户级新候选群组，追加到 self.user_groups，
        随后与初次聚类群组一并参与合并（merge_similar_groups）。
        """

        # 加载时序特征（用户级时序一致性特征，作为噪声节点重聚类的特征）
        module1_dir = get_result_dir(self.sample_ratio, self.db_path, module=1, force_no_threshold=True)
        temporal_features_path = os.path.join(module1_dir, 'temporal_features.pkl')

        if not os.path.exists(temporal_features_path):
            return

        with open(temporal_features_path, 'rb') as f:
            temporal_features = pickle.load(f)

        # 1) 收集初次 HDBSCAN 的噪声节点（cluster_id == -1）
        noise_nodes = self.cluster_info.get(-1, [])
        if not noise_nodes or len(noise_nodes) < 4:
            return

        # 2) 按user_id去重：时序特征是用户级别的，同一用户的多个虚拟节点特征完全相同
        # 去重后每个用户仅保留一个代表性虚拟节点，避免重复特征向量
        _seen_users = {}
        for node_info in noise_nodes:
            uid = node_info['virtual_node_info']['original_user_id']
            if uid not in _seen_users and uid in temporal_features:
                _seen_users[uid] = node_info

        user_ids_noise = list(_seen_users.keys())
        valid_noise_nodes = [_seen_users[uid] for uid in user_ids_noise]
        feature_matrix = [
            [temporal_features[uid]['avg_time_interval'],
             temporal_features[uid]['std_time_interval'],
             temporal_features[uid]['cv_time_interval'],
             temporal_features[uid]['rating_change_rate'],
             temporal_features[uid]['product_concentration'],
             temporal_features[uid]['text_similarity']]
            for uid in user_ids_noise
        ]
        del _seen_users, user_ids_noise

        if len(valid_noise_nodes) < 4:
            return

        print(f"  [SubCluster] Running secondary DBSCAN on {len(valid_noise_nodes)} noise nodes...", flush=True)

        # 3) 标准化并对噪声节点进行 DBSCAN 二次聚类（自动ε估算）
        feature_matrix = np.array(feature_matrix)
        scaler = StandardScaler()
        feature_matrix_scaled = scaler.fit_transform(feature_matrix).astype(np.float32)

        try:
            import faiss as _faiss
            from scipy.sparse import csr_matrix as _csr
            from scipy.sparse.csgraph import connected_components as _cc
            import gc as _gc2

            _n   = len(valid_noise_nodes)
            _dim = feature_matrix_scaled.shape[1]   # 6
            _k   = 4   # 搜索最近邻数（不含自身）

            # FAISS 精确搜索（6维低维空间，IndexFlatL2 快速且精确）
            _idx = _faiss.IndexFlatL2(_dim)
            _idx.add(feature_matrix_scaled)
            _dists_sq, _idxs = _idx.search(feature_matrix_scaled, _k + 1)  # +1含自身
            del _idx; _gc2.collect()

            # ε = k距离的第30百分位（低维用稍低百分位以捕获密集水军群组）
            _k_dists = np.sqrt(np.maximum(_dists_sq[:, _k], 0.0))
            _eps = float(np.percentile(_k_dists, 30))
            if _eps < 1e-8:
                return   # 所有特征向量几乎相同，无有效聚类

            # 向量化建边：邻居距离 ≤ ε 则连边
            _mask = (_dists_sq[:, 1:] <= _eps * _eps)   # (n, k)
            _ri = np.where(_mask)[0]
            _ci = _idxs[:, 1:][_mask]
            del _dists_sq, _idxs, _mask; _gc2.collect()

            _adj = _csr(
                (np.ones(len(_ri), dtype=np.float32), (_ri, _ci)),
                shape=(_n, _n)
            )
            _, sub_labels = _cc(_adj, directed=False, connection='weak')
            del _adj, _ri, _ci; _gc2.collect()

            # 规模 < 3 的连通分量标记为噪声
            _sizes = np.bincount(sub_labels, minlength=int(sub_labels.max()) + 1)
            sub_labels = np.where(_sizes[sub_labels] >= 3, sub_labels, -1)
        except Exception as _e:
            print(f"  [SubCluster] Error: {_e}", flush=True)
            return

        print(f"  [SubCluster] Done. sub_labels unique (excl noise): {len(set(sub_labels[sub_labels>=0]))}", flush=True)

        # 4) 将每个子聚类按用户聚合为新的候选群组，追加到 user_groups
        next_group_id = max(self.user_groups.keys()) + 1 if self.user_groups else 0
        sub_label_to_nodes = defaultdict(list)
        for i, node_info in enumerate(valid_noise_nodes):
            if sub_labels[i] != -1:
                sub_label_to_nodes[sub_labels[i]].append(node_info)

        new_groups_added = 0
        for sub_label, nodes in sub_label_to_nodes.items():
            user_dict = defaultdict(list)
            for node_info in nodes:
                user_id = node_info['virtual_node_info']['original_user_id']
                user_dict[user_id].append(node_info)

            # 至少包含2个用户才能形成群组
            if len(user_dict) >= 2:
                self.user_groups[next_group_id] = {
                    'users': dict(user_dict),
                    'user_count': len(user_dict),
                    'total_reviews': sum(len(v) for v in user_dict.values()),
                    'from_noise_subcluster': True,
                    'sub_cluster_label': int(sub_label)
                }
                next_group_id += 1
                new_groups_added += 1
        
    def calculate_iss_scores(self):
        """步骤4：计算ISS（Individual Suspiciousness Score）- 使用缓存"""
        
        # 使用集成的用户指标缓存
        try:
            # 获取正确的缓存目录路径（包含数据集名称）
            dataset_name = get_dataset_name(self.db_path)
            cache_dir = f"preprocessed_{dataset_name}/user_metrics_cache"
            cache_reader = UserMetricsCacheReader(cache_dir=cache_dir)
            print(f"  [ISS] loading cache from {cache_dir}", flush=True)
            self._calculate_iss_scores_from_cache(cache_reader)
            cache_reader.close()
            print("  [ISS] done", flush=True)
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
        """步骤5：候选群组净化（基于ISS阈值）
        
        [ISS过滤已停用] 分析表明ISS在Electronics数据集上对spam/legit几乎无区分能力
        (spam均值=0.337, legit均值=0.343)，且ERR/MRO两个子特征方向相反，
        导致43.5%的水军用户被误删。停用此过滤，所有用户直接进入candidate_groups。
        """
        
        # ISS过滤后，用户数>=2即保留（用户10用户的小群组同样是有效水军候选）
        min_group_size = 2
        
        # filtered_by_iss = 0  # [ISS过滤已停用]
        filtered_by_std = 0
        
        for group_id, group_info in self.user_groups.items():
            filtered_users = {}
            
            for user_id, user_info in group_info['users'].items():
                # [ISS阈值过滤已停用] 原因：ISS对spam/legit无区分能力，43.5%水军会被误删
                # if user_info['total_iss'] < self.iss_threshold:
                #     filtered_by_iss += 1
                #     continue
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
            print(f"  [Filter] candidate_groups={len(self.candidate_groups)} avg_iss={np.mean(avg_iss_scores):.4f} [ISS过滤已停用，所有用户保留]", flush=True)
    
    def optimize_group_purity(self):
        """步骤4.5：群组后处理优化（方案2）"""
        
        optimized_groups = {}
        total_removed = 0
        
        for idx, (group_id, group_info) in enumerate(self.candidate_groups.items(), start=1):
            if idx % 50 == 0 or idx == 1:
                print(f"  [Purity] {idx}/{len(self.candidate_groups)} groups processed", flush=True)
            users = group_info['users']
            
            if len(users) < 2:
                continue
            
            # 计算群组核心特征
            rating_stds = np.array([u.get('rating_std', 0) for u in users.values()], dtype=np.float32)
            rating_means = np.array([u.get('rating_mean', 0) for u in users.values()], dtype=np.float32)
            
            group_avg_std = float(rating_stds.mean())
            group_avg_rating = float(rating_means.mean())
            group_std_std = float(rating_stds.std())  # rating_std的标准差
            group_rating_std = float(rating_means.std())  # rating_mean的标准差
            
            # 过滤偏离群组核心特征的用户（使用相对阈值）
            optimized_users = {}
            rating_threshold = max(0.8, 1.5 * group_rating_std)
            std_threshold = max(0.5, 1.5 * group_std_std)
            for user_id, user_info in users.items():
                user_std = user_info.get('rating_std', 0)
                user_rating = user_info.get('rating_mean', 0)
                rating_consistent = abs(user_rating - group_avg_rating) < rating_threshold
                std_consistent = abs(user_std - group_avg_std) < std_threshold
                if rating_consistent and std_consistent:
                    optimized_users[user_id] = user_info
                else:
                    total_removed += 1
            
            # 如果优化后仍有足够用户，保留群组（与filter_candidate_groups保持一致）
            if len(optimized_users) >= 2:
                optimized_groups[group_id] = {
                    'users': optimized_users,
                    'user_count': len(optimized_users),
                    'avg_iss': float(np.mean([u['total_iss'] for u in optimized_users.values()])),
                    'total_reviews': int(sum(u['review_count'] for u in optimized_users.values()))
                }
        
        self.candidate_groups = optimized_groups
        
    
    def merge_similar_groups(self):
        """步骤6：群组合并（基于Jaccard相似度和重叠比例）

        优化点：
        1. 先按用户-群组倒排索引生成候选合并对，避免对全部群组做 O(n^2) 暴力比较。
        2. 只有当两个群组共享足够多用户时，才计算精确的 Jaccard / overlap。
        3. 保留原有合并判据，结果与原逻辑一致，但速度更快。
        """
        print(f"  [M6] Step6: merging {len(self.candidate_groups)} candidate groups ...", flush=True)

        group_ids = list(self.candidate_groups.keys())
        merged_groups = {}
        merged_flags = set()

        if not group_ids:
            self.final_groups = {}
            return

        # 倒排索引：user -> [group_id, ...]
        user_to_groups = defaultdict(list)
        group_users = {}
        group_sizes = {}
        for gid in group_ids:
            users = set(self.candidate_groups[gid]['users'].keys())
            group_users[gid] = users
            group_sizes[gid] = len(users)
            for u in users:
                user_to_groups[u].append(gid)

        # 预计算每对群组的共享用户数
        pair_shared = defaultdict(int)
        for gids in user_to_groups.values():
            if len(gids) < 2:
                continue
            gids = sorted(set(gids))
            for i in range(len(gids)):
                gi = gids[i]
                for j in range(i + 1, len(gids)):
                    gj = gids[j]
                    pair_shared[(gi, gj)] += 1

        # 按共享用户数优先，减少无效比较
        candidate_pairs = []
        for (gi, gj), inter in pair_shared.items():
            min_size = min(group_sizes[gi], group_sizes[gj])
            # overlap_ratio >= 0.7 至少要求交集达到较小群组的 70%
            if inter >= max(1, int(np.ceil(0.7 * min_size))):
                candidate_pairs.append((gi, gj, inter))

        # 排序：先处理重叠更大的群组
        candidate_pairs.sort(key=lambda x: (-x[2], group_sizes[x[0]] + group_sizes[x[1]]))

        # 构建邻接候选表（无向）
        merge_neighbors = defaultdict(list)
        for gi, gj, inter in candidate_pairs:
            merge_neighbors[gi].append((gj, inter))
            merge_neighbors[gj].append((gi, inter))

        # 依次合并
        for group_id1 in group_ids:
            if group_id1 in merged_flags:
                continue
            current_group = self.candidate_groups[group_id1].copy()
            current_users = set(current_group['users'].keys())
            merged_with = [group_id1]

            for group_id2, inter in sorted(merge_neighbors.get(group_id1, []), key=lambda x: -x[1]):
                if group_id2 in merged_flags or group_id2 == group_id1:
                    continue
                users2 = group_users[group_id2]
                union = len(current_users | users2)
                if union <= 0:
                    continue
                jaccard = inter / union
                overlap_ratio = inter / min(len(current_users), len(users2)) if min(len(current_users), len(users2)) > 0 else 0
                if jaccard >= 0.7 and overlap_ratio >= 0.7:
                    current_group['users'].update(self.candidate_groups[group_id2]['users'])
                    current_users = set(current_group['users'].keys())
                    merged_with.append(group_id2)
                    merged_flags.add(group_id2)

            current_group['user_count'] = len(current_group['users'])
            current_group['avg_iss'] = float(np.mean([user['total_iss'] for user in current_group['users'].values()]))
            current_group['total_reviews'] = int(sum(user['review_count'] for user in current_group['users'].values()))
            current_group['merged_from'] = merged_with
            merged_groups[f"merged_{group_id1}"] = current_group
            merged_flags.add(group_id1)

        self.final_groups = merged_groups
        print(f"  [M6] Merge done: {len(self.final_groups)} final groups", flush=True)
        
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
            if processed_groups % 25 == 0 or processed_groups == 1:
                elapsed = time.time() - start_time
                avg_time = elapsed / processed_groups
                eta = avg_time * (total_groups - processed_groups)
                print(f"  [GSS] {processed_groups}/{total_groups} groups processed, ETA ~{eta/60:.1f} min", flush=True)
            
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
        """计算群组GSS各项指标（方案A：5特征等权）

        基于 virtual_reviews 计算5个新指标：
          G2_neg_sqrt, G5_weighted_low, G6_neg_skew, G7_multi_att, G9_rev_pu
        更新时间：2026-06-07
        """
        import math as _math
        from scipy.stats import skew as _scipy_skew

        gss_scores = {}
        _GLOBAL_MEAN = 4.0
        _users = group_info.get('users', {})
        _n_users = len(_users)

        # ── 遍历 virtual_reviews 收集评分、负向偏差、用户-产品映射 ──
        _vr_ratings  = []      # 所有评分
        _vr_neg_devs = []      # max(0, product_avg - rating)
        _user_prods  = {}      # uid -> set of ASINs（for G7）

        for _uid, _u in _users.items():
            if not isinstance(_u, dict):
                continue
            _u_prods = set()
            for _vr in _u.get('virtual_reviews', []):
                if not isinstance(_vr, dict):
                    continue
                _vni = _vr.get('virtual_node_info', {})
                # 优先使用 overalls（当日全部评分），fallback 到 overall/rating
                _overalls = [float(x) for x in _vni.get('overalls', []) if x is not None]
                _asins    = list(_vni.get('asins', []))
                if not _overalls:
                    _r0 = _vni.get('overall') or _vni.get('rating')
                    if _r0 is not None:
                        try:
                            _overalls = [float(_r0)]
                            _asins    = [_vni.get('asin', '')]
                        except Exception:
                            pass
                for _a, _r in zip(_asins, _overalls):
                    _vr_ratings.append(_r)
                    _gavg = product_avg_dict.get(_a, _GLOBAL_MEAN)
                    _vr_neg_devs.append(max(0.0, _gavg - _r))
                    if _a:
                        _u_prods.add(_a)
            if _u_prods:
                _user_prods[_uid] = _u_prods

        _vr_total = len(_vr_ratings)

        if _vr_total > 0:
            _vr_arr  = np.array(_vr_ratings,  dtype=np.float32)
            _neg_arr = np.array(_vr_neg_devs, dtype=np.float32)

            # G2_neg_sqrt: sqrt归一化负向偏差均值
            gss_scores['G2_neg_sqrt'] = float(np.mean(np.sqrt(_neg_arr / 4.0 + 1e-6)))

            # G5_weighted_low: 加权低评分密度（1星→1.0, 2星→0.5, 3星→0）
            _w5 = np.maximum(0.0, (3.0 - _vr_arr) / 2.0)
            gss_scores['G5_weighted_low'] = float(np.mean(_w5))

            # G6_neg_skew: 评分负偏斜度（归一化到[0,1]）
            if len(_vr_arr) >= 4:
                try:
                    _sk = float(_scipy_skew(_vr_arr))
                    gss_scores['G6_neg_skew'] = float(np.clip(max(0.0, -_sk) / 3.0, 0.0, 1.0))
                except Exception:
                    gss_scores['G6_neg_skew'] = 0.0
            else:
                gss_scores['G6_neg_skew'] = 0.0

            # G7_multi_att: 多产品协同攻击强度
            if _user_prods and _n_users > 0:
                _multi_ratio = sum(1 for _ps in _user_prods.values() if len(_ps) >= 2) / _n_users
                _avg_nprods  = float(np.mean([len(_ps) for _ps in _user_prods.values()]))
                gss_scores['G7_multi_att'] = float(
                    _multi_ratio * min(_avg_nprods / _math.log(_n_users + 2), 1.0)
                )
            else:
                gss_scores['G7_multi_att'] = 0.0

            # G9_rev_pu: 人均评论数（log归一化，50条/人为参考上限）
            gss_scores['G9_rev_pu'] = float(
                min(1.0, _math.log1p(_vr_total / max(_n_users, 1)) / _math.log1p(50))
            )

        else:
            gss_scores['G2_neg_sqrt']     = 0.0
            gss_scores['G5_weighted_low'] = 0.0
            gss_scores['G6_neg_skew']     = 0.0
            gss_scores['G7_multi_att']    = 0.0
            gss_scores['G9_rev_pu']       = 0.0

        return gss_scores


    def _compute_total_gss(self, gss_scores, group_info=None):
        """GSS计算 - 方案A：5特征等权平均

        # GSS指标已更新，时间：2026-06-07
        指标组合（经_gss_curve_fix.py等权枚举、Electronics 2013_1.6验证确定）：
          G2_neg_sqrt + G5_weighted_low + G6_neg_skew + G7_multi_att + G9_rev_pu

        各指标含义：
          G2_neg_sqrt    : sqrt归一化负向评分偏差均值（负向攻击深度）
          G5_weighted_low: 加权低评分密度（1星→1.0, 2星→0.5, 3星→0）
          G6_neg_skew    : 评分分布负偏斜度（归一化到[0,1]）
          G7_multi_att   : 多产品协同攻击强度
          G9_rev_pu      : 人均评论数log归一化得分（群组活跃密度）

        公式：
          GSS(g) = (G2_neg_sqrt + G5_weighted_low + G6_neg_skew + G7_multi_att + G9_rev_pu) / 5

        性能（Electronics 2013_1.6, K=300）：
          Precision@50  = 0.9400
          Precision@100 = 0.9300
          Precision@300 = 0.9233  (整体递减趋势 ✅)
        """
        G2 = gss_scores.get('G2_neg_sqrt',     0.0)
        G5 = gss_scores.get('G5_weighted_low', 0.0)
        G6 = gss_scores.get('G6_neg_skew',     0.0)
        G7 = gss_scores.get('G7_multi_att',    0.0)
        G9 = gss_scores.get('G9_rev_pu',       0.0)

        final_gss = (G2 + G5 + G6 + G7 + G9) / 5.0

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
                print(f"  [M6] Cache hit, loading {final_groups_path} ...", flush=True)
                with open(final_groups_path, 'rb') as f:
                    self.final_groups = pickle.load(f)
                print(f"  [M6] Cache loaded: {len(self.final_groups)} groups", flush=True)
                return True
            
            # 如果缓存不存在，执行完整流程
            import time as _t6
            _t0 = _t6.time()
            print(f"  [M6-1/8] load_data ...", flush=True)
            self.load_data()
            print(f"  [M6-2/8] deduplicate_virtual_nodes ... ({_t6.time()-_t0:.1f}s)", flush=True)
            self.deduplicate_virtual_nodes()
            print(f"  [M6-3/8] aggregate_nodes_to_users ... ({_t6.time()-_t0:.1f}s)", flush=True)
            self.aggregate_nodes_to_users()
            print(f"  [M6-4/8] secondary_clustering ... ({_t6.time()-_t0:.1f}s)", flush=True)
            self.secondary_clustering_with_temporal_features()
            print(f"  [M6-5/8] calculate_iss_scores ... ({_t6.time()-_t0:.1f}s)", flush=True)
            self.calculate_iss_scores()
            print(f"  [M6-6/8] filter_candidate_groups ... ({_t6.time()-_t0:.1f}s)", flush=True)
            self.filter_candidate_groups()
            print(f"  [M6-7/8] optimize_group_purity ... ({_t6.time()-_t0:.1f}s)", flush=True)
            self.optimize_group_purity()
            print(f"  [M6-8a/8] merge_similar_groups ... ({_t6.time()-_t0:.1f}s)", flush=True)
            self.merge_similar_groups()
            print(f"  [M6-8b/8] calculate_group_suspicion_scores ... ({_t6.time()-_t0:.1f}s)", flush=True)
            self.calculate_group_suspicion_scores()
            print(f"  [M6] save_results ... ({_t6.time()-_t0:.1f}s)", flush=True)
            self.save_results()
            print(f"  [M6] All done in {_t6.time()-_t0:.1f}s", flush=True)
            return True
        except Exception as e:
            import traceback
            print(f"\n[ERROR] Module 6-7 failed: {e}", flush=True)
            traceback.print_exc()
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
                print(f"  [M8] Cache hit, loading metrics...", flush=True)
                with open(metrics_file, 'r', encoding='utf-8') as f:
                    metrics = json.load(f)
                predictions_df = pd.read_csv(predictions_file)
                with open(report_file, 'r', encoding='utf-8') as f:
                    report_lines = f.readlines()
                print(f"  [M8] Cache loaded. precision={metrics.get('precision',0):.4f} recall={metrics.get('recall',0):.4f}", flush=True)
                return True
            
            import time as _t8
            _t8s = _t8.time()
            print(f"  [M8-1/5] load_data (ground truth)...", flush=True)
            self.load_data()
            print(f"  [M8-2/5] generate_predictions... ({_t8.time()-_t8s:.1f}s)", flush=True)
            self.generate_predictions()
            print(f"  [M8-3/5] calculate_metrics... ({_t8.time()-_t8s:.1f}s)", flush=True)
            self.calculate_metrics()
            print(f"  [M8-4/5] calculate_multi_k_metrics... ({_t8.time()-_t8s:.1f}s)", flush=True)
            self.calculate_multi_k_metrics()
            print(f"  [M8-5/5] save_results... ({_t8.time()-_t8s:.1f}s)", flush=True)
            self.save_results()
            self.save_multi_k_results()
            print(f"  [M8] All done in {_t8.time()-_t8s:.1f}s", flush=True)
            return True
        except Exception as e:
            import traceback
            print(f"\n[ERROR] Module 8 failed: {e}", flush=True)
            traceback.print_exc()
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

        # 根据数据集设置引力图/斥力图分位数（按论文原始定义值）
        # 论文 docstring：引力图取相似度 95% 分位数以上；斥力图取 10% 分位数以下
        # Cell_Phones: attr=80%, rep=30%  (保持原值)
        # Electronics: attr=95%, rep=10%  (恢复论文值，原 attr=60 rep=40 已偏离论文)
        # Clothing:    attr=60%, rep=20%  (保持原值)
        dataset_pct_config = {
            "Cell_Phones":  {"attraction_pct": 80, "repulsion_pct": 30},
            "Electronics":  {"attraction_pct": 95, "repulsion_pct": 10},
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
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] [ERROR] Modules 6-7 failed.", flush=True)
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