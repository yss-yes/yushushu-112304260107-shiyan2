import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
import joblib
import os

# 确保 nltk 分词器已下载
try:
    word_tokenize('test')
except:
    nltk.download('punkt')

# 定义数据路径
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

# 确保模型目录存在
os.makedirs(model_dir, exist_ok=True)

# 读取训练数据
def load_data():
    """加载训练数据"""
    train_path = os.path.join(data_dir, 'labeledTrainData.tsv')
    train_df = pd.read_csv(train_path, sep='\t', quoting=3)
    test_path = os.path.join(data_dir, 'testData.tsv')
    test_df = pd.read_csv(test_path, sep='\t', quoting=3)
    return train_df, test_df

# 文本预处理函数
def preprocess_text(text):
    """
    预处理文本：去除HTML标签、移除非字母字符、转换为小写、分词
    """
    # 去除HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    # 移除非字母字符，只保留英文
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # 转换为小写
    text = text.lower()
    # 分词
    words = word_tokenize(text)
    return words

# 训练Word2Vec模型
def train_word2vec(sentences):
    """
    训练Word2Vec模型
    """
    model = Word2Vec(
        sentences=sentences,
        vector_size=300,
        window=5,
        min_count=40,
        workers=4,
        random_state=42
    )
    return model

# 计算评论的均值embedding
def get_mean_embedding(model, words):
    """
    计算评论的均值embedding
    """
    vectors = [model.wv[word] for word in words if word in model.wv]
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

# 训练模型并生成提交文件
def train_model():
    """
    训练模型：读取数据、预处理、训练Word2Vec、提取特征、训练逻辑回归、生成提交文件
    """
    print("加载数据...")
    train_df, test_df = load_data()
    
    print("预处理文本...")
    # 对训练集评论进行预处理
    train_df['tokens'] = train_df['review'].apply(preprocess_text)
    # 对测试集评论进行预处理
    test_df['tokens'] = test_df['review'].apply(preprocess_text)
    
    print("训练Word2Vec模型...")
    # 基于训练集评论分词结果训练Word2Vec
    word2vec_model = train_word2vec(train_df['tokens'].tolist())
    
    print("提取特征...")
    # 提取训练集特征（均值embedding）
    X_train = np.array([get_mean_embedding(word2vec_model, tokens) for tokens in train_df['tokens']])
    y_train = train_df['sentiment']
    
    # 提取测试集特征（均值embedding）
    X_test = np.array([get_mean_embedding(word2vec_model, tokens) for tokens in test_df['tokens']])
    
    print("训练逻辑回归模型...")
    # 使用逻辑回归模型
    model = LogisticRegression(
        C=10,
        penalty='l2',
        solver='liblinear',
        max_iter=1000,
        random_state=42
    )
    
    # 5折交叉验证AUC分数
    cv_auc = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc').mean()
    print(f"5折交叉验证AUC分数: {cv_auc:.4f}")
    
    # 训练模型
    model.fit(X_train, y_train)
    
    # 计算训练集AUC
    y_pred_proba = model.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, y_pred_proba)
    print(f"训练集AUC: {train_auc:.4f}")
    
    # 将结果写入文件
    with open('auc_results.txt', 'w', encoding='utf-8') as f:
        f.write(f"5折交叉验证AUC分数: {cv_auc:.4f}\n")
        f.write(f"训练集AUC: {train_auc:.4f}\n")
    print("AUC结果已写入到 auc_results.txt 文件")
    
    print("保存模型...")
    # 保存模型和Word2Vec模型
    model_path = os.path.join(model_dir, 'model.pkl')
    word2vec_path = os.path.join(model_dir, 'word2vec_model.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(word2vec_model, word2vec_path)
    
    print(f"模型保存成功: {model_path}")
    print(f"Word2Vec模型保存成功: {word2vec_path}")
    
    print("生成提交文件...")
    # 对测试集进行预测，获取0或1的分类结果
    y_pred = model.predict(X_test)
    
    # 处理测试集id列，去除多余的引号
    test_df['id'] = test_df['id'].apply(lambda x: x.strip('"'))
    
    # 生成提交文件
    submission = pd.DataFrame({
        'id': test_df['id'],
        'sentiment': y_pred
    })
    
    # 生成新的提交文件
    submission_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'submission_new.csv')
    
    # 直接写入文件，避免pandas的引号处理
    with open(submission_path, 'w', encoding='utf-8', newline='') as f:
        # 写入表头
        f.write('id,sentiment\n')
        # 写入数据
        for _, row in submission.iterrows():
            f.write(f"{row['id']},{row['sentiment']}\n")
    
    print(f"提交文件生成成功: {submission_path}")
    print(f"预测完成，共预测 {len(submission)} 条评论")

if __name__ == "__main__":
    train_model()
