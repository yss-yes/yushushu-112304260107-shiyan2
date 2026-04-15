import pandas as pd
import numpy as np
import re
import joblib
import os

# 定义数据路径
data_dir = 'data'
model_dir = 'models'

# 文本预处理函数（与训练时相同）
def preprocess_text(text):
    """
    预处理文本：去除HTML标签、标点符号，转为小写
    """
    # 去除HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    # 去除非字母字符
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # 转为小写
    text = text.lower()
    # 重新组合为文本
    return text

# 预测并生成提交文件
def predict():
    """
    加载模型、预测测试数据、生成 submission.csv 文件
    """
    print("加载模型...")
    # 加载训练好的模型和向量izer
    model_path = os.path.join(model_dir, 'model.pkl')
    vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        print("错误：模型文件不存在，请先运行 train.py 训练模型")
        return
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    print("加载测试数据...")
    # 读取测试数据
    test_path = os.path.join(data_dir, 'testData.tsv')
    test_df = pd.read_csv(test_path, sep='\t', quoting=3)
    
    # 处理 id 列，去除多余的引号
    test_df['id'] = test_df['id'].apply(lambda x: x.strip('"'))
    
    print("预处理文本...")
    # 对测试评论进行预处理
    test_df['clean_review'] = test_df['review'].apply(preprocess_text)
    
    print("提取特征...")
    # 使用训练好的向量izer提取特征
    X_test = vectorizer.transform(test_df['clean_review'])
    
    print("预测...")
    # 进行预测
    y_pred = model.predict(X_test)
    
    print("生成提交文件...")
    # 生成提交文件
    submission = pd.DataFrame({
        'id': test_df['id'],
        'sentiment': y_pred
    })
    
    print(f"提交文件形状: {submission.shape}")
    print(f"前5行数据:")
    print(submission.head())
    
    # 检查 id 列是否还有引号
    print(f"\nid 列的第一个值: {submission['id'].iloc[0]}")
    print(f"id 列的第一个值是否包含引号: {'"' in submission['id'].iloc[0]}")
    
    submission_path = 'submission.csv'
    print(f"\n准备写入文件: {submission_path}")
    
    # 直接写入文件，避免 pandas 的引号处理
    with open(submission_path, 'w', encoding='utf-8', newline='') as f:
        # 写入表头
        f.write('id,sentiment\n')
        # 写入数据
        for _, row in submission.iterrows():
            f.write(f"{row['id']},{row['sentiment']}\n")
    
    if os.path.exists(submission_path):
        print(f"提交文件生成成功: {submission_path}")
        print(f"文件大小: {os.path.getsize(submission_path)} bytes")
        # 读取生成的文件，检查内容
        with open(submission_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()[:5]
            print("\n生成的文件前5行:")
            for line in lines:
                print(repr(line.strip()))
    else:
        print(f"错误: 提交文件未生成")
    
    print(f"\n预测完成，共预测 {len(submission)} 条评论")

if __name__ == "__main__":
    try:
        predict()
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
