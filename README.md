<<<<<<< HEAD
# 机器学习实验：基于 Word2Vec 的情感预测

## 1. 学生信息

- **姓名**：余姝舒
- **学号**：112304260107
- **班级**：数据1231

> 注意：姓名和学号必须填写，否则本次实验提交无效。

***

## 2. 实验任务

本实验基于给定文本数据，使用 **Word2Vec 将文本转为向量特征**，再结合 **分类模型** 完成情感预测任务，并将结果提交到 Kaggle 平台进行评分。

本实验重点包括：

- 文本预处理
- Word2Vec 词向量训练或加载
- 句子向量表示
- 分类模型训练
- Kaggle 结果提交与分析

***

## 3. 比赛与提交信息

- **比赛名称**：Bag of Words Meets Bags of Popcorn
- **比赛链接**：<https://www.kaggle.com/competitions/word2vec-nlp-tutorial/overview>
- **提交日期**：2026-04-15
- **GitHub 仓库地址**：<https://github.com/yss-yes/yushushu-112304260107-shiyan2>
- **GitHub README 地址**：<https://github.com/yss-yes/yushushu-112304260107-shiyan2/blob/main/README.md>

> 注意：GitHub 仓库首页或 README 页面中，必须能看到“姓名 + 学号”，否则无效。

***

## 4. Kaggle 成绩

请填写你最终提交到 Kaggle 的结果：

- **Public Score**：0.95589
- **Private Score**（如有）：
- **排名**（如能看到可填写）：

***

## 5. Kaggle 截图

请在下方插入 Kaggle 提交结果截图，要求能清楚看到分数信息。

![kaggle截图](112304260107_余姝舒_kaggle_score.png)

> 建议将截图保存在 `images` 文件夹中。
> 截图文件名示例：`2023123456_张三_kaggle_score.png`

***

## 6. 实验方法说明

### （1）文本预处理

请说明你对文本做了哪些处理，例如：

- 分词
- 去停用词
- 去除标点或特殊符号
- 转小写

**我的做法：**

- 去除 HTML 标签：使用正则表达式去除评论中的 HTML 标签，如 `<br />` 等
- 移除非字母字符：只保留英文，去除其他字符
- 转换为小写：将所有文本转换为小写
- 分词：使用 NLTK 进行分词处理

***

### （2）Word2Vec 特征表示

请说明你如何使用 Word2Vec，例如：

- 是自己训练 Word2Vec，还是使用已有模型
- 词向量维度是多少
- 句子向量如何得到（平均、加权平均、池化等）

**我的做法：**

- 自己训练 Word2Vec 模型
- 词向量维度：300
- 句子向量表示：每个评论的词向量求均值（均值 embedding）
- Word2Vec 参数：window=5, min\_count=40, workers=4, random\_state=42

***

### （3）分类模型

请说明你使用了什么分类模型，例如：

- Logistic Regression
- Random Forest
- SVM
- XGBoost

并说明最终采用了哪一个模型。

**我的做法：**

- 使用逻辑回归（Logistic Regression）模型
- 模型参数：C=10, penalty='l2', solver='liblinear', max\_iter=1000, random\_state=42
- 最终采用的模型：逻辑回归

***

## 7. 实验流程

请简要说明你的实验流程。

示例：

1. 读取训练集和测试集
2. 对文本进行预处理
3. 训练或加载 Word2Vec 模型
4. 将每条文本表示为句向量
5. 用训练集训练分类器
6. 在测试集上预测结果
7. 生成 submission 文件并提交 Kaggle

**我的实验流程：**

1. 读取训练集和测试集数据
2. 对文本进行预处理：去除 HTML 标签、移除非字母字符、转换为小写、分词
3. 基于训练集评论训练 Word2Vec 模型
4. 计算每条评论的词向量均值作为特征
5. 使用逻辑回归模型进行训练
6. 对测试集进行预测，生成概率值
7. 生成 submission.csv 文件并提交到 Kaggle

***

## 8. 文件说明

请说明仓库中各文件或文件夹的作用。

示例：

- `data/`：存放数据文件
- `src/`：存放源代码
- `notebooks/`：存放实验 notebook
- `images/`：存放 README 中使用的图片
- `submission/`：存放提交文件

**我的项目结构：**

```text
project/
├─ data/             # 存放数据文件
│  ├─ labeledTrainData.tsv
│  ├─ sampleSubmission.csv
│  ├─ testData.tsv
│  └─ unlabeledTrainData.tsv
├─ models/           # 存放训练好的模型
│  ├─ model.pkl         # 逻辑回归模型
│  └─ word2vec_model.pkl  # Word2Vec 模型
├─ src/              # 存放源代码
│  ├─ train.py      # 训练脚本
│  └─ predict.py    # 预测脚本
├─ images/           # 存放 README 中使用的图片
├─ submission.csv    # 提交文件
├─ README.md         # 实验报告
└─ requirements.txt  # 依赖库
```

***

## 9. 实验总结

请简要总结你的实验过程、遇到的问题及解决方案、实验结果分析等。

**我的总结：**

- 实验过程：按照老师要求的方案，使用 Word2Vec 提取文本特征，结合逻辑回归模型进行情感分析
- 遇到的问题：生成的 submission.csv 文件中 sentiment 列是概率值，不是 0 和 1 的分类结果
- 解决方案：修改代码，确保生成的是概率值格式的提交文件，符合 Kaggle 的要求
- 实验结果：最终在 Kaggle 上取得了 0.95589 的 AUC 分数，表现优异
- 技术优势：Word2Vec 能够有效捕捉词之间的语义关系，均值 embedding 方法简单且有效，逻辑回归模型在文本分类任务中表现稳定

***

## 10. 参考资料

请列出你在实验中参考的资料，例如论文、博客、文档等。

1. Kaggle 比赛：Bag of Words Meets Bags of Popcorn
2. Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient Estimation of Word Representations in Vector Space. arXiv preprint arXiv:1301.3781.
3. scikit-learn 文档：<https://scikit-learn.org/stable/documentation.html>
4. Gensim 文档：<https://radimrehurek.com/gensim/>

=======
# yushushu-112304260107-shiyan2
>>>>>>> c70bac9746a85f66b590c16bc7f30b560b23d1d3
