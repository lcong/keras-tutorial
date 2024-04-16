"""
2.1 定义：
分类技术通过分析输入数据的特征，将其分配到两个或多个预定义的类别中。不同于回归模型预测连续数值，分类模型的输出是类别标签。
2.2 江湖分类：
在武侠的世界中，大侠和败类的故事层出不穷。
假设我们手头有一批关于武侠人物的数据，包括他们的武功、行侠仗义的次数、以及是否拥有绝世武功等特征，我们的任务是根据这些特征将他们分类为英雄或恶人。
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

plt.rc("font", family="YouYuan")

# 生成较大的虚拟武侠人物数据集
np.random.seed(42)  # 确保结果可复现
X, y = make_classification(
    n_samples=1000,
    n_features=4,
    n_informative=3,
    n_redundant=1,
    n_clusters_per_class=2,
    random_state=42,
)

# 特征包括：武功等级、内力深厚度、正义行为次数、邪恶行为次数
features = ["武功等级", "内力深厚度", "正义行为次数", "邪恶行为次数"]

# 可视化数据集中的两个特征：武功等级和内力深厚度
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="winter", alpha=0.6)  # 使用冬天的颜色图表示类别
plt.xlabel("武功等级")
plt.ylabel("内力深厚度")
plt.title("武侠世界人物特征分布")
plt.colorbar(label="分类")  # 显示颜色条
plt.show()

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 使用随机森林分类器
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# 预测测试集的结果
y_pred = classifier.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"模型的准确率: {accuracy:.2f}")

# 特征重要性可视化
feature_importances = classifier.feature_importances_
plt.figure(figsize=(10, 6))
plt.bar(features, feature_importances, color="purple")
plt.xlabel("特征")
plt.ylabel("重要性")
plt.title("特征对分类结果的影响")
plt.show()


# 可视化决策树（如果需要）
from sklearn import tree

plt.figure(figsize=(20, 10))
tree.plot_tree(
    classifier[0], feature_names=features, class_names=["恶人", "英雄"], filled=True
)
plt.show()
