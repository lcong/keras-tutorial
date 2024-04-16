"""
2.1 定义：
分类技术通过分析输入数据的特征，将其分配到两个或多个预定义的类别中。不同于回归模型预测连续数值，分类模型的输出是类别标签。
2.2 江湖分类：
在武侠的世界中，大侠和败类的故事层出不穷。
假设我们手头有一批关于武侠人物的数据，包括他们的武功、行侠仗义的次数、以及是否拥有绝世武功等特征，我们的任务是根据这些特征将他们分类为英雄或恶人。
"""

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

plt.rc("font", family="YouYuan")

# 生成虚拟的武侠人物数据
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# 可视化生成的数据
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.title("武侠人物数据分布")
plt.xlabel("特征 1")
plt.ylabel("特征 2")
plt.show()

# 应用 K-均值聚类
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# 可视化聚类结果
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap="viridis")
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c="red", s=200, alpha=0.5)
plt.title("武侠人物聚类结果")
plt.xlabel("特征 1")
plt.ylabel("特征 2")
plt.show()
