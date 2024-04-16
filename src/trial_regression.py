"""
回归，顾名思义，是一种预测数值的技术，比如温度、价格、或者是某种指标的大小。
它试图找出输入变量（我们可以称之为特征）和输出变量（我们关心的目标数值）之间的关系。
1.2 武侠世界的房价预测：
不同地域的客栈价格差异巨大，我们将使用一个简化的线性回归模型来预测客栈的价格。假设客栈价格受到以下几个因素的影响：客栈的位置、房间的大小、是否靠近江湖名胜等。
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


plt.rc("font", family="YouYuan")

# 设置随机种子，以确保每次运行生成的数据相同
np.random.seed(42)

# 生成虚拟“江湖客栈”数据集
X = np.random.randint(
    0, 2, (100, 3)
)  # 生成0和1的随机数，模拟客栈的位置、房间大小和是否靠近名胜
print(X)

reputation = np.random.rand(100, 1) * 10  # 生成0到10之间的随机数，模拟客栈的知名度
X = np.hstack((X, reputation))  # 合并特征数组

print(X)

# 根据特征和一些随机噪声生成客栈价格
y = 200 + np.dot(X, [150, 100, 50, 20]) + np.random.randn(100) * 50

# 数据可视化：展示客栈的知名度与价格的关系
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 3], y, color="blue")  # X[:, 3] 是知名度特征
plt.title("客栈的知名度与价格关系")
plt.xlabel("知名度")
plt.ylabel("价格")
plt.grid(True)
plt.show()

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 创建并训练线性回归模型
regression_model = LinearRegression()
regression_model.fit(X_train, y_train)

# 进行预测和评估
y_pred = regression_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"模型的均方误差（MSE）: {mse:.2f}")

# 可视化预测价格与实际价格
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color="red")
plt.plot([y.min(), y.max()], [y.min(), y.max()], "k--", lw=4)
plt.xlabel("实际价格")
plt.ylabel("预测价格")
plt.title("预测价格 VS 实际价格")
plt.show()

# 可视化特征重要性
features = ["位置", "房间大小", "靠近名胜", "知名度"]
feature_importance = regression_model.coef_
plt.figure(figsize=(10, 6))
plt.bar(features, feature_importance, color="green")
plt.xlabel("特征")
plt.ylabel("重要性")
plt.title("特征对客栈价格的影响")
plt.show()
