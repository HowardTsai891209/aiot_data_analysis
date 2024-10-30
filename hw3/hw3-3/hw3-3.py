import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import LinearSVC
import streamlit as st

# Streamlit 程序標題
st.title("SVM 3D Scatter Plot with Adjustable Parameters")

# 使用者調整參數的拉桿
distance_threshold = st.slider("Distance Threshold", min_value=0.0, max_value=20.0, value=4.0, step=0.1)
semi_major_axis = st.slider("Semi-Major Axis", min_value=1.0, max_value=20.0, value=10.0, step=0.1)
semi_minor_axis = st.slider("Semi-Minor Axis", min_value=1.0, max_value=20.0, value=5.0, step=0.1)

# Step 1: 產生 600 個隨機點
np.random.seed(0)
num_points = 600
mean = 0
variance = 10
x1 = np.random.normal(mean, np.sqrt(variance), num_points)
x2 = np.random.normal(mean, np.sqrt(variance), num_points)

# 計算到原點的距離
distances = np.sqrt(x1**2 + x2**2)

# 根據距離分配標籤 Y=0 或 Y=1
Y = np.where(distances < distance_threshold, 0, 1)

# 檢查類別數量
unique_classes = np.unique(Y)
while len(unique_classes) < 2:
    x1 = np.random.normal(mean, np.sqrt(variance), num_points)
    x2 = np.random.normal(mean, np.sqrt(variance), num_points)
    distances = np.sqrt(x1**2 + x2**2)
    Y = np.where(distances < distance_threshold, 0, 1)
    unique_classes = np.unique(Y)

# Step 2: 計算 x3 作為 x1 和 x2 的高斯函數
def gaussian_function(x1, x2):
    return np.exp(-0.1 * (x1**2 + x2**2))

x3 = gaussian_function(x1, x2)

# Step 3: 使用 LinearSVC 訓練模型找到分隔超平面
X = np.column_stack((x1, x2, x3))
clf = LinearSVC(random_state=0, max_iter=10000)
clf.fit(X, Y)
coef = clf.coef_[0]
intercept = clf.intercept_

# 創建 3D 散點圖並繪製分隔超平面
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1[Y == 0], x2[Y == 0], x3[Y == 0], c='blue', marker='o', label='Y=0')
ax.scatter(x1[Y == 1], x2[Y == 1], x3[Y == 1], c='red', marker='s', label='Y=1')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('x3')
ax.set_title('3D Scatter Plot with Y Color and Separating Hyperplane')
ax.legend()

# 創建網格以繪製分隔超平面
xx, yy = np.meshgrid(np.linspace(min(x1), max(x1), 50),
                     np.linspace(min(x2), max(x2), 50))
zz = (-coef[0] * xx - coef[1] * yy - intercept) / coef[2]
ax.plot_surface(xx, yy, zz, color='gray', alpha=0.5)

# 創建 2D 散點圖
fig2d, ax2d = plt.subplots()
ax2d.scatter(x1, x2, c=Y, cmap='bwr', alpha=0.5)
ellipse = plt.Circle((0, 0), distance_threshold, color='gray', fill=False, linestyle='--')
ax2d.add_patch(ellipse)
ax2d.set_xlabel('X1')
ax2d.set_ylabel('X2')
ax2d.set_title('2D Scatter Plot of 600 Random Points')

# 顯示 2D 和 3D 圖形
st.pyplot(fig2d)
st.pyplot(fig)
