# 導入必要的 Python 庫
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# 載入資料
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# 簡單的資料預處理
# 填補年齡的缺失值，以年齡的中位數代替
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
test_df['Age'].fillna(test_df['Age'].median(), inplace=True)

# 填補票價的缺失值，以票價的中位數代替
test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)

# 填補船艙登船口的缺失值，以最常出現的值代替
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)
test_df['Embarked'].fillna(test_df['Embarked'].mode()[0], inplace=True)

# 把類別型變量 (Sex 和 Embarked) 轉換為數值
train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1})
train_df['Embarked'] = train_df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
test_df['Embarked'] = test_df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# 選擇有用的特徵
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = train_df[features]
y = train_df['Survived']
X_test = test_df[features]

# 分割訓練與測試集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立隨機森林模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 預測驗證集
y_pred = model.predict(X_val)

# 計算混淆矩陣和準確度
conf_matrix = confusion_matrix(y_val, y_pred)
accuracy = accuracy_score(y_val, y_pred)

print("Confusion Matrix:")
print(conf_matrix)
print("\nAccuracy:", accuracy)

# 生成測試集預測
y_test_pred = model.predict(X_test)

# 創建 submission DataFrame
output = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': y_test_pred})

# 將結果寫入 CSV 文件
output.to_csv("output.csv", index=False)
print("預測結果已保存至 output.csv")
