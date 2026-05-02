import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# ============================================
# 1. 讀取資料（直接使用原始速度值，不進行二值化）
# ============================================
df = pd.read_csv("all_data.csv")

# 確認欄位順序 (根據原始程式碼推斷)
# 0: posX, 1: posY, 2: posZ,
# 3: velX, 4: velY, 5: velZ,
# 6: landX, 7: (landY 未使用), 8: landZ
X = df.iloc[:, [0, 1, 2, 3, 4, 5]].values   # 完整初始狀態 (6個特徵)
y = df.iloc[:, [6, 8]].values               # 目標: landX, landZ

# ============================================
# 2. 分割訓練/測試集
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================
# 3. 標準化 (KNN 必須)
# ============================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================
# 4. 建立並訓練 KNN 回歸器 (多輸出)
# ============================================
knn = KNeighborsRegressor(n_neighbors=1)
knn.fit(X_train_scaled, y_train)

# ============================================
# 5. 預測與評估
# ============================================
y_pred = knn.predict(X_test_scaled)

# 分別計算兩個目標的指標
mae_x = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
r2_x = r2_score(y_test[:, 0], y_pred[:, 0])
mae_z = mean_absolute_error(y_test[:, 1], y_pred[:, 1])
r2_z = r2_score(y_test[:, 1], y_pred[:, 1])

print("=== 落點預測模型評估 (輸入: posX, posY, posZ, velX, velY, velZ) ===")
print(f"LandX - MAE: {mae_x:.6f}, R²: {r2_x:.6f}")
print(f"LandZ - MAE: {mae_z:.6f}, R²: {r2_z:.6f}")

# ============================================
# 6. 儲存模型與標準化器
# ============================================
joblib.dump(knn, "knn_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n[完成] 模型與標準化工具已儲存為 knn_model.pkl 與 scaler.pkl")