import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# ============================================
# 1. 讀取資料（保留連續速度方向信息）
# ============================================
df = pd.read_csv("all_data.csv")

# 確認欄位順序 (根據原始程式碼推斷)
# 0: posX, 1: posY, 2: posZ,
# 3: dirX, 4: dirY, 5: dirZ, 6: speed,
# 7: landX, 8: (landY 未使用), 9: landZ
X = df.iloc[:, [0, 1, 2, 3, 4, 5, 6]].values   # 完整初始狀態 (7個特徵)
y = df.iloc[:, [7, 9]].values               # 目標: landX, landZ
# 訓練時使用單位方向向量 + 速度大小，以讓方向不受速度大小影響

# ============================================
# 2. 分割訓練/測試集
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================
# 3. 標準化 (樹模型不一定需要，但保持一致)
# ============================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================
# 4. 建立並訓練 RandomForest 回歸器 (多輸出)
# ============================================
model = RandomForestRegressor(
    n_estimators=50,
    max_depth=20,
    min_samples_leaf=10,
    n_jobs=-1,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# ============================================
# 5. 預測與評估
# ============================================
y_pred = model.predict(X_test_scaled)

# 分別計算兩個目標的指標
mae_x = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
r2_x = r2_score(y_test[:, 0], y_pred[:, 0])
mae_z = mean_absolute_error(y_test[:, 1], y_pred[:, 1])
r2_z = r2_score(y_test[:, 1], y_pred[:, 1])

print("=== 落點預測模型評估 (輸入: posX, posY, posZ, dirX, dirY, dirZ, speed) ===")
print(f"LandX - MAE: {mae_x:.6f}, R²: {r2_x:.6f}")
print(f"LandZ - MAE: {mae_z:.6f}, R²: {r2_z:.6f}")

# ============================================
# 6. 儲存模型與標準化器
# ============================================
joblib.dump(model, "knn_model.pkl", compress=3)
joblib.dump(scaler, "scaler.pkl", compress=3)

print("\n[完成] 模型與標準化工具已儲存為 knn_model.pkl 與 scaler.pkl")