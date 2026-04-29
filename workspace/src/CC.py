import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score


df = pd.read_csv("all_data.csv")


def binarize(v):
    return 1 if v > 0 else -1

for col in [3, 4, 5]:
    df.iloc[:, col] = df.iloc[:, col].apply(binarize)

def encode_direction(vx, vy, vz):
    if vx == 1 and vy == 1 and vz == 1:
        return 1
    elif vx == 1 and vy == 1 and vz == -1:
        return 2
    elif vx == 1 and vy == -1 and vz == 1:
        return 3
    elif vx == -1 and vy == 1 and vz == 1:
        return 4
    elif vx == -1 and vy == -1 and vz == 1:
        return 5
    elif vx == -1 and vy == 1 and vz == -1:
        return 6
    elif vx == 1 and vy == -1 and vz == -1:
        return 7
    elif vx == -1 and vy == -1 and vz == -1:
        return 8
    else:
        return 0

df["dir_class"] = df.apply(
    lambda row: encode_direction(row[3], row[4], row[5]), axis=1
)


X = df.iloc[:, :3].values             
dir_feat = df["dir_class"].values.reshape(-1, 1)
X = np.hstack([X, dir_feat])               

# ===============================
y_x = df.iloc[:, 6].values   
y_z = df.iloc[:, 8].values   


X_train, X_test, y_x_train, y_x_test = train_test_split(
    X, y_x, test_size=0.2, random_state=42
)
_, _, y_z_train, y_z_test = train_test_split(
    X, y_z, test_size=0.2, random_state=42
)


k = 1
knn_x = KNeighborsRegressor(n_neighbors=k)
knn_z = KNeighborsRegressor(n_neighbors=k)

knn_x.fit(X_train, y_x_train)
knn_z.fit(X_train, y_z_train)

y_x_pred = knn_x.predict(X_test)
y_z_pred = knn_z.predict(X_test)

print(f"MAE: {mean_absolute_error(y_x_test, y_x_pred):.4f}")
print(f"R² : {r2_score(y_x_test, y_x_pred):.4f}")


print(f"MAE: {mean_absolute_error(y_z_test, y_z_pred):.4f}")
print(f"R² : {r2_score(y_z_test, y_z_pred):.4f}")

joblib.dump(knn_x, "knn_model_x.pkl")
joblib.dump(knn_z, "knn_model_z.pkl")
print("\n模型已保存为 knn_model_x.pkl 和 knn_model_z.pkl")