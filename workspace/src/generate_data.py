import numpy as np
import pandas as pd
import math

# Parameters from main.py
x_boundary_min = -0.1494
x_boundary_max = 0.1494
z_boundary_min = -0.1182
z_boundary_max = 0.1182
ball_radius = 0.02
bottom_board_y = 0.042
top_board_y = 0.312
vel_magnitude = 0.057735

x_min_eff = x_boundary_min + ball_radius
x_max_eff = x_boundary_max - ball_radius
z_min_eff = z_boundary_min + ball_radius
z_max_eff = z_boundary_max - ball_radius

def predict_landing_with_walls_and_radius(posX, posY, posZ, velX, velY, velZ, target_y,
                                          x_boundary_min, x_boundary_max,
                                          z_boundary_min, z_boundary_max,
                                          ball_radius=0.02):
    if abs(velY) < 1e-6:
        return (None, None, None)
    t = (target_y - posY) / velY
    if t <= 0:
        return (None, None, None)

    x_min_eff = x_boundary_min + ball_radius
    x_max_eff = x_boundary_max - ball_radius
    z_min_eff = z_boundary_min + ball_radius
    z_max_eff = z_boundary_max - ball_radius

    def reflect_1d(p, v, low, high, dt):
        span = high - low
        if span <= 0:
            return p + v * dt
        p_abs = p + v * dt
        offset = p_abs - low
        if offset >= 0:
            cycles = int(offset / span)
        else:
            cycles = -int((-offset) / span) - 1 if (offset % span) != 0 else -int((-offset) / span)
        remainder = offset - cycles * span
        if cycles % 2 == 0:
            return low + remainder
        else:
            return high - remainder

    landX = reflect_1d(posX, velX, x_min_eff, x_max_eff, t)
    landZ = reflect_1d(posZ, velZ, z_min_eff, z_max_eff, t)
    return (landX, target_y, landZ)

# Generate data
data = []

# Sample positions - 增加資料量以提高覆蓋
posX_samples = np.linspace(x_min_eff, x_max_eff,14)
posY_samples = np.linspace(0.055, 0.33, 11)
posZ_samples = np.linspace(z_min_eff, z_max_eff, 12)

print("開始生成數據...")
print("正在生成所有可能的速度方向和速度大小...")

# 生成所有可能的速度方向（球面均勻採樣）
# 使用 Fibonacci sphere 來生成均勻分佈的方向
def fibonacci_sphere(samples=200):
    """生成球面上均勻分佈的點"""
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians
    
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y
        
        theta = phi * i  # golden angle increment
        
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append((x, y, z))
    
    return np.array(points)

# 生成300個均勻分佈的方向
directions = fibonacci_sphere(samples=300)

# 速度大小範圍：保留方向並覆蓋不同速度
speed_values = np.linspace(0.06, 0.30, 7)

direction_stats = {}

for posX in posX_samples:
    for posY in posY_samples:
        for posZ in posZ_samples:
            for dir_idx, (dir_x, dir_y, dir_z) in enumerate(directions):
                # 方向已經是單位向量
                dir_norm = np.sqrt(dir_x**2 + dir_y**2 + dir_z**2)
                if dir_norm == 0:
                    continue
                dir_x /= dir_norm
                dir_y /= dir_norm
                dir_z /= dir_norm

                for speed in speed_values:
                    velX = dir_x * speed
                    velY = dir_y * speed
                    velZ = dir_z * speed

                    # 決定目標 y (基於 vy 的符號)
                    if velY > 1e-6:
                        target_y = top_board_y
                    elif velY < -1e-6:
                        target_y = bottom_board_y
                    else:
                        continue  # 跳過水平速度

                    result = predict_landing_with_walls_and_radius(
                        posX, posY, posZ, velX, velY, velZ, target_y,
                        x_boundary_min, x_boundary_max,
                        z_boundary_min, z_boundary_max,
                        ball_radius
                    )

                    if result[0] is not None:
                        landX, landY, landZ = result
                        data.append([posX, posY, posZ, dir_x, dir_y, dir_z, speed, landX, landY, landZ])
                        
                        # 統計方向
                        key = (round(dir_x, 3), round(dir_y, 3), round(dir_z, 3))
                        direction_stats[key] = direction_stats.get(key, 0) + 1

print(f"\n方向多樣性統計:")
print(f"  總方向數: {len(direction_stats)}")
print(f"  總數據點數: {len(data)}")

df = pd.DataFrame(data, columns=['posX', 'posY', 'posZ', 'dirX', 'dirY', 'dirZ', 'speed', 'landX', 'landY', 'landZ'])
df.to_csv('all_data.csv', index=False)
print(f"\n✓ 已生成 {len(data)} 筆數據點，包含 {len(direction_stats)} 種不同方向")