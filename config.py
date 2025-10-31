import numpy as np
# --- 1. 시뮬레이션 파라미터 ---

# 1.1. 시스템 파라미터
pt_dbm = 10.0
fc = 3.5e9
d_tx_rx = 50.0

# 1.2. 채널 파라미터
lt = 3
lr = 3
random_seed = 42

# 1.3. MA (Movable Antenna) 파라미터
A_normalized = 2.0
grid_points = 200

# --- 2. 물리 상수 ---
c = 3.0e8

# --- 3. 계산된 파라미터 ---
lambda_wave = c / fc
A = A_normalized * lambda_wave
pt_watts = (10**(pt_dbm / 10)) / 1000

# --- 4. 고정 위치 (로컬 좌표계 기준) ---
t_pos_fixed = np.array([0.0, 0.0])
FPA_pos_fixed = np.array([0.0, 0.0])