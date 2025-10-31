import numpy as np
import config as cfg

# --- 유틸리티 함수 ---

def watts_to_dbm(watts):
    epsilon = 1e-30
    return 10 * np.log10((watts + epsilon) * 1000)

def calculate_path_loss_db(distance_m, f_c_hz):
    epsilon = 1e-9
    lambda_wave = cfg.c / f_c_hz

    path_loss_linear = ((4 * np.pi * (distance_m + epsilon)) / lambda_wave) ** 2
    path_loss_db = 10 * np.log10(path_loss_linear)
    return path_loss_db

def create_channel_environment():
    np.random.seed(cfg.random_seed)

    # 1. 송신(Tx) 측 출발 각도(AoD) 생성
    theta_t = np.random.uniform(-np.pi / 2, np.pi / 2, cfg.lt)
    phi_t = np.random.uniform(-np.pi / 2, np.pi / 2, cfg.lt)

    # 2. 수신(Rx) 측 도착 각도(AoA) 생성
    theta_r = np.random.uniform(-np.pi / 2, np.pi / 2, cfg.lr)
    phi_r = np.random.uniform(-np.pi / 2, np.pi / 2, cfg.lr)

    # 3. 경로 응답 행렬 (Sigma) 생성
    sigma_matrix = (np.random.randn(cfg.lr, cfg.lt) +
                    1j * np.random.randn(cfg.lr, cfg.lt)) / np.sqrt(2)

    return theta_t, phi_t, theta_r, phi_r, sigma_matrix

# --- 논문 핵심 모델 함수 ---

def calculate_g_t(t_pos, theta_t, phi_t):
    x_t, y_t = t_pos

    rho_t_j_vec = (x_t * np.cos(theta_t) * np.sin(phi_t) +
                   y_t * np.sin(theta_t))

    g_t = np.exp(1j * 2 * np.pi * rho_t_j_vec / cfg.lambda_wave)

    return g_t


def calculate_f_r(r_pos, theta_r, phi_r):
    x_r, y_r = r_pos

    rho_r_i_vec = (x_r * np.cos(theta_r) * np.sin(phi_r) +
                   y_r * np.sin(theta_r))

    f_r = np.exp(1j * 2 * np.pi * rho_r_i_vec / cfg.lambda_wave)

    return f_r


def calculate_channel_coefficient(g_t, f_r, sigma_matrix):
    f_r_hermitian = f_r.conj().T.reshape(1, -1)

    g_t_col = g_t.reshape(-1, 1)

    h = (f_r_hermitian @ sigma_matrix @ g_t_col).item()

    return h


# --- 전력 및 분석 계산 함수 ---

def calculate_channel_gain(h_tr):
    return np.abs(h_tr) ** 2

def calculate_received_power_dbm(p_t_dbm, path_loss_db, small_scale_gain_linear): # 최종 수신 전력을 dBm 단위로 계산하는 함수입니다 (링크 버짓 사용).
    epsilon = 1e-30

    small_scale_gain_db = 10 * np.log10(small_scale_gain_linear + epsilon)

    received_power_dbm = p_t_dbm - path_loss_db + small_scale_gain_db

    return received_power_dbm


def calculate_path_distances(theta_r, phi_r):
    cfg.lr = len(theta_r)
    if cfg.lr < 2:
        print("  (경로가 1개뿐이라 경로 간 거리를 계산할 수 없습니다.)")
        return

    vectors = np.zeros((cfg.lr, 3))
    vectors[:, 0] = np.cos(theta_r) * np.sin(phi_r)
    vectors[:, 1] = np.sin(theta_r)
    vectors[:, 2] = np.cos(theta_r) * np.cos(phi_r)

    print(f"\n[분석 2: {cfg.lr}개 수신 경로(AoA)간 방향 벡터 거리]")

    for i in range(cfg.lr):
        for j in range(i + 1, cfg.lr):
            dist = np.linalg.norm(vectors[i] - vectors[j])

            dot_product = np.dot(vectors[i], vectors[j])

            angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
            angle_deg = np.degrees(angle_rad)

            print(f"  - 경로 {i + 1} ↔ 경로 {j + 1}:")
            print(f"    - 방향 벡터 간 거리: {dist:.4f} (0~2 사이 값)")
            print(f"    - 두 경로 사이 각도: {angle_deg:.2f} 도")

def aoa_unit_vectors(theta_r, phi_r):
    v = np.zeros((len(theta_r), 3))
    v[:, 0] = np.cos(theta_r) * np.sin(phi_r)  # x 성분
    v[:, 1] = np.sin(theta_r)                  # y 성분
    v[:, 2] = np.cos(theta_r) * np.cos(phi_r)  # z 성분
    return v

def pairwise_path_metrics(vectors):
    L = vectors.shape[0]
    pairs = []
    for i in range(L):
        for j in range(i+1, L):
            dist = np.linalg.norm(vectors[i] - vectors[j])
            cosang = np.clip(np.dot(vectors[i], vectors[j]), -1.0, 1.0) # 내적 및 클리핑
            ang_deg = np.degrees(np.arccos(cosang))

            pairs.append({"i": i, "j": j, "euclid": dist, "angle_deg": ang_deg})
    return pairs