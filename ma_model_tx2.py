import numpy as np
import config as cfg
import matplotlib.pyplot as plt

# --- 기존 함수들 ---

def watts_to_dbm(watts):
    epsilon = 1e-30
    return 10 * np.log10((watts + epsilon) * 1000)


def calculate_path_loss_db(distance_m, f_c_hz):
    epsilon = 1e-9  # 0m 거리 방지
    lambda_wave = cfg.c / f_c_hz
    path_loss_linear = ((4 * np.pi * (distance_m + epsilon)) / lambda_wave) ** 2
    path_loss_db = 10 * np.log10(path_loss_linear)
    return path_loss_db


def create_channel_environment():
    np.random.seed(cfg.random_seed)

    # 1. 송신(Tx) 측 AoDs
    theta_t = np.random.uniform(-np.pi / 2, np.pi / 2, cfg.lt)
    phi_t = np.random.uniform(-np.pi / 2, np.pi / 2, cfg.lt)

    # 2. 수신(Rx) 측 AoAs
    theta_r = np.random.uniform(-np.pi / 2, np.pi / 2, cfg.lr)
    phi_r = np.random.uniform(-np.pi / 2, np.pi / 2, cfg.lr)

    # 3. 경로 응답 행렬 (Sigma)
    # (평균 전력 1로 정규화된 CSCG)
    sigma_matrix = (np.random.randn(cfg.lr, cfg.lt) +
                    1j * np.random.randn(cfg.lr, cfg.lt)) / np.sqrt(2)

    return theta_t, phi_t, theta_r, phi_r, sigma_matrix


def calculate_g_t(t_pos, theta_t, phi_t):
    x_t, y_t = t_pos

    # rho_t_j 계산 (크기 L_T)
    rho_t_j_vec = (x_t * np.cos(theta_t) * np.sin(phi_t) +
                   y_t * np.sin(theta_t))

    # g(t) 벡터 계산 (크기 L_T)
    g_t = np.exp(1j * 2 * np.pi * rho_t_j_vec / cfg.lambda_wave)

    return g_t


def calculate_f_r(r_pos, theta_r, phi_r):
    x_r, y_r = r_pos

    # rho_r_i 계산 (크기 L_R)
    rho_r_i_vec = (x_r * np.cos(theta_r) * np.sin(phi_r) +
                   y_r * np.sin(theta_r))

    # f(r) 벡터 계산 (크기 L_R)
    f_r = np.exp(1j * 2 * np.pi * rho_r_i_vec / cfg.lambda_wave)

    return f_r


def calculate_channel_coefficient(g_t, f_r, sigma_matrix):
    # f(r)^H (Hermitian transpose), (1, L_R)
    f_r_hermitian = f_r.conj().T.reshape(1, -1)

    # g(t), (L_T, 1)
    g_t_col = g_t.reshape(-1, 1)

    # h = (1, L_R) @ (L_R, L_T) @ (L_T, 1)
    h = (f_r_hermitian @ sigma_matrix @ g_t_col).item()

    return h

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

# --- 두 개의 Rx에 대한 수신 전력 계산 ---

def generate_two_rx_positions(tx_pos, theta_offset=0.1):
    """
    Tx 위치에서 두 개의 서로 다른 각도로 신호를 쏠 때,
    두 개의 수신 위치를 생성합니다. theta_offset은 두 신호의 각도 차이입니다.
    """
    x_t, y_t, z_t = tx_pos

    # 첫 번째 수신 위치
    rx1 = [x_t + np.cos(theta_offset), y_t + np.sin(theta_offset), z_t]

    # 두 번째 수신 위치 (다른 각도)
    rx2 = [x_t + np.cos(-theta_offset), y_t + np.sin(-theta_offset), z_t]

    return np.array(rx1), np.array(rx2)


def calculate_two_rx_power(tx_pos, fc, Pt_dBm, Gt_dBi=8.0, Gr_dBi=8.0, multipath=None, region_center=None):
    rx1, rx2 = generate_two_rx_positions(tx_pos)

    # 두 Rx에서의 전력 계산
    Pr1_dBm, g1 = calculate_received_power_dbm(tx_pos, rx1, fc, Pt_dBm=Pt_dBm, Gt_dBi=Gt_dBi, Gr_dBi=Gr_dBi, multipath=multipath, region_center=region_center)
    Pr2_dBm, g2 = calculate_received_power_dbm(tx_pos, rx2, fc, Pt_dBm=Pt_dBm, Gt_dBi=Gt_dBi, Gr_dBi=Gr_dBi, multipath=multipath, region_center=region_center)

    # 두 Rx의 거리를 구합니다
    distance = np.linalg.norm(rx1 - rx2)

    return Pr1_dBm, Pr2_dBm, g1, g2, distance

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


# --- 2D 시각화 ---

def plot_two_rx_positions(tx_pos, fc, Pt_dBm, region_center=None):
    Pr1_dBm, Pr2_dBm, g1, g2, distance = calculate_two_rx_power(tx_pos, fc, Pt_dBm, region_center=region_center)

    # 수신 위치
    rx1, rx2 = generate_two_rx_positions(tx_pos)

    # 그래프 그리기
    fig, ax = plt.subplots(figsize=(14, 7))

    # Tx 위치 표시
    ax.plot(tx_pos[0], tx_pos[1], 'ks', markersize=10, label=f'Tx Antenna (t={tx_pos})')

    # Rx1, Rx2 위치 표시
    ax.plot(rx1[0], rx1[1], 'ro', markersize=10, label=f'Rx1 (r1={rx1})')
    ax.plot(rx2[0], rx2[1], 'bo', markersize=10, label=f'Rx2 (r2={rx2})')

    # Tx에서 두 Rx로 가는 경로 그리기
    ax.plot([tx_pos[0], rx1[0]], [tx_pos[1], rx1[1]], 'g--', label=f'Path 1')
    ax.plot([tx_pos[0], rx2[0]], [tx_pos[1], rx2[1]], 'b--', label=f'Path 2')

    # Rx1과 Rx2 사이의 거리 표시
    ax.text((rx1[0] + rx2[0]) / 2, (rx1[1] + rx2[1]) / 2, f'Distance = {distance:.2f} m', fontsize=12, color='black')

    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title(f'System Layout (Tx at {tx_pos}, Distance between Rx1 and Rx2: {distance:.2f} m)')
    ax.legend()
    plt.grid(True)
    plt.show()

def make_two_rx_positions_random_square(center,
                                        region_size_m=None,
                                        min_sep_ratio=0.2,
                                        seed=None):

    rng = np.random.default_rng(seed)
    A = cfg.A if region_size_m is None else float(region_size_m)
    half_A = A / 2.0

    # 최소 분리 거리
    min_sep = min_sep_ratio * A

    # 반복적으로 샘플해서 조건 만족할 때까지
    for _ in range(10_000):
        # [-half_A, half_A] 범위에서 균일
        r1 = center + rng.uniform(low=-half_A, high=half_A, size=2)
        r2 = center + rng.uniform(low=-half_A, high=half_A, size=2)
        if np.linalg.norm(r1 - r2) >= min_sep:
            return r1, r2

    return r1, r2