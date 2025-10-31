import numpy as np
import matplotlib.pyplot as plt
import config as cfg
import ma_model_tx2 as mm
# import ma_model_tx1 as mm
import time

# --- SNR 유틸리티 함수 ---
def noise_power_dbm(BW=20e6, NF_dB=5.0, T=290.0):
    k = 1.380649e-23
    N0_W = k * T * BW

    return 10*np.log10(N0_W) + 30 + NF_dB

def snr_map_db(power_map_dbm, BW=20e6, NF_dB=5.0, T=290.0):
    n_dbm = noise_power_dbm(BW, NF_dB, T)
    # SNR(dB) = 수신 신호 전력(dBm) - 잡음 전력(dBm)
    return power_map_dbm - n_dbm

# --- Noise / SNR helpers (scalar) ---
def noise_power_dbm(BW=20e6, NF_dB=5.0, T=290.0):
    """열잡음 + NF를 dBm으로 반환"""
    k = 1.380649e-23
    N0_W = k * T * BW
    N0_dBm = 10 * np.log10(N0_W) + 30
    return N0_dBm + NF_dB

def snr_db(pr_dbm, BW=20e6, NF_dB=5.0, T=290.0):
    """SNR(dB) = 수신전력(dBm) - 잡음전력(dBm)"""
    return pr_dbm - noise_power_dbm(BW=BW, NF_dB=NF_dB, T=T)


# --- 분석 및 시각화 함수 ---

def run_single_point_analysis(g_t_fixed, channel_env, path_loss_db):
    theta_t, phi_t, theta_r, phi_r, sigma_matrix = channel_env
    r_example = np.array([0.5 * cfg.lambda_wave, 0.2 * cfg.lambda_wave])

    f_r = mm.calculate_f_r(r_example, theta_r, phi_r)
    h = mm.calculate_channel_coefficient(g_t_fixed, f_r, sigma_matrix)
    gain_lin = mm.calculate_channel_gain(h)
    pr_dbm = mm.calculate_received_power_dbm(cfg.pt_dbm, path_loss_db, gain_lin)

    snr_db = pr_dbm - noise_power_dbm(BW=20e6, NF_dB=5.0, T=290.0)

    print("\n--- 단일 지점 상세 분석 ---")
    print(f"r = {r_example} m")
    print(f"|h(t,r)|^2 = {gain_lin:.4f}")
    print(f"P_rx = {pr_dbm:.2f} dBm,  SNR ≈ {snr_db:.2f} dB")
    print(f"AoA count L_r = {len(theta_r)}, AoD count L_t = {len(theta_t)}")

    summary = (f"|h(t,r)|² = {gain_lin:.3f}\n"
               f"P_rx = {pr_dbm:.2f} dBm\n"
               f"SNR ≈ {snr_db:.2f} dB\n"
               f"AoA count L_r = {len(theta_r)}\n"
               f"AoD count L_t = {len(theta_t)}")
    return r_example, summary


def run_2d_map_simulation(g_t_fixed, channel_env, path_loss_db):
    print(f"\n--- 2D 맵 시뮬레이션 ({cfg.A_normalized}λ x {cfg.A_normalized}λ) ---")
    print(f"{cfg.grid_points}x{cfg.grid_points} 맵 계산 시작...")

    theta_t, phi_t, theta_r, phi_r, sigma_matrix = channel_env

    half_A = cfg.A / 2
    x_r_vals = np.linspace(-half_A, half_A, cfg.grid_points)
    y_r_vals = np.linspace(-half_A, half_A, cfg.grid_points)

    received_power_map_dbm = np.zeros((cfg.grid_points, cfg.grid_points))

    start_time = time.time()

    for i in range(cfg.grid_points):
        for j in range(cfg.grid_points):
            r_current = np.array([x_r_vals[i], y_r_vals[j]])

            f_r_current = mm.calculate_f_r(r_current, theta_r, phi_r)
            h_tr_current = mm.calculate_channel_coefficient(g_t_fixed, f_r_current, sigma_matrix)
            gain_current_linear = mm.calculate_channel_gain(h_tr_current)
            received_power_map_dbm[j, i] = mm.calculate_received_power_dbm(
                cfg.pt_dbm,
                path_loss_db,
                gain_current_linear
            )

    end_time = time.time()
    print(f"... 맵 계산 완료 (소요 시간: {end_time - start_time:.2f} 초)")

    return received_power_map_dbm


def plot_2d_map(power_map_dbm, label='Received Power (dBm)'):

    extent = [-cfg.A_normalized / 2, cfg.A_normalized / 2,
              -cfg.A_normalized / 2, cfg.A_normalized / 2]

    plt.figure(figsize=(10, 8))
    im = plt.imshow(power_map_dbm,
                    extent=extent,
                    origin='lower',
                    aspect='equal',
                    cmap='jet')

    plt.colorbar(im, label=label)
    plt.xlabel('Normalized coordinate x_r / λ')
    plt.ylabel('Normalized coordinate y_r / λ')
    plt.gca().set_aspect('equal', adjustable='box')

    title = (f'Received Power Map (L_t={cfg.lt}, L_r={cfg.lr}) '
             f'(Tx at {cfg.d_tx_rx}m distance)')
    plt.title(title)
    plt.show()


def compare_ma_vs_fpa(power_map_dbm, g_t_fixed, channel_env, path_loss_db):
    theta_t, phi_t, theta_r, phi_r, sigma_matrix = channel_env

    f_r_fpa = mm.calculate_f_r(cfg.FPA_pos_fixed, theta_r, phi_r)
    h_tr_fpa = mm.calculate_channel_coefficient(g_t_fixed, f_r_fpa, sigma_matrix)
    gain_fpa_linear = mm.calculate_channel_gain(h_tr_fpa)
    power_fpa_dbm = mm.calculate_received_power_dbm(
        cfg.pt_dbm,
        path_loss_db,
        gain_fpa_linear
    )

    power_ma_max_dbm = np.max(power_map_dbm)

    print(f"\n--- MA vs FPA 성능 비교 (at {cfg.d_tx_rx}m) ---")
    print(f"FPA (at local [0,0]) 수신 전력: {power_fpa_dbm:.2f} dBm")
    print(f"MA (영역 내 최대) 수신 전력: {power_ma_max_dbm:.2f} dBm")
    print(f"성능 향상 (MA Gain): {power_ma_max_dbm - power_fpa_dbm:.2f} dB")

# --- rx = 1 ---
# def plot_system_layout(r_example_pos, channel_env, summary_text=None):
#     print(f"\n[시각화 1: 시스템 레이아웃]")
#
#     theta_t, phi_t, theta_r, phi_r, sigma_matrix = channel_env
#     L_r = cfg.lr
#     A_meters = cfg.A
#     D_tx_rx = cfg.d_tx_rx
#     half_A = A_meters / 2
#
#     fig, (axL, axR) = plt.subplots(
#         1, 2, figsize=(14, 6),
#         gridspec_kw={'width_ratios': [1.6, 1]}
#     )
#
#     # ---------- (좌) 글로벌 배치 (Global Layout) ----------
#     axL.plot(-D_tx_rx, 0, 'ks', markersize=10, label='Tx antenna')
#     axL.text(-D_tx_rx, 0.15, 'Tx', ha='center') # 'Tx' 텍스트 표시
#
#
#     rx_box = plt.Rectangle((-half_A, -half_A), A_meters, A_meters,
#                            edgecolor='tab:blue', facecolor='none', lw=1.8,
#                            label=f'Rx MA region ({cfg.A_normalized}λ × {cfg.A_normalized}λ)')
#     axL.add_patch(rx_box)
#
#     axL.plot(cfg.FPA_pos_fixed[0], cfg.FPA_pos_fixed[1], 'gx', ms=10, mew=2, label='Rx FPA (fixed)')
#     axL.plot(r_example_pos[0], r_example_pos[1], 'ro', ms=8, label='Rx MA (current)')
#
#     axL.annotate(f'd = {D_tx_rx:.1f} m',
#                  xy=(-D_tx_rx, 0), xytext=(-D_tx_rx/2, -0.1*A_meters),
#                  arrowprops=dict(arrowstyle='<->'), ha='center')
#
#     axL.set_title('Global layout')
#     axL.set_xlabel('x (m)'); axL.set_ylabel('y (m)')
#     axL.set_xlim(-D_tx_rx - A_meters, A_meters)
#     ylim = cfg.d_tx_rx * 0.05
#     axL.set_ylim(-ylim, ylim)
#     axL.set_aspect('equal', adjustable='box')
#     axL.grid(True, ls=':')
#     axL.legend(loc='lower right')
#
#     v_aoa = mm.aoa_unit_vectors(theta_r, phi_r)
#     pair_info = mm.pairwise_path_metrics(v_aoa)
#
#     # ---------- (우) Rx 영역 줌인 (Rx Region Zoom-in) ----------
#     axR.add_patch(plt.Rectangle((-half_A, -half_A), A_meters, A_meters,
#                                 edgecolor='tab:blue', facecolor='none', lw=1.8))
#
#     axR.plot(cfg.FPA_pos_fixed[0], cfg.FPA_pos_fixed[1], 'gx', ms=10, mew=2, label='Rx FPA (fixed)')
#     axR.plot(r_example_pos[0], r_example_pos[1], 'ro', ms=8, label='Rx MA (current)')
#
#     d_rx = np.linalg.norm(r_example_pos - cfg.FPA_pos_fixed)
#     axR.text(0.02, 0.98, f'‖r_MA - r_FPA‖ = {d_rx:.3f} m',
#              transform=axR.transAxes, ha='left', va='top',
#              bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.7'))
#
#     arrow_length = half_A * 0.9
#     for i in range(L_r):
#         dx = -arrow_length * np.cos(theta_r[i]) * np.sin(phi_r[i])
#         dy = -arrow_length * np.sin(theta_r[i])
#         if np.hypot(dx, dy) < 1e-6:
#             dx = -arrow_length * 0.707; dy = -arrow_length * 0.707
#         axR.arrow(r_example_pos[0] - dx, r_example_pos[1] - dy, dx, dy,
#                   head_width=half_A*0.06, head_length=half_A*0.1,
#                   fc='green', ec='green', alpha=0.7, length_includes_head=True)
#
#         axR.text(r_example_pos[0] - 1.05*dx, r_example_pos[1] - 1.05*dy,
#                  f'Path {i+1}', color='green', fontsize=9)
#
#     axR.set_title('Rx region (zoom-in)')
#     axR.set_xlim(-half_A, half_A); axR.set_ylim(-half_A, half_A)
#     axR.set_xlabel('x (m)'); axR.set_ylabel('y (m)')
#     axR.set_aspect('equal', adjustable='box'); axR.grid(True, ls=':')
#     axR.legend(loc='upper right')
#
#     if summary_text:
#         axR.text(0.02, 0.98, summary_text, transform=axR.transAxes,
#                  va='top', ha='left',
#                  bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='0.75')) # 텍스트 박스 스타일 지정
#
#     plt.tight_layout()
#     plt.show()
#
#     if pair_info:
#         print("\n[Pairwise AoA metrics at Rx]")
#         for p in pair_info:
#             print(f"  Path {p['i']+1} ↔ Path {p['j']+1}: "
#                   f"euclid={p['euclid']:.4f}, angle={p['angle_deg']:.2f}°")

# --- legend 수정 ---
# def plot_system_layout(r_example_pos, channel_env, summary_text=None,
#                        show_two_rx=True, offset_lambda=0.35, angle_deg=10.0):
#     """
#     show_two_rx=True이면 Rx 영역에 '현재 MA' 빨간 점을 2개 찍고
#     두 점 사이 거리도 표시한다.
#     """
#     import numpy as np
#     theta_t, phi_t, theta_r, phi_r, sigma_matrix = channel_env
#     L_r = cfg.lr
#     A_meters = cfg.A
#     D_tx_rx = cfg.d_tx_rx
#
#     fig = plt.figure(figsize=(14, 6))
#     gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0])
#     ax0 = fig.add_subplot(gs[0, 0])  # Global layout
#     ax1 = fig.add_subplot(gs[0, 1])  # Rx zoom-in
#
#     half_A = A_meters / 2
#
#     # ---------- Left (Global) ----------
#     ax0.add_patch(plt.Rectangle((-D_tx_rx-0.5, -0.4), 1.0, 0.8,
#                                 edgecolor='k', facecolor='k', alpha=0.8, label='Tx antenna'))
#     # Rx 지역 사각형 (전역좌표에서 원점 근처)
#     ax0.add_patch(plt.Rectangle((-half_A, -half_A), A_meters, A_meters,
#                                 edgecolor='tab:blue', facecolor='tab:blue', alpha=0.12,
#                                 label=f'Rx MA region ({cfg.A_normalized:.1f}λ × {cfg.A_normalized:.1f}λ)'))
#     # 거리지시
#     ax0.annotate(f"d = {D_tx_rx:.1f} m",
#                  xy=(-D_tx_rx * 0.15, +0.06 * half_A), ha='center', va='bottom', fontsize=11)
#
#     ax0.set_xlim(-D_tx_rx - 2, 2)
#     ax0.set_ylim(-2.8 * half_A, 2.8 * half_A)
#     ax0.set_yticks([])
#     ax0.set_ylabel('')
#     ax0.set_xlabel('x (m)')
#     ax0.set_title('Global layout')
#     ax0.set_aspect('equal')
#     ax0.grid(True, ls='--', alpha=0.35)
#
#     ax0.legend(loc='upper left', bbox_to_anchor=(0.0, -0.22),
#                   frameon=True, fontsize=11)
#
#     plt.subplots_adjust(left=0.08, right=0.82, top=0.92, bottom=0.22, wspace=0.30)
#
#     # ---------- Right (Rx zoom-in) ----------
#     # Rx 중심, FPA(초록 X)
#     ax1.plot(cfg.FPA_pos_fixed[0], cfg.FPA_pos_fixed[1], 'x', ms=10, mew=2, color='tab:green', label='Rx FPA (fixed)')
#
#     # 빨간 점: 1개 또는 2개
#     if show_two_rx:
#         r1, r2 = two_rx_positions(center_xy=cfg.FPA_pos_fixed, lam=cfg.lambda_wave,
#                                   offset_lambda=offset_lambda, angle_deg=angle_deg)
#         ax1.plot(r1[0], r1[1], 'o', ms=9, color='tab:red', label='Rx MA (current #1)')
#         ax1.plot(r2[0], r2[1], 'o', ms=9, mfc='none', mec='tab:red', mew=2, label='Rx MA (current #2)')
#         # 두 점 사이 거리
#         d12 = np.linalg.norm(r1 - r2)
#         ax1.text(min(r1[0], r2[0]), min(r1[1], r2[1]) - 0.02,
#                  f"‖r₁−r₂‖ = {d12:.3f} m", color='tab:red')
#     else:
#         ax1.plot(r_example_pos[0], r_example_pos[1], 'o', ms=9, color='tab:red', label='Rx MA (current)')
#
#     # 수신 AoA 화살표(초록)
#     arrow_len = half_A * 0.85
#     for i in range(L_r):
#         dx = -arrow_len * np.cos(theta_r[i]) * np.sin(phi_r[i])
#         dy = -arrow_len * np.sin(theta_r[i])
#         # 화살
#         ax1.arrow(cfg.FPA_pos_fixed[0] - dx, cfg.FPA_pos_fixed[1] - dy, dx, dy,
#                   head_width=half_A*0.06, head_length=half_A*0.1,
#                   fc='green', ec='green', alpha=0.7, length_includes_head=True)
#         ax1.text(cfg.FPA_pos_fixed[0] - dx*1.05, cfg.FPA_pos_fixed[1] - dy*1.05,
#                  f'Path {i+1}', color='green', fontsize=9)
#
#     # 요약 박스(겹치지 않게 좌상단)
#     if summary_text:
#         ax1.text(-half_A+0.002, half_A-0.002, summary_text,
#                  va='top', ha='left', fontsize=9,
#                  bbox=dict(boxstyle='round', fc='white', ec='0.5', alpha=0.9))
#
#     ax1.set_xlim(-half_A, half_A)
#     ax1.set_ylim(-half_A, half_A)
#     ax1.set_aspect('equal')
#     ax1.set_xlabel('x (m)')
#     ax1.set_ylabel('y (m)')
#     ax1.set_title('Rx region (zoom-in)')
#     # 범례도 축 밖으로
#     ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0),
#                      borderaxespad=0., frameon=True, fontsize=9, ncol=1)
#     ax1.grid(True, ls='--', alpha=0.4)
#     plt.tight_layout()
#     plt.show()

def plot_system_layout_two_rx(r1, r2, g_t_fixed, channel_env, path_loss_db, summary_text=None):
    theta_t, phi_t, theta_r, phi_r, _ = channel_env
    Lr, Lt = len(theta_r), len(theta_t)

    A_meters = cfg.A
    half_A = A_meters / 2

    # 1) Figure/axes
    fig = plt.figure(figsize=(14, 6))
    gs  = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1.05, 1.35])
    ax0 = fig.add_subplot(gs[0, 0])   # Global
    ax1 = fig.add_subplot(gs[0, 1])   # Rx zoom

    fig.subplots_adjust(left=0.07, right=0.92, top=0.92, bottom=0.28, wspace=0.28)

    # 2) 왼쪽(Global)
    # Tx
    ax0.add_patch(plt.Rectangle((-cfg.d_tx_rx-0.5, -0.4), 1.0, 0.8,
                                edgecolor='k', facecolor='k', alpha=0.85, label='Tx antenna'))
    # Rx MA
    rect = plt.Rectangle((-half_A, -half_A), A_meters, A_meters,
                         edgecolor='tab:blue', facecolor='tab:blue', alpha=0.12,
                         lw=1.8, label=f'Rx MA region ({cfg.A_normalized:.1f}λ × {cfg.A_normalized:.1f}λ)')
    ax0.add_patch(rect)

    ax0.set_xlim(-cfg.d_tx_rx-2, 2)
    ax0.set_ylim(-half_A*1.8, half_A*1.8)

    ax0.set_xlabel('x (m)'); ax0.set_ylabel('y (m)')
    ax0.set_title('Global layout')
    ax0.set_aspect('auto')
    ax0.grid(True, ls='--', alpha=0.35)

    # 거리 표기
    ax0.annotate(f"d = {cfg.d_tx_rx:.1f} m", xy=(-cfg.d_tx_rx*0.15, half_A*0.7),
                 ha='center', va='bottom', fontsize=11)
    # FPA 위치 표시
    handles0, labels0 = ax0.get_legend_handles_labels()
    leg_ax = fig.add_axes([0.16, 0.08, 0.68, 0.10])   # [left,bottom,width,height] (figure coords)
    leg_ax.axis('off')
    leg_ax.legend(handles0, labels0, loc='center', ncol=2, frameon=True, fontsize=11)

    # ----- Right (Rx zoom) -----
    ax1.set_title('Rx region (zoom-in)')
    ax1.set_xlabel('x (m)'); ax1.set_ylabel('y (m)')
    ax1.set_aspect('equal')
    ax1.set_xlim(-0.08, 0.08); ax1.set_ylim(-0.08, 0.09)
    ax1.grid(True, ls='--', alpha=0.4)

    # FPA 위치 표시
    ax1.plot(cfg.FPA_pos_fixed[0], cfg.FPA_pos_fixed[1], marker='x', color='g',
             ms=10, mew=2, label='Rx FPA (fixed)')

    # 수신 경로 화살표
    arrow_len = half_A * 0.9
    for i in range(Lr):
        dx = -arrow_len * np.cos(theta_r[i]) * np.sin(phi_r[i])
        dy = -arrow_len * np.sin(theta_r[i])
        ax1.arrow(cfg.FPA_pos_fixed[0] - dx, cfg.FPA_pos_fixed[1] - dy, dx, dy,
                  head_width=half_A*0.06, head_length=half_A*0.1,
                  fc='green', ec='green', alpha=0.7, length_includes_head=True)
        ax1.text(cfg.FPA_pos_fixed[0] - dx*1.05, cfg.FPA_pos_fixed[1] - dy*1.05,
                 f'Path {i+1}', color='green', fontsize=9)

    # r1/r2 포인트 및 메트릭
    m1 = rx_metrics(r1, g_t_fixed, channel_env, path_loss_db)
    m2 = rx_metrics(r2, g_t_fixed, channel_env, path_loss_db)

    ax1.plot(r1[0], r1[1], 'o', color='crimson', ms=8, label='Rx MA (current #1)')
    ax1.plot(r2[0], r2[1], 'o', color='crimson', mfc='white', ms=8, label='Rx MA (current #2)')

    # 두 점 사이 거리
    dist = np.linalg.norm(np.array(r1) - np.array(r2))
    mid  = (np.array(r1) + np.array(r2)) / 2
    ax1.text(mid[0]+0.015, mid[1], f"||r1−r2|| = {dist:.3f} m", color='crimson')

    # 정보 박스(각 Rx별)
    box_kw = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, ec='0.7')
    txt1 = (f"r1=({r1[0]:.3f},{r1[1]:.3f}) m\n"
            f"|h|²={m1['gain']:.3f}\n"
            f"P={m1['Pr_dBm']:.2f} dBm\n"
            f"SNR={m1['SNR_dB']:.1f} dB\n"
            f"AoA={Lr}, AoD={Lt}")
    ax1.text(r1[0]+0.012, r1[1]+0.014, txt1, fontsize=9, bbox=box_kw)

    txt2 = (f"r2=({r2[0]:.3f},{r2[1]:.3f}) m\n"
            f"|h|²={m2['gain']:.3f}\n"
            f"P={m2['Pr_dBm']:.2f} dBm\n"
            f"SNR={m2['SNR_dB']:.1f} dB\n"
            f"AoA={Lr}, AoD={Lt}")
    ax1.text(r2[0]-0.12, r2[1]-0.03, txt2, fontsize=9, bbox=box_kw)

    ax1.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98),
                     borderaxespad=0., frameon=True, fontsize=9, ncol=1)

    ax1.margins(x=0.08, y=0.08)
    plt.tight_layout(pad=1.6)

    # 좌/우 패널 여백
    plt.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.15, wspace=0.30)
    plt.show()

    # 콘솔에도 상세 출력
    def _print_metrics(tag, m):
        print(f"[{tag}] r=({m['r'][0]:.4f},{m['r'][1]:.4f}) m | |h|^2={m['gain']:.4f} | "
              f"P={m['Pr_dBm']:.2f} dBm | SNR={m['SNR_dB']:.2f} dB")
    print("\n[Per-Rx metrics]")
    _print_metrics('r1', m1); _print_metrics('r2', m2)
    print(f"AoA list (deg): {np.round(m1['AoA_deg'],2)}")
    print(f"AoD list (deg): {np.round(m1['AoD_deg'],2)}")



def two_rx_positions(center_xy, lam, offset_lambda=0.4, angle_deg=0.0):
    r = offset_lambda * lam
    th = np.radians(angle_deg)

    # 수직 방향 단위벡터 (중앙축에 수직)
    n = np.array([-np.sin(th), np.cos(th)])
    p1 = np.asarray(center_xy) + r * n
    p2 = np.asarray(center_xy) - r * n
    return p1, p2

def rx_metrics(r_pos, g_t_fixed, channel_env, path_loss_db, BW=20e6, NF_dB=5.0):
    theta_t, phi_t, theta_r, phi_r, sigma = channel_env
    f_r = mm.calculate_f_r(r_pos, theta_r, phi_r)
    h   = mm.calculate_channel_coefficient(g_t_fixed, f_r, sigma)
    gain = mm.calculate_channel_gain(h)                   # |h|^2
    pr_dbm = mm.calculate_received_power_dbm(cfg.pt_dbm, path_loss_db, gain)
    snr_db_val = snr_db(pr_dbm, NF_dB=NF_dB, BW=BW)

    return {
        "r": np.array(r_pos),
        "h": h,
        "gain": gain,
        "Pr_dBm": pr_dbm,
        "SNR_dB": snr_db_val,
        # 참조용으로 AoA/AoD 전체(경로별) 목록을 같이 넘김
        "AoA_deg": np.degrees(theta_r),
        "AoD_deg": np.degrees(theta_t),
    }

def make_two_rx_positions(center, offset):
    r1 = center + np.array([0.0, +offset])
    r2 = center + np.array([0.0, -offset])
    return r1, r2