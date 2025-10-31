import config as cfg
# import ma_model_tx1 as mm
import ma_model_tx2 as mm
import run_simulation as rs
import numpy as np


# --- 메인 실행 ---
if __name__ == "__main__":
    # --- 1. 초기 설정 ---
    print(f"--- 시뮬레이션 파라미터 ---")
    print(f"송신 전력 (P_t): {cfg.pt_dbm:.2f} dBm")
    print(f"주파수 (f_c): {cfg.fc / 1e9:.2f} GHz (파장: {cfg.lambda_wave:.4f} m)")
    print(f"Tx-Rx 대규모 거리 (d): {cfg.d_tx_rx:.1f} m")

    # 1.1. 대규모 경로 손실 계산
    path_loss_db = mm.calculate_path_loss_db(cfg.d_tx_rx, cfg.fc)
    print(f"  >> 대규모 경로 손실 (Path Loss): {path_loss_db:.2f} dB")

    # 1.2. 채널 환경 생성
    channel_env = mm.create_channel_environment()
    theta_t, phi_t, theta_r, phi_r, sigma_matrix = channel_env
    print(f"\n--- 채널 환경 (Seed: {cfg.random_seed}) ---")
    print(f"Tx 경로 수 (L_t): {cfg.lt}, Rx 경로 수 (L_r): {cfg.lr}")
    print(f"MA 수신 영역 (A): {cfg.A_normalized}λ x {cfg.A_normalized}λ")

    # 1.3. 고정된 g(t) 계산
    g_t_fixed = mm.calculate_g_t(cfg.t_pos_fixed, theta_t, phi_t)

    # --- 2. 단일 지점 분석 ---
    # rs.run_single_point_analysis(g_t_fixed, channel_env, path_loss_db)

    # --- 3. 2D 맵 시뮬레이션 ---
    r_example_used, summary = rs.run_single_point_analysis(g_t_fixed, channel_env, path_loss_db)
    # rs.plot_system_layout(r_example_used, channel_env, summary_text=summary)
    center = np.array([0.0, 0.0])
    r1, r2 = mm.make_two_rx_positions_random_square(center,
                                           region_size_m=None,
                                           min_sep_ratio=0.2,
                                           seed=2025)

    rs.plot_system_layout_two_rx(r1, r2, g_t_fixed, channel_env, path_loss_db)

    # rs.plot_system_layout_two_rx(r_example_used, channel_env, summary_text=summary,
    #                       show_two_rx=True,
    #                       offset_lambda=0.35,
    #                       angle_deg=10.0)

    power_map = rs.run_2d_map_simulation(g_t_fixed, channel_env, path_loss_db)

    snr_map = rs.snr_db(power_map, BW=20e6, NF_dB=5.0, T=290.0)
    rs.plot_2d_map(snr_map, label='SNR (dB)')

    # --- 4. MA vs FPA 비교 ---
    rs.compare_ma_vs_fpa(power_map, g_t_fixed, channel_env, path_loss_db)

    # --- 5. 시각화 ---
    # rs.plot_2d_map(power_map)

    mm.calculate_path_distances(channel_env[2], channel_env[3])  # (theta_r, phi_r)
