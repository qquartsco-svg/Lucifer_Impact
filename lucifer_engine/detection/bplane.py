"""detection.bplane — B-plane 충돌 기하학

B-plane: 지구 중심을 통과하며 쌍곡선 점근선에 수직인 평면.
혜성의 지구 접근 경로를 이 평면 위의 B 벡터로 표현.

핵심 개념:
  B 벡터:   충돌 기준 방향까지의 최근접 벡터 (B-plane 내)
  |B|:      충돌 파라미터 (impact parameter) [km]
  θ:        B-plane 내 방위각
  t_CA:     최근접 통과 시각 (JD)
  p_impact: 충돌 확률 (타원 오차 타원 기반)

충돌 조건: |B| < r_capture = R_earth * sqrt(1 + v_inf²/(2·GM_earth/R_earth))
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from math import pi, sqrt, atan2, cos, sin

from ..orbit.kepler import AU_TO_M, DAY_TO_S

# 지구 물리 상수
GM_EARTH_KM3 = 3.986004418e5     # km³/s²
R_EARTH_KM   = 6371.0            # km
AU_TO_KM     = 1.495978707e8     # 1 AU → km


@dataclass
class BPlaneResult:
    """B-plane 분석 결과."""
    b_mag_km: float       # |B| 충돌 파라미터 [km]
    b_t_km: float         # B·T 성분 [km]  (황도 방향)
    b_r_km: float         # B·R 성분 [km]  (황도 수직)
    theta_rad: float      # B-plane 방위각 [rad]
    t_closest_jd: float   # 최근접 통과 JD
    v_inf_kms: float      # 쌍곡선 과잉 속도 v∞ [km/s]
    r_capture_km: float   # 충돌 포획 반지름 [km]
    will_impact: bool     # |B| < r_capture 여부
    stretch_factor: float # 선형 신장 인자 (불확도 전파용)


def compute_bplane(pos_au: np.ndarray, vel_au_day: np.ndarray,
                   jd_current: float) -> BPlaneResult:
    """쌍곡선 근사로 B-plane 충돌 파라미터 계산.

    Parameters
    ----------
    pos_au      : 혜성 위치 (지구 중심 기준) [AU]
    vel_au_day  : 혜성 속도 [AU/day]
    jd_current  : 현재 JD

    Notes
    -----
    pos, vel 은 지구 중심 좌표여야 함.
    태양 중심 좌표라면 먼저 지구 위치/속도를 빼서 변환.
    """
    # AU/day → km/s
    AU_PER_DAY_TO_KM_S = AU_TO_KM / 86400.0
    pos_km  = pos_au  * AU_TO_KM
    vel_kms = vel_au_day * AU_PER_DAY_TO_KM_S

    r_mag = np.linalg.norm(pos_km)
    v_mag = np.linalg.norm(vel_kms)

    # 에너지 (양수 = 쌍곡선)
    xi = v_mag**2 / 2.0 - GM_EARTH_KM3 / r_mag
    v_inf = sqrt(max(2.0 * xi, 0.0))

    # 각운동량
    h = np.cross(pos_km, vel_kms)
    h_mag = np.linalg.norm(h)

    # 충돌 파라미터 b = h / v_inf
    b_mag = h_mag / v_inf if v_inf > 0 else h_mag

    # B-plane 기저 벡터 구성
    # S: 점근 입사 방향 (v∞ 방향)
    S_hat = vel_kms / v_mag

    # T: 황도면 기준 (T = S × ẑ 정규화)
    Z = np.array([0.0, 0.0, 1.0])
    T = np.cross(S_hat, Z)
    T_norm = np.linalg.norm(T)
    if T_norm < 1e-10:
        T = np.array([1.0, 0.0, 0.0])
    else:
        T /= T_norm
    R_hat = np.cross(S_hat, T)

    # B 벡터 (h × S_hat / v_inf)
    B_vec = np.cross(h, S_hat) / (v_inf if v_inf > 0 else 1.0)
    B_T = float(np.dot(B_vec, T))
    B_R = float(np.dot(B_vec, R_hat))
    theta = atan2(B_R, B_T)

    # 최근접 통과 시각
    # t_CA = -(r·v)/v² (선형 근사)
    dt_s = -float(np.dot(pos_km, vel_kms)) / v_mag**2
    t_ca_jd = jd_current + dt_s / 86400.0

    # 포획 반지름 (Öpik 공식)
    r_cap = R_EARTH_KM * sqrt(1.0 + 2.0 * GM_EARTH_KM3 / (R_EARTH_KM * v_inf**2)) if v_inf > 0 else R_EARTH_KM * 10.0

    # 선형 신장 인자 (시간 불확도가 B-plane 불확도로 전환되는 비율)
    # dB/dt ≈ v_inf * sin(angle)  (근사)
    stretch = v_inf * abs(dt_s) / max(b_mag, 1.0)

    return BPlaneResult(
        b_mag_km=b_mag,
        b_t_km=B_T,
        b_r_km=B_R,
        theta_rad=theta,
        t_closest_jd=t_ca_jd,
        v_inf_kms=v_inf,
        r_capture_km=r_cap,
        will_impact=b_mag < r_cap,
        stretch_factor=stretch,
    )


def bplane_from_encounter(pos_helio: np.ndarray, vel_helio: np.ndarray,
                          earth_pos: np.ndarray, earth_vel: np.ndarray,
                          jd: float) -> BPlaneResult:
    """태양 중심 좌표에서 지구 중심 B-plane 계산.

    Parameters
    ----------
    pos_helio  : 혜성 태양 중심 위치 [AU]
    vel_helio  : 혜성 태양 중심 속도 [AU/day]
    earth_pos  : 지구 태양 중심 위치 [AU]
    earth_vel  : 지구 태양 중심 속도 [AU/day]
    jd         : 계산 시각 JD
    """
    rel_pos = pos_helio - earth_pos
    rel_vel = vel_helio - earth_vel
    return compute_bplane(rel_pos, rel_vel, jd)


__all__ = [
    "BPlaneResult", "compute_bplane", "bplane_from_encounter",
    "GM_EARTH_KM3", "R_EARTH_KM", "AU_TO_KM",
]
