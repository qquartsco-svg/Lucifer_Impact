"""orbit.kepler — 케플러 궤도 역학 코어

6개 궤도요소 ↔ 상태벡터 변환 + 궤도 전파.

궤도요소 정의:
  a   [AU]   반장축 (semi-major axis)
  e   [-]    이심률 (eccentricity)  0≤e<1: 타원, e=1: 포물선, e>1: 쌍곡선
  i   [rad]  궤도경사각 (inclination)
  Ω   [rad]  승교점 적경 (RAAN, longitude of ascending node)
  ω   [rad]  근점인수 (argument of periapsis)
  M0  [rad]  기준시각 평균근점각 (mean anomaly at epoch)

좌표계: 태양 중심 적도 관성계 (J2000.0 기준)
단위계: AU, day, AU/day  (GM_sun = 0.01720209895² AU³/day²)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple
from math import pi, sqrt, sin, cos, atan2, acos, tan

# 태양 중심 중력 상수 k² (Gauss 상수)
K_GAUSS = 0.01720209895          # AU^(3/2) day^-1 M_sun^(-1/2)
GM_SUN  = K_GAUSS ** 2           # AU³/day²  (M_sun = 1)

# 단위 변환
AU_TO_M = 1.495978707e11         # 1 AU in meters
DAY_TO_S = 86400.0


@dataclass
class OrbitalElements:
    """케플러 궤도 6요소.

    타원 궤도:    0 ≤ e < 1
    포물선 궤도:  e = 1.0  (q = a 로 해석, perihelion distance)
    쌍곡선 궤도:  e > 1.0
    """
    a: float        # AU,  타원: 반장축 / 쌍곡선: 허수 반장축 절댓값
    e: float        # 이심률
    i: float        # rad, 궤도경사각
    raan: float     # rad, 승교점 적경 Ω
    argp: float     # rad, 근점인수 ω
    M0: float       # rad, 평균근점각 (epoch 기준)
    epoch_jd: float = 2451545.0   # J2000.0

    # 선택: 혜성 비중력 가속 파라미터
    A1: float = 0.0  # 래디얼 성분 [AU/day²]
    A2: float = 0.0  # 트랜스버스 성분 [AU/day²]
    A3: float = 0.0  # 법선 성분 [AU/day²]

    @property
    def q(self) -> float:
        """근일점 거리 [AU]"""
        if self.e < 1.0:
            return self.a * (1.0 - self.e)
        else:
            return self.a * (self.e - 1.0)

    @property
    def n(self) -> float:
        """평균 운동 [rad/day]  (타원 궤도만)"""
        if self.e >= 1.0:
            return float('nan')
        return K_GAUSS / (self.a ** 1.5)

    @property
    def T_period(self) -> float:
        """공전 주기 [day]  (타원 궤도만)"""
        if self.e >= 1.0:
            return float('inf')
        return 2.0 * pi / self.n

    def mean_anomaly_at(self, jd: float) -> float:
        """JD 시각의 평균근점각 M [rad]"""
        dt = jd - self.epoch_jd
        return self.M0 + self.n * dt


def solve_kepler(M: float, e: float, tol: float = 1e-12, max_iter: int = 50) -> float:
    """케플러 방정식 M = E - e·sin(E) → E 풀기 (타원 궤도).

    Newton-Raphson 반복. 수렴 보장.
    """
    M = M % (2.0 * pi)
    E = M + e * sin(M) * (1.0 + e * cos(M))
    for _ in range(max_iter):
        f  =  E - e * sin(E) - M
        df =  1.0 - e * cos(E)
        dE = -f / df
        E += dE
        if abs(dE) < tol:
            break
    return E


def solve_kepler_hyperbolic(M: float, e: float, tol: float = 1e-12) -> float:
    """쌍곡선 케플러 방정식 M = e·sinh(F) - F → F 풀기."""
    F = M / (e - 1.0)
    for _ in range(50):
        f  = e * np.sinh(F) - F - M
        df = e * np.cosh(F) - 1.0
        dF = -f / df
        F += dF
        if abs(dF) < tol:
            break
    return F


def elements_to_state(el: OrbitalElements, jd: float) -> Tuple[np.ndarray, np.ndarray]:
    """궤도요소 → 상태벡터 (위치[AU], 속도[AU/day]).

    타원·쌍곡선 모두 지원.
    좌표계: 태양 중심 적도 관성계.
    """
    M = el.mean_anomaly_at(jd) % (2.0 * pi)

    if el.e < 1.0:
        # 타원 궤도
        E = solve_kepler(M, el.e)
        nu = 2.0 * atan2(
            sqrt(1.0 + el.e) * sin(E / 2.0),
            sqrt(1.0 - el.e) * cos(E / 2.0)
        )
        r = el.a * (1.0 - el.e * cos(E))
        p = el.a * (1.0 - el.e ** 2)
        v_r  = K_GAUSS / sqrt(p) * el.e * sin(nu)
        v_nu = K_GAUSS / sqrt(p) * (1.0 + el.e * cos(nu))

    elif el.e > 1.0:
        # 쌍곡선 궤도
        F = solve_kepler_hyperbolic(M, el.e)
        nu = 2.0 * atan2(
            sqrt(el.e + 1.0) * np.sinh(F / 2.0),
            sqrt(el.e - 1.0) * np.cosh(F / 2.0)
        )
        p = el.a * (el.e ** 2 - 1.0)
        r = p / (1.0 + el.e * cos(nu))
        v_r  = K_GAUSS / sqrt(p) * el.e * sin(nu)
        v_nu = K_GAUSS / sqrt(p) * (1.0 + el.e * cos(nu))

    else:
        raise ValueError("포물선 궤도(e=1) 미지원. e=0.9999 또는 1.0001 사용.")

    # 궤도면 내 위치·속도 (perifocal frame)
    pos_p = np.array([r * cos(nu), r * sin(nu), 0.0])
    vel_p = np.array([v_r * cos(nu) - v_nu * sin(nu),
                      v_r * sin(nu) + v_nu * cos(nu), 0.0])

    # 회전 행렬: 궤도면 → 적도 관성계
    R = _rotation_matrix(el.raan, el.i, el.argp)

    return R @ pos_p, R @ vel_p


def state_to_elements(pos: np.ndarray, vel: np.ndarray,
                      epoch_jd: float = 2451545.0) -> OrbitalElements:
    """상태벡터 → 궤도요소 (태양 중심 적도 관성계).

    Parameters
    ----------
    pos : [AU]   위치 벡터 (3D)
    vel : [AU/day] 속도 벡터 (3D)
    epoch_jd : 기준 JD

    Returns
    -------
    OrbitalElements
    """
    r_mag = np.linalg.norm(pos)
    v_mag = np.linalg.norm(vel)

    # 각운동량 벡터
    h = np.cross(pos, vel)
    h_mag = np.linalg.norm(h)

    # 이심률 벡터
    ecc_vec = np.cross(vel, h) / GM_SUN - pos / r_mag
    e = np.linalg.norm(ecc_vec)

    # 에너지 → 반장축
    xi = v_mag**2 / 2.0 - GM_SUN / r_mag
    if abs(xi) < 1e-15:
        a = 1e10  # 포물선에 가까움
    else:
        a = -GM_SUN / (2.0 * xi)

    # 궤도경사각
    i = acos(np.clip(h[2] / h_mag, -1, 1))

    # 승교점 벡터 N = k × h
    k = np.array([0.0, 0.0, 1.0])
    N = np.cross(k, h)
    N_mag = np.linalg.norm(N)

    if N_mag < 1e-12:
        raan = 0.0
    else:
        raan = atan2(N[1], N[0]) % (2 * pi)

    # 근점인수
    if N_mag < 1e-12 or e < 1e-10:
        argp = 0.0
    else:
        argp = atan2(
            np.dot(np.cross(N, ecc_vec), h) / (N_mag * e * h_mag),
            np.dot(N, ecc_vec) / (N_mag * e)
        ) % (2 * pi)

    # 진근점각 → 평균근점각
    if e < 1e-10:
        nu = 0.0
    else:
        nu = atan2(
            np.dot(np.cross(ecc_vec, pos), h) / (e * r_mag * h_mag),
            np.dot(ecc_vec, pos) / (e * r_mag)
        ) % (2 * pi)

    if e < 1.0:
        E = 2.0 * atan2(sqrt(1.0 - e) * sin(nu / 2.0),
                        sqrt(1.0 + e) * cos(nu / 2.0))
        M0 = (E - e * sin(E)) % (2 * pi)
    else:
        F = 2.0 * atanh(sqrt((e - 1.0) / (e + 1.0)) * tan(nu / 2.0))
        M0 = (e * np.sinh(F) - F) % (2 * pi)

    return OrbitalElements(a=a, e=e, i=i, raan=raan, argp=argp,
                           M0=M0, epoch_jd=epoch_jd)


def _rotation_matrix(raan: float, inc: float, argp: float) -> np.ndarray:
    """궤도면 → 적도 관성계 회전 행렬 R_z(-Ω)·R_x(-i)·R_z(-ω)."""
    cO, sO = cos(raan), sin(raan)
    ci, si = cos(inc),  sin(inc)
    co, so = cos(argp), sin(argp)

    return np.array([
        [cO*co - sO*so*ci,  -cO*so - sO*co*ci,  sO*si],
        [sO*co + cO*so*ci,  -sO*so + cO*co*ci, -cO*si],
        [so*si,              co*si,              ci   ],
    ])


def propagate_kepler(el: OrbitalElements, jd_start: float,
                     jd_end: float, dt_day: float = 1.0):
    """케플러 전파: jd_start → jd_end 까지 dt_day 간격으로 상태 계산.

    Returns
    -------
    times : ndarray (N,)  [JD]
    positions : ndarray (N, 3) [AU]
    velocities : ndarray (N, 3) [AU/day]
    """
    times = np.arange(jd_start, jd_end + dt_day * 0.5, dt_day)
    positions  = np.zeros((len(times), 3))
    velocities = np.zeros((len(times), 3))

    for k, t in enumerate(times):
        positions[k], velocities[k] = elements_to_state(el, t)

    return times, positions, velocities


# atanh 간편 함수
def atanh(x: float) -> float:
    return 0.5 * np.log((1 + x) / (1 - x))


__all__ = [
    "OrbitalElements",
    "solve_kepler",
    "solve_kepler_hyperbolic",
    "elements_to_state",
    "state_to_elements",
    "propagate_kepler",
    "GM_SUN", "K_GAUSS", "AU_TO_M", "DAY_TO_S",
]
