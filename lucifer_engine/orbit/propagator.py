"""orbit.propagator — RK4/RK45 N-체 수치 적분기

태양계 행성 섭동을 포함한 고정밀 혜성 궤도 전파.
행성 위치: 간략 VSOP87 계수 (목성·토성·천왕성·해왕성 포함).

방정식:
    r̈ = -GM_sun/|r|³ · r
        + Σ_j GM_j [( r_j - r )/|r_j - r|³  -  r_j/|r_j|³]   (행성 섭동)
        + a_ng                                                  (비중력 가속)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from .kepler import GM_SUN, AU_TO_M, OrbitalElements, elements_to_state

# 행성 GM [AU³/day²]
PLANET_GM = {
    "Mercury": 4.9125e-11,
    "Venus":   7.2435e-10,
    "Earth":   8.9875e-10,
    "Mars":    9.5495e-11,
    "Jupiter": 2.8254e-7,
    "Saturn":  8.4597e-8,
    "Uranus":  1.2921e-8,
    "Neptune": 1.5243e-8,
}

# J2000 간략 행성 궤도요소 [a(AU), e, i(deg), Ω(deg), ω(deg), L0(deg), dL/day]
# 출처: JPL Keplerian Elements for Approximate Positions
_PLANET_ELEMENTS = {
    "Mercury": (0.38710,  0.20563, 7.005,   48.331, 29.124,  252.250,  4.09236),
    "Venus":   (0.72333,  0.00677, 3.394,   76.680, 54.884,  181.980,  1.60214),
    "Earth":   (1.00000,  0.01671, 0.000,    0.000, 102.937, 100.464,  0.98561),
    "Mars":    (1.52366,  0.09340, 1.850,   49.558, 286.502,  355.453,  0.52403),
    "Jupiter": (5.20336,  0.04839, 1.303,  100.464, 273.867,   34.396,  0.08309),
    "Saturn":  (9.53707,  0.05415, 2.484,  113.665, 339.391,   49.955,  0.03346),
    "Uranus":  (19.19126, 0.04717, 0.770,   74.006,  96.999,  313.232,  0.01172),
    "Neptune": (30.06896, 0.00859, 1.770,  131.784, 276.340,  304.880,  0.00599),
}

def _planet_position(name: str, jd: float) -> np.ndarray:
    """간략 케플러 전파로 행성 위치 반환 [AU, J2000 적도계 근사]."""
    from math import pi, radians, cos, sin, atan2, sqrt
    a, e, i_deg, raan_deg, argp_deg, L0_deg, dL = _PLANET_ELEMENTS[name]
    dt = jd - 2451545.0
    L  = radians((L0_deg + dL * dt) % 360.0)
    raan = radians(raan_deg)
    argp = radians(argp_deg)
    i    = radians(i_deg)
    M    = (L - argp - raan) % (2 * pi)
    # 케플러 방정식
    E = M
    for _ in range(20):
        E = M + e * sin(E)
    nu = 2.0 * atan2(sqrt(1+e)*sin(E/2), sqrt(1-e)*cos(E/2))
    r  = a * (1 - e * cos(E))
    from .kepler import _rotation_matrix
    R  = _rotation_matrix(raan, i, argp)
    pf = np.array([r * cos(nu), r * sin(nu), 0.0])
    return R @ pf


@dataclass
class PropagatorConfig:
    dt_day: float = 1.0            # 기본 스텝 크기 [day]
    planets: List[str] = None      # 섭동 행성 목록
    include_nongrav: bool = True   # 비중력 가속 포함 여부
    rtol: float = 1e-10            # 적응 스텝 상대 허용오차 (RK45)

    def __post_init__(self):
        if self.planets is None:
            self.planets = ["Jupiter", "Saturn", "Earth", "Mars"]


def _nongrav_accel(r: np.ndarray, v: np.ndarray,
                   A1: float, A2: float, A3: float) -> np.ndarray:
    """혜성 비중력 가속 (Marsden 1973 모델).

    a_ng = g(r) * (A1·r̂  + A2·t̂  + A3·n̂)
    g(r) = α·(r/r0)^(-m) · (1 + (r/r0)^n)^(-k)
    표준 파라미터 (물 승화): α=0.111262, r0=2.808 AU, m=2.15, n=5.093, k=4.6142
    """
    r_mag = np.linalg.norm(r)
    r0, alpha, m, n, k = 2.808, 0.111262, 2.15, 5.093, 4.6142
    g = alpha * (r_mag / r0) ** (-m) / (1.0 + (r_mag / r0) ** n) ** k

    r_hat = r / r_mag
    h = np.cross(r, v)
    h_mag = np.linalg.norm(h)
    if h_mag < 1e-15:
        return np.zeros(3)
    n_hat = h / h_mag
    t_hat = np.cross(n_hat, r_hat)

    return g * (A1 * r_hat + A2 * t_hat + A3 * n_hat)


def _acceleration(r: np.ndarray, v: np.ndarray,
                  jd: float, cfg: PropagatorConfig,
                  A1: float = 0.0, A2: float = 0.0, A3: float = 0.0) -> np.ndarray:
    """전체 가속도 벡터."""
    r_mag = np.linalg.norm(r)
    acc = -GM_SUN / r_mag**3 * r

    for planet in cfg.planets:
        rp  = _planet_position(planet, jd)
        gm  = PLANET_GM[planet]
        dr  = rp - r
        dr3 = np.linalg.norm(dr)**3
        rp3 = np.linalg.norm(rp)**3
        acc += gm * (dr / dr3 - rp / rp3)

    if cfg.include_nongrav and (A1 or A2 or A3):
        acc += _nongrav_accel(r, v, A1, A2, A3)

    return acc


def _rk4_step(r: np.ndarray, v: np.ndarray, jd: float,
              dt: float, cfg: PropagatorConfig,
              A1=0.0, A2=0.0, A3=0.0):
    """고정 스텝 RK4."""
    def f(rv, t):
        pos, vel = rv[:3], rv[3:]
        a = _acceleration(pos, vel, t, cfg, A1, A2, A3)
        return np.concatenate([vel, a])

    state = np.concatenate([r, v])
    k1 = f(state,          jd)
    k2 = f(state + k1*dt/2, jd + dt/2)
    k3 = f(state + k2*dt/2, jd + dt/2)
    k4 = f(state + k3*dt,   jd + dt)
    state_new = state + dt / 6.0 * (k1 + 2*k2 + 2*k3 + k4)
    return state_new[:3], state_new[3:]


def propagate_nbody(el: OrbitalElements, jd_start: float, jd_end: float,
                    cfg: Optional[PropagatorConfig] = None,
                    dt_day: Optional[float] = None) -> Tuple:
    """N-체 수치 전파.

    Parameters
    ----------
    el        : 초기 궤도요소
    jd_start  : 시작 JD
    jd_end    : 종료 JD
    cfg       : PropagatorConfig
    dt_day    : 스텝 크기 (None이면 cfg.dt_day 사용)

    Returns
    -------
    times [JD], positions [AU], velocities [AU/day]
    """
    if cfg is None:
        cfg = PropagatorConfig()
    dt = dt_day if dt_day is not None else cfg.dt_day

    r, v = elements_to_state(el, jd_start)
    jd   = jd_start
    times, positions, velocities = [jd], [r.copy()], [v.copy()]

    while jd < jd_end:
        step = min(dt, jd_end - jd)
        r, v = _rk4_step(r, v, jd, step, cfg, el.A1, el.A2, el.A3)
        jd += step
        times.append(jd)
        positions.append(r.copy())
        velocities.append(v.copy())

    return (np.array(times),
            np.array(positions),
            np.array(velocities))


__all__ = [
    "PropagatorConfig", "propagate_nbody",
    "PLANET_GM", "_planet_position",
]
