"""detection.moid — Minimum Orbit Intersection Distance (MOID)

두 궤도 사이의 최소 거리 계산.
지구 충돌 가능성의 1차 필터로 사용.

방법: 격자 탐색 + Brent 최소화 (2D, ν₁×ν₂ 공간에서 거리 최솟값)

기준: MOID < 0.05 AU → Potentially Hazardous Asteroid (PHA) 조건
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple
from math import pi, cos, sin, atan2, sqrt
from scipy.optimize import minimize

from ..orbit.kepler import OrbitalElements, elements_to_state, _rotation_matrix, GM_SUN


@dataclass
class MOIDResult:
    moid_au: float          # 최소 궤도 교차 거리 [AU]
    moid_km: float          # [km]
    nu1: float              # 혜성 진근점각 at MOID [rad]
    nu2: float              # 지구 진근점각 at MOID [rad]
    is_pha: bool            # PHA 기준 (MOID < 0.05 AU)
    is_earth_crosser: bool  # 지구 궤도 교차 (MOID < Earth_r)


_AU_TO_KM = 1.495978707e8  # 1 AU in km
_EARTH_RADIUS_AU = 6.371e3 / _AU_TO_KM  # 지구 반지름 [AU]
_PHA_MOID_THRESHOLD = 0.05  # AU


def _orbit_point(el: OrbitalElements, nu: float) -> np.ndarray:
    """진근점각 nu에서 궤도 위치 [AU] (태양 중심 관성계).

    근일점 기준 거리 공식: r = p/(1+e·cos ν)
    """
    e = el.e
    if e < 1.0:
        p = el.a * (1.0 - e**2)
    else:
        p = el.a * (e**2 - 1.0)

    r = p / (1.0 + e * cos(nu))
    pos_pf = np.array([r * cos(nu), r * sin(nu), 0.0])
    R = _rotation_matrix(el.raan, el.i, el.argp)
    return R @ pos_pf


def _distance_sq(nu1: float, nu2: float,
                 el1: OrbitalElements, el2: OrbitalElements) -> float:
    """두 진근점각에서 두 궤도 점 사이 거리²."""
    p1 = _orbit_point(el1, nu1)
    p2 = _orbit_point(el2, nu2)
    return float(np.dot(p1 - p2, p1 - p2))


def compute_moid(el_comet: OrbitalElements, el_earth: OrbitalElements,
                 n_grid: int = 180) -> MOIDResult:
    """MOID 계산 (격자 탐색 + 정밀 최적화).

    Parameters
    ----------
    el_comet  : 혜성 궤도요소
    el_earth  : 지구 궤도요소  (기본값: 내장 지구 요소 사용 가능)
    n_grid    : 초기 격자 해상도 (nu₁, nu₂ 각 n_grid 포인트)

    Returns
    -------
    MOIDResult
    """
    nu_arr = np.linspace(0, 2 * pi, n_grid, endpoint=False)

    # 격자 탐색: 최솟값 후보 찾기
    best_d2   = np.inf
    best_nu1  = 0.0
    best_nu2  = 0.0

    for nu1 in nu_arr:
        p1 = _orbit_point(el_comet, nu1)
        for nu2 in nu_arr:
            p2 = _orbit_point(el_earth, nu2)
            d2 = float(np.dot(p1 - p2, p1 - p2))
            if d2 < best_d2:
                best_d2  = d2
                best_nu1 = nu1
                best_nu2 = nu2

    # 정밀 최적화 (L-BFGS-B)
    def obj(x):
        return _distance_sq(x[0], x[1], el_comet, el_earth)

    res = minimize(obj, x0=[best_nu1, best_nu2],
                   method='L-BFGS-B',
                   options={'ftol': 1e-20, 'gtol': 1e-12, 'maxiter': 500})

    moid = sqrt(max(res.fun, 0.0))
    nu1_opt, nu2_opt = float(res.x[0]), float(res.x[1])

    return MOIDResult(
        moid_au=moid,
        moid_km=moid * _AU_TO_KM,
        nu1=nu1_opt,
        nu2=nu2_opt,
        is_pha=moid < _PHA_MOID_THRESHOLD,
        is_earth_crosser=moid < _EARTH_RADIUS_AU,
    )


# 지구 기본 궤도요소 (J2000.0)
EARTH_ELEMENTS = OrbitalElements(
    a=1.000,  e=0.01671, i=0.0,
    raan=0.0, argp=1.7966,  # ~102.9° in rad
    M0=1.7530,  # ~100.5° in rad
    epoch_jd=2451545.0,
)


def moid_from_comet(el_comet: OrbitalElements) -> MOIDResult:
    """혜성 궤도요소만으로 지구 MOID 계산 (지구 요소 기본값 사용)."""
    return compute_moid(el_comet, EARTH_ELEMENTS)


__all__ = [
    "MOIDResult", "compute_moid", "moid_from_comet",
    "EARTH_ELEMENTS", "_AU_TO_KM", "_EARTH_RADIUS_AU",
]
