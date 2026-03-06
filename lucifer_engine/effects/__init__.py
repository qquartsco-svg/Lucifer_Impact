"""effects — 충돌 에너지, 크레이터, 쓰나미 물리 모델.

orbit/detection 없이 단독 실행 가능한 독립 레이어.
외부 진입점: run_effects() — 파라미터 하나로 전체 파이프라인 실행.
"""
from .energy  import ImpactParams, ImpactResult, estimate_impact, impact_from_dict
from .crater  import CraterParams, CraterResult, compute_crater
from .tsunami import TsunamiParams, TsunamiResult, compute_tsunami, tsunami_propagation_profile
from typing import Optional, Tuple


def run_effects(
    D_km:           float,
    rho_gcm3:       float  = 0.6,
    v_kms:          float  = 20.0,
    theta_deg:      float  = 45.0,
    composition:    str    = "ice",
    is_ocean:       bool   = False,
    water_depth_km: float  = 4.0,
    coast_dist_km:  float  = 1000.0,
    rho_target:     float  = 2.7e3,
    Y_Pa:           float  = 1e7,
) -> Tuple[ImpactResult, CraterResult, Optional[TsunamiResult]]:
    """충돌 파라미터 → [에너지, 크레이터, 쓰나미] 한번에 반환.

    orbit / detection 없이 단독 실행 가능한 effects 파이프라인.
    CookiieBrain _06_lucifer_impact 폴백으로도 직접 사용 가능.

    Parameters
    ----------
    D_km        : 충돌체 직경 [km]
    rho_gcm3    : 충돌체 밀도 [g/cm³]  (혜성~0.6, 암석~3.0)
    v_kms       : 지표 충돌 속도 [km/s]  ※ 대기권 감속 미적용
    theta_deg   : 입사각 [°]  (수직=90°)
    composition : 'ice' | 'rock' | 'iron' | 'mixed'
    is_ocean    : True이면 쓰나미 계산 포함
    water_depth_km : 충돌 해역 수심 [km]
    coast_dist_km  : 대상 해안선 거리 [km]
    rho_target  : 타깃 밀도 [kg/m³]  (기본: 화강암 2700)
    Y_Pa        : 타깃 강도 [Pa]     (연암 1e6, 경암 1e8)

    Returns
    -------
    (ImpactResult, CraterResult, TsunamiResult | None)

    Examples
    --------
    >>> ir, cr, ts = run_effects(D_km=5.0, is_ocean=True, v_kms=25.0)
    >>> print(ir.E_eff_MT, cr.D_final_km, ts.run_up_m)
    """
    # 1. 에너지 + 환경 델타
    ip = ImpactParams(
        D_km=D_km, rho_gcm3=rho_gcm3, v_kms=v_kms,
        theta_deg=theta_deg, composition=composition,
    )
    ir = estimate_impact(ip)

    # 2. 크레이터
    cp = CraterParams(
        D_km=D_km, rho_i=rho_gcm3 * 1e3, v_kms=v_kms,
        theta_deg=theta_deg,
        rho_t=rho_target, Y_Pa=Y_Pa,
        target_type="ocean" if is_ocean else "rock",
        water_depth_km=water_depth_km,
    )
    cr = compute_crater(cp)

    # 3. 쓰나미 (해양 충돌 시만)
    ts: Optional[TsunamiResult] = None
    if is_ocean:
        tp = TsunamiParams(
            E_eff_MT=ir.E_eff_MT,
            D_impactor_km=D_km,
            water_depth_km=water_depth_km,
            target_coast_km=coast_dist_km,
        )
        ts = compute_tsunami(tp)

    return ir, cr, ts


__all__ = [
    # 단독 파이프라인 진입점
    "run_effects",
    # 개별 모델
    "ImpactParams", "ImpactResult", "estimate_impact", "impact_from_dict",
    "CraterParams", "CraterResult", "compute_crater",
    "TsunamiParams", "TsunamiResult", "compute_tsunami", "tsunami_propagation_profile",
]
