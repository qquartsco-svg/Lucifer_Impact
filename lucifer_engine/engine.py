"""engine.py — LuciferEngine 메인 오케스트레이터

전체 분석 파이프라인:
  1. 혜성 데이터 로드 (내장 DB / JPL / MPC)
  2. 궤도 전파 (케플러 or N-체)
  3. MOID + B-plane 충돌 기하 분석
  4. Monte Carlo 충돌 확률 → Torino/Palermo Scale
  5. 충돌 에너지 + 크레이터 + 쓰나미 계산
  6. 통합 리포트 출력
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any

import numpy as np

from .orbit      import (OrbitalElements, PropagatorConfig,
                          propagate_kepler, propagate_nbody,
                          elements_to_state, GM_SUN)
from .detection  import (moid_from_comet, compute_moid, EARTH_ELEMENTS,
                          bplane_from_encounter,
                          monte_carlo_impact, bplane_probability,
                          OrbitalUncertainty, RiskAssessment)
from .effects    import (ImpactParams, estimate_impact, ImpactResult,
                          CraterParams, compute_crater,
                          TsunamiParams, compute_tsunami)
from .io         import CometRecord, get_comet, BUILTIN_COMETS
from .orbit.propagator import _planet_position


@dataclass
class FullReport:
    """전체 분석 리포트."""
    comet_name:   str
    orbit_class:  str
    diameter_km:  float
    rho_gcm3:     float

    # 궤도 분석
    moid_au:       float
    moid_km:       float
    is_pha:        bool

    # 위험 척도
    p_impact:      float
    torino_scale:  int
    torino_desc:   str
    palermo_scale: float
    palermo_label: str
    years_to_encounter: float

    # 충돌 영향
    impact:  Optional[ImpactResult]    = None
    crater:  Optional[Any]             = None   # CraterResult
    tsunami: Optional[Any]             = None   # TsunamiResult

    # 원시 데이터
    closest_approach_jd: float = 0.0
    v_inf_kms:           float = 0.0
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            f"  LuciferEngine 충돌 분석 리포트",
            "=" * 60,
            f"  혜성:       {self.comet_name}",
            f"  궤도 분류:  {self.orbit_class}",
            f"  직경:       {self.diameter_km:.1f} km",
            f"  밀도:       {self.rho_gcm3:.2f} g/cm³",
            "",
            f"  MOID:       {self.moid_au:.6f} AU  ({self.moid_km:.0f} km)",
            f"  PHA 여부:   {'YES ⚠' if self.is_pha else 'NO'}",
            "",
            f"  충돌 확률:  {self.p_impact:.2e}",
            f"  Torino:     {self.torino_scale} — {self.torino_desc}",
            f"  Palermo:    {self.palermo_scale:.2f}  ({self.palermo_label})",
            f"  충돌까지:   {self.years_to_encounter:.1f} 년",
        ]
        if self.impact:
            im = self.impact
            lines += [
                "",
                f"  ── 충돌 에너지 ──",
                f"  전체 에너지: {im.E_total_MT:.2e} MT",
                f"  유효 에너지: {im.E_eff_MT:.2e} MT",
                f"  충격 강도:   {im.shock_strength}",
                f"  충격파 과압: {im.delta_pressure_atm:.2f} atm (1km)",
                f"  해수면 상승: {im.delta_sea_level_m:.4f} m",
                f"  극-적도 ΔT: {im.delta_pole_eq_K:.2f} K",
            ]
        if self.crater:
            cr = self.crater
            lines += [
                "",
                f"  ── 크레이터 ──",
                f"  최종 직경:  {cr.D_final_km:.1f} km",
                f"  테두리:     {cr.D_rim_km:.1f} km",
                f"  깊이:       {cr.depth_km:.2f} km",
                f"  용융 부피:  {cr.melt_volume_km3:.3f} km³",
                f"  레짐:       {cr.regime}",
            ]
        if self.tsunami:
            ts = self.tsunami
            lines += [
                "",
                f"  ── 쓰나미 ──",
                f"  초기 파고:  {ts.H0_m:.1f} m",
                f"  해안 파고:  {ts.H_coast_m:.2f} m",
                f"  처오름 높이:{ts.run_up_m:.2f} m",
                f"  도달 시간:  {ts.t_arrival_min:.1f} 분",
                f"  피해 등급:  {ts.inundation_category}",
            ]
        for note in self.notes:
            lines.append(f"  NOTE: {note}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """JSON 직렬화 가능한 딕셔너리 변환."""
        d = {
            "comet_name": self.comet_name,
            "orbit_class": self.orbit_class,
            "diameter_km": self.diameter_km,
            "rho_gcm3": self.rho_gcm3,
            "moid_au": self.moid_au,
            "moid_km": self.moid_km,
            "is_pha": self.is_pha,
            "p_impact": self.p_impact,
            "torino_scale": self.torino_scale,
            "torino_desc": self.torino_desc,
            "palermo_scale": self.palermo_scale,
            "palermo_label": self.palermo_label,
            "years_to_encounter": self.years_to_encounter,
            "notes": self.notes,
        }
        if self.impact:
            d["impact"] = asdict(self.impact)
        if self.crater:
            d["crater"] = asdict(self.crater)
        if self.tsunami:
            d["tsunami"] = asdict(self.tsunami)
        return d


class LuciferEngine:
    """혜성 궤도·충돌 탐색 통합 엔진.

    Example
    -------
    engine = LuciferEngine.from_builtin("Halley")
    report = engine.full_analysis(years_ahead=10.0)
    print(report.summary())
    """

    def __init__(self, record: CometRecord,
                 propagator: Optional[PropagatorConfig] = None):
        self.record     = record
        self.propagator = propagator or PropagatorConfig()

    # ── 팩토리 메서드 ──────────────────────────────────────────────────────

    @classmethod
    def from_builtin(cls, name: str) -> "LuciferEngine":
        """내장 혜성 DB에서 로드."""
        rec = get_comet(name)
        if rec is None:
            raise ValueError(f"'{name}' 혜성을 내장 DB에서 찾을 수 없습니다. "
                             f"가능한 목록: {list(BUILTIN_COMETS.keys())}")
        return cls(rec)

    @classmethod
    def from_elements(cls, name: str,
                      a: float, e: float, i_deg: float,
                      raan_deg: float, argp_deg: float, M0_deg: float,
                      epoch_jd: float = 2451545.0,
                      diameter_km: float = 1.0,
                      rho_gcm3: float = 0.5) -> "LuciferEngine":
        """직접 궤도요소 입력."""
        from math import radians
        el = OrbitalElements(
            a=a, e=e, i=radians(i_deg),
            raan=radians(raan_deg), argp=radians(argp_deg),
            M0=radians(M0_deg), epoch_jd=epoch_jd,
        )
        rec = CometRecord(
            name=name, orbit_class="custom",
            diameter_km=diameter_km, rho_gcm3=rho_gcm3,
            elements=el, source="manual",
        )
        return cls(rec)

    @classmethod
    def from_sbdb_json(cls, json_path: str) -> "LuciferEngine":
        """로컬 JPL SBDB JSON 파일에서 로드."""
        from .io.sbdb import load_sbdb_json_file
        rec = load_sbdb_json_file(json_path)
        return cls(rec)

    # ── 궤도 전파 ──────────────────────────────────────────────────────────

    def propagate(self, jd_start: float, jd_end: float,
                  mode: str = "kepler", dt_day: float = 1.0):
        """궤도 전파.

        Parameters
        ----------
        mode : 'kepler' (빠름) or 'nbody' (정밀)
        """
        el = self.record.elements
        if el is None:
            raise RuntimeError("궤도요소가 없습니다.")

        if mode == "nbody":
            return propagate_nbody(el, jd_start, jd_end,
                                   cfg=self.propagator, dt_day=dt_day)
        else:
            return propagate_kepler(el, jd_start, jd_end, dt_day)

    # ── 충돌 분석 ──────────────────────────────────────────────────────────

    def moid_analysis(self) -> "MOIDResult":
        """MOID 계산."""
        el = self.record.elements
        if el is None:
            raise RuntimeError("궤도요소 없음.")
        return moid_from_comet(el)

    def risk_assessment(self,
                        years_ahead: float = 100.0,
                        n_mc: int = 10_000,
                        uncertainty: Optional[OrbitalUncertainty] = None,
                        jd_ref: float = 2451545.0) -> RiskAssessment:
        """Monte Carlo 충돌 확률 + Torino/Palermo 계산."""
        el = self.record.elements
        if el is None:
            raise RuntimeError("궤도요소 없음.")

        unc = uncertainty or OrbitalUncertainty()

        # 충돌 에너지 추정
        ip = ImpactParams(D_km=self.record.diameter_km,
                          rho_gcm3=self.record.rho_gcm3)
        ir = estimate_impact(ip)

        return monte_carlo_impact(
            el_mean=el,
            uncertainty=unc,
            jd_encounter=jd_ref + years_ahead * 365.25,
            years_to_impact=years_ahead,
            kinetic_energy_mt=ir.E_eff_MT,
            n_samples=n_mc,
        )

    def impact_analysis(self, v_kms: float = 20.0,
                        theta_deg: float = 45.0,
                        is_ocean: bool = False,
                        water_depth_km: float = 4.0,
                        coast_dist_km: float = 1000.0):
        """충돌 에너지 + 크레이터 + 쓰나미 전체 계산."""
        D   = self.record.diameter_km
        rho = self.record.rho_gcm3

        ip = ImpactParams(D_km=D, rho_gcm3=rho, v_kms=v_kms,
                          theta_deg=theta_deg,
                          composition=("ice" if rho < 1.0 else "rock"))
        ir = estimate_impact(ip)

        cp = CraterParams(D_km=D, rho_i=rho * 1e3, v_kms=v_kms,
                          theta_deg=theta_deg,
                          target_type="ocean" if is_ocean else "rock",
                          water_depth_km=water_depth_km)
        cr = compute_crater(cp)

        ts = None
        if is_ocean:
            tp = TsunamiParams(E_eff_MT=ir.E_eff_MT, D_impactor_km=D,
                               water_depth_km=water_depth_km,
                               target_coast_km=coast_dist_km)
            ts = compute_tsunami(tp)

        return ir, cr, ts

    # ── 전체 파이프라인 ────────────────────────────────────────────────────

    def full_analysis(self,
                      jd_start: float = 2451545.0,
                      years_ahead: float = 100.0,
                      n_mc: int = 5_000,
                      is_ocean: bool = False,
                      v_kms: float = 20.0,
                      theta_deg: float = 45.0) -> FullReport:
        """전체 분석 파이프라인 실행 → FullReport 반환."""
        notes = []
        el    = self.record.elements
        if el is None:
            notes.append("궤도요소 없음 — 기본값 사용")
            el = BUILTIN_COMETS["Lucifer"].elements

        # 1. MOID
        moid_r = moid_from_comet(el)

        # 2. 위험 척도
        risk = self.risk_assessment(years_ahead=years_ahead, n_mc=n_mc,
                                    jd_ref=jd_start)

        # 3. 충돌 영향
        ir, cr, ts = self.impact_analysis(v_kms=v_kms, theta_deg=theta_deg,
                                          is_ocean=is_ocean)

        if moid_r.is_pha:
            notes.append("PHA (Potentially Hazardous Asteroid/Comet) 기준 충족")
        if risk.torino_scale >= 1:
            notes.append(f"Torino Scale {risk.torino_scale} — 감시 강화 필요")

        return FullReport(
            comet_name=self.record.name,
            orbit_class=self.record.orbit_class,
            diameter_km=self.record.diameter_km,
            rho_gcm3=self.record.rho_gcm3,
            moid_au=moid_r.moid_au,
            moid_km=moid_r.moid_km,
            is_pha=moid_r.is_pha,
            p_impact=risk.p_impact,
            torino_scale=risk.torino_scale,
            torino_desc=risk.torino_description,
            palermo_scale=risk.palermo_scale,
            palermo_label=risk.palermo_label,
            years_to_encounter=years_ahead,
            impact=ir,
            crater=cr,
            tsunami=ts,
            notes=notes,
        )

    def __repr__(self) -> str:
        return (f"LuciferEngine(name={self.record.name!r}, "
                f"D={self.record.diameter_km:.1f}km, "
                f"e={self.record.elements.e if self.record.elements else 'N/A'})")
