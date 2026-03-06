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
class ReportOrbit:
    """궤도 분석 서브-스키마."""
    # 입력 요소 계약
    a_au:        float
    e:           float
    i_deg:       float
    raan_deg:    float
    argp_deg:    float
    M0_deg:      float
    epoch_jd:    float
    # 좌표계 계약 (항상 명시)
    frame:       str = "ICRF/equatorial-J2000"
    time_scale:  str = "TDB (Barycentric Dynamical Time)"
    gm_sun:      str = "0.01720209895² AU³/day² (Gauss k²)"


@dataclass
class ReportDetection:
    """충돌 탐지 서브-스키마."""
    moid_au:           float
    moid_km:           float
    moid_nu1_deg:      float   # 혜성 진근점각 at MOID
    moid_nu2_deg:      float   # 지구 진근점각 at MOID
    moid_method:       str     # "grid180+L-BFGS-B (multi-start 권장)"
    is_pha:            bool
    pha_threshold_au:  float = 0.05
    # B-plane (접근 시 계산, 없으면 None)
    b_mag_km:          Optional[float] = None
    r_capture_km:      Optional[float] = None
    will_impact:       Optional[bool]  = None
    gravitational_focusing: str = "Öpik (1951), r_cap=R⊕√(1+2GM/Rv²)"


@dataclass
class ReportProbability:
    """충돌 확률 서브-스키마."""
    p_impact:           float
    torino_scale:       int
    torino_desc:        str
    palermo_scale:      float
    palermo_label:      str
    years_to_encounter: float
    # MC 계약
    n_mc_samples:       int
    n_impacts:          int
    mc_seed:            int
    sampling_model:     str = "Gaussian 6-element perturbation (diagonal cov)"
    # 배경률 계약
    background_model:   str = "Chesley & Chodas (2002), log N = 0.5616 - 0.8026·logE"
    torino_reference:   str = "Binzel (2000), IAU 2005 revision"
    palermo_reference:  str = "Chesley et al. (2002) AAS 02-176"


@dataclass
class ReportConfig:
    """재현성 계약 — 이 필드가 있으면 결과를 재현할 수 있다."""
    engine_version:   str
    propagation_mode: str         # "kepler" | "nbody"
    planets_included: List[str]
    nongrav_on:       bool
    mc_seed:          int
    n_mc_samples:     int
    is_ocean:         bool
    v_impact_kms:     float
    theta_deg:        float
    water_depth_km:   float
    coast_dist_km:    float


@dataclass
class FullReport:
    """전체 분석 리포트 — 재현/서명 가능한 계약 구조.

    Schema
    ------
    report.comet         : 혜성 기본 정보
    report.orbit         : 입력 궤도요소 + 좌표계/단위 계약
    report.detection     : MOID, B-plane, 중력집중 계약
    report.probability   : MC 확률, Torino/Palermo + seed 기록
    report.impact        : 에너지, 환경 델타
    report.crater        : Holsapple 크레이터
    report.tsunami       : Ward & Asphaug 쓰나미 (해양 충돌 시)
    report.config        : 재현용 파라미터 전체 (seed, version, 입력)
    report.notes         : 경고/정보 메시지
    """
    # 혜성 기본
    comet_name:   str
    orbit_class:  str
    diameter_km:  float
    rho_gcm3:     float

    # 서브-스키마 (계약 필드)
    orbit:       Optional[ReportOrbit]       = None
    detection:   Optional[ReportDetection]   = None
    probability: Optional[ReportProbability] = None
    config:      Optional[ReportConfig]      = None

    # 충돌 영향
    impact:  Optional[ImpactResult] = None
    crater:  Optional[Any]          = None
    tsunami: Optional[Any]          = None

    # 레거시 플랫 필드 (summary 출력용)
    moid_au:            float = 0.0
    moid_km:            float = 0.0
    is_pha:             bool  = False
    p_impact:           float = 0.0
    torino_scale:       int   = 0
    torino_desc:        str   = ""
    palermo_scale:      float = -99.0
    palermo_label:      str   = ""
    years_to_encounter: float = 0.0
    closest_approach_jd: float = 0.0
    v_inf_kms:           float = 0.0
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        cfg = self.config
        seed_str = f"seed={cfg.mc_seed}, n={cfg.n_mc_samples}" if cfg else "N/A"
        lines = [
            "=" * 62,
            f"  LuciferEngine 충돌 분석 리포트  v{cfg.engine_version if cfg else '?'}",
            f"  재현 키: {seed_str}",
            "=" * 62,
            f"  혜성:       {self.comet_name}",
            f"  궤도 분류:  {self.orbit_class}",
            f"  직경:       {self.diameter_km:.1f} km  |  밀도: {self.rho_gcm3:.2f} g/cm³",
        ]
        if self.orbit:
            o = self.orbit
            lines += [
                "",
                f"  ── 궤도요소 [{o.frame}] ──",
                f"  a={o.a_au:.4f} AU  e={o.e:.6f}  i={o.i_deg:.3f}°",
                f"  Ω={o.raan_deg:.3f}°  ω={o.argp_deg:.3f}°  M0={o.M0_deg:.3f}°",
                f"  epoch: JD {o.epoch_jd}  [{o.time_scale}]",
                f"  GM: {o.gm_sun}",
            ]
        lines += [
            "",
            f"  ── 충돌 탐지 ──",
            f"  MOID:       {self.moid_au:.6f} AU  ({self.moid_km:.0f} km)",
            f"  PHA 여부:   {'YES ⚠' if self.is_pha else 'NO'}",
        ]
        if self.detection:
            d = self.detection
            lines.append(f"  MOID 방법:  {d.moid_method}")
            if d.b_mag_km is not None:
                lines.append(f"  B-plane |B|={d.b_mag_km:.0f} km  r_cap={d.r_capture_km:.0f} km")
        lines += [
            "",
            f"  ── 충돌 확률 ──",
            f"  확률:       {self.p_impact:.2e}",
            f"  Torino:     {self.torino_scale} — {self.torino_desc}",
            f"  Palermo:    {self.palermo_scale:.2f}  ({self.palermo_label})",
            f"  충돌까지:   {self.years_to_encounter:.1f} 년",
        ]
        if self.probability:
            pr = self.probability
            lines.append(f"  MC:         n={pr.n_mc_samples}  seed={pr.mc_seed}  hits={pr.n_impacts}")
            lines.append(f"  샘플링:     {pr.sampling_model}")
        if self.impact:
            im = self.impact
            lines += [
                "",
                f"  ── 충돌 에너지 ──",
                f"  전체 에너지: {im.E_total_MT:.2e} MT",
                f"  유효 에너지: {im.E_eff_MT:.2e} MT",
                f"  충격 강도:   {im.shock_strength}",
                f"  충격파 과압: {im.delta_pressure_atm:.2f} atm (1 km)",
                f"  해수면 상승: {im.delta_sea_level_m:.4f} m",
                f"  극-적도 ΔT: {im.delta_pole_eq_K:.2f} K",
            ]
        if self.crater:
            cr = self.crater
            lines += [
                "",
                f"  ── 크레이터 [Holsapple 1993] ──",
                f"  최종 직경:  {cr.D_final_km:.1f} km  (과도기: {cr.D_transient_km:.1f} km)",
                f"  테두리:     {cr.D_rim_km:.1f} km  |  깊이: {cr.depth_km:.2f} km",
                f"  용융 부피:  {cr.melt_volume_km3:.3f} km³  |  레짐: {cr.regime}",
                f"  분출물 두께(1×R): {cr.ejecta_thick_1km:.2f} m",
            ]
        if self.tsunami:
            ts = self.tsunami
            lines += [
                "",
                f"  ── 쓰나미 [Ward & Asphaug 2000] ──",
                f"  초기 파고:  {ts.H0_m:.1f} m  |  해안 파고: {ts.H_coast_m:.2f} m",
                f"  처오름 높이:{ts.run_up_m:.2f} m  |  도달 시간: {ts.t_arrival_min:.1f} 분",
                f"  파속:       {ts.wave_speed_kms:.3f} km/s  |  주기: {ts.wave_period_s:.0f} s",
                f"  피해 등급:  {ts.inundation_category}",
            ]
        for note in self.notes:
            lines.append(f"  NOTE: {note}")
        lines.append("=" * 62)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """JSON 직렬화 가능한 딕셔너리 변환 (재현 계약 포함)."""
        d: Dict[str, Any] = {
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
        if self.orbit:       d["orbit"]       = asdict(self.orbit)
        if self.detection:   d["detection"]   = asdict(self.detection)
        if self.probability: d["probability"] = asdict(self.probability)
        if self.config:      d["config"]      = asdict(self.config)
        if self.impact:      d["impact"]      = asdict(self.impact)
        if self.crater:      d["crater"]      = asdict(self.crater)
        if self.tsunami:     d["tsunami"]     = asdict(self.tsunami)
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
                      mc_seed: int = 42,
                      is_ocean: bool = False,
                      v_kms: float = 20.0,
                      theta_deg: float = 45.0,
                      water_depth_km: float = 4.0,
                      coast_dist_km: float = 1000.0,
                      propagation_mode: str = "kepler") -> FullReport:
        """전체 분석 파이프라인 실행 → FullReport 반환.

        모든 입력 파라미터는 report.config에 기록되어 재현 가능.
        """
        from math import degrees

        notes = []
        el    = self.record.elements
        if el is None:
            notes.append("궤도요소 없음 — Lucifer 기본값 사용")
            el = BUILTIN_COMETS["Lucifer"].elements

        # ── 1. 궤도 서브-스키마 ────────────────────────────────────────────
        orbit_schema = ReportOrbit(
            a_au=el.a, e=el.e,
            i_deg=degrees(el.i), raan_deg=degrees(el.raan),
            argp_deg=degrees(el.argp), M0_deg=degrees(el.M0),
            epoch_jd=el.epoch_jd,
        )

        # ── 2. MOID ────────────────────────────────────────────────────────
        moid_r = moid_from_comet(el)

        detection_schema = ReportDetection(
            moid_au=moid_r.moid_au,
            moid_km=moid_r.moid_km,
            moid_nu1_deg=degrees(moid_r.nu1),
            moid_nu2_deg=degrees(moid_r.nu2),
            moid_method="grid-180×180 + L-BFGS-B  ※다중 최솟값 주의: 멀티스타트 미적용",
            is_pha=moid_r.is_pha,
        )

        # ── 3. 위험 척도 ───────────────────────────────────────────────────
        ip_pre = ImpactParams(D_km=self.record.diameter_km,
                              rho_gcm3=self.record.rho_gcm3, v_kms=v_kms,
                              theta_deg=theta_deg)
        ir_pre = estimate_impact(ip_pre)

        from .detection.probability import monte_carlo_impact
        from .detection import OrbitalUncertainty
        risk = monte_carlo_impact(
            el_mean=el,
            uncertainty=OrbitalUncertainty(),
            jd_encounter=jd_start + years_ahead * 365.25,
            years_to_impact=years_ahead,
            kinetic_energy_mt=ir_pre.E_eff_MT,
            n_samples=n_mc,
            seed=mc_seed,
        )

        probability_schema = ReportProbability(
            p_impact=risk.p_impact,
            torino_scale=risk.torino_scale,
            torino_desc=risk.torino_description,
            palermo_scale=risk.palermo_scale,
            palermo_label=risk.palermo_label,
            years_to_encounter=years_ahead,
            n_mc_samples=risk.n_mc_samples,
            n_impacts=risk.n_impacts,
            mc_seed=mc_seed,
        )

        # ── 4. 충돌 영향 ───────────────────────────────────────────────────
        ir, cr, ts = self.impact_analysis(v_kms=v_kms, theta_deg=theta_deg,
                                          is_ocean=is_ocean,
                                          water_depth_km=water_depth_km,
                                          coast_dist_km=coast_dist_km)

        # ── 5. Config (재현 계약) ──────────────────────────────────────────
        config = ReportConfig(
            engine_version="1.0.0",
            propagation_mode=propagation_mode,
            planets_included=self.propagator.planets,
            nongrav_on=self.propagator.include_nongrav,
            mc_seed=mc_seed,
            n_mc_samples=n_mc,
            is_ocean=is_ocean,
            v_impact_kms=v_kms,
            theta_deg=theta_deg,
            water_depth_km=water_depth_km,
            coast_dist_km=coast_dist_km,
        )

        # ── Notes ──────────────────────────────────────────────────────────
        if moid_r.is_pha:
            notes.append("PHA (Potentially Hazardous) 기준 충족 (MOID < 0.05 AU)")
        if risk.torino_scale >= 1:
            notes.append(f"Torino Scale {risk.torino_scale} — 감시 강화 필요")
        if el.A1 == 0 and el.A2 == 0:
            notes.append("비중력 가속 미적용 (A1=A2=A3=0). 혜성은 A2>0 권장.")

        return FullReport(
            comet_name=self.record.name,
            orbit_class=self.record.orbit_class,
            diameter_km=self.record.diameter_km,
            rho_gcm3=self.record.rho_gcm3,
            # 서브-스키마
            orbit=orbit_schema,
            detection=detection_schema,
            probability=probability_schema,
            config=config,
            # 충돌 영향
            impact=ir, crater=cr, tsunami=ts,
            # 레거시 플랫 필드
            moid_au=moid_r.moid_au,
            moid_km=moid_r.moid_km,
            is_pha=moid_r.is_pha,
            p_impact=risk.p_impact,
            torino_scale=risk.torino_scale,
            torino_desc=risk.torino_description,
            palermo_scale=risk.palermo_scale,
            palermo_label=risk.palermo_label,
            years_to_encounter=years_ahead,
            notes=notes,
        )

    def __repr__(self) -> str:
        return (f"LuciferEngine(name={self.record.name!r}, "
                f"D={self.record.diameter_km:.1f}km, "
                f"e={self.record.elements.e if self.record.elements else 'N/A'})")
