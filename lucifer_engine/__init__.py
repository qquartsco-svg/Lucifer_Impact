"""
LuciferEngine — 완전 독립 혜성 궤도·충돌 탐색 시스템
=====================================================

아키텍처:
  orbit/      케플러 궤도 역학 + N-체 수치 전파
  detection/  MOID, B-plane, Torino/Palermo Scale
  effects/    충돌 에너지, 크레이터, 쓰나미
  io/         JPL SBDB, MPC 데이터 인터페이스

빠른 시작:
  from lucifer_engine import LuciferEngine
  engine = LuciferEngine.from_builtin("Lucifer")
  report = engine.full_analysis(jd_start=2451545.0, jd_end=2453000.0)
  print(report)
"""

from __future__ import annotations

__version__ = "1.0.0"
__author__  = "CookiieBrain · LuciferEngine"

from .orbit     import OrbitalElements, propagate_kepler, propagate_nbody, PropagatorConfig
from .detection import (MOIDResult, moid_from_comet,
                        BPlaneResult, bplane_from_encounter,
                        RiskAssessment, monte_carlo_impact,
                        torino_scale, palermo_scale)
from .effects   import (ImpactParams, ImpactResult, estimate_impact,
                        CraterParams, CraterResult, compute_crater,
                        TsunamiParams, TsunamiResult, compute_tsunami)
from .io        import CometRecord, get_comet, BUILTIN_COMETS

from .engine import LuciferEngine, FullReport

__all__ = [
    # Core engine
    "LuciferEngine", "FullReport",
    # Orbit
    "OrbitalElements", "propagate_kepler", "propagate_nbody", "PropagatorConfig",
    # Detection
    "MOIDResult", "moid_from_comet",
    "BPlaneResult", "bplane_from_encounter",
    "RiskAssessment", "monte_carlo_impact",
    "torino_scale", "palermo_scale",
    # Effects
    "ImpactParams", "ImpactResult", "estimate_impact",
    "CraterParams", "CraterResult", "compute_crater",
    "TsunamiParams", "TsunamiResult", "compute_tsunami",
    # IO
    "CometRecord", "get_comet", "BUILTIN_COMETS",
]
