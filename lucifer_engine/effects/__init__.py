"""effects — 충돌 에너지, 크레이터, 쓰나미 물리 모델."""
from .energy  import ImpactParams, ImpactResult, estimate_impact, impact_from_dict
from .crater  import CraterParams, CraterResult, compute_crater
from .tsunami import TsunamiParams, TsunamiResult, compute_tsunami, tsunami_propagation_profile

__all__ = [
    "ImpactParams", "ImpactResult", "estimate_impact", "impact_from_dict",
    "CraterParams", "CraterResult", "compute_crater",
    "TsunamiParams", "TsunamiResult", "compute_tsunami", "tsunami_propagation_profile",
]
