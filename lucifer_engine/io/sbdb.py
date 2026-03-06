"""io.sbdb — JPL Small-Body Database (SBDB) 데이터 인터페이스

JPL SBDB API 응답 파싱 + OrbitalElements 변환.
실제 혜성·소행성 데이터를 LuciferEngine에 직접 연결.

API: https://ssd-api.jpl.nasa.gov/sbdb.api
MPC: Minor Planet Center 형식 파싱

지원 형식:
  1. JPL SBDB JSON (API v1.0)
  2. MPC 8-line 혜성 궤도요소 형식
  3. 직접 딕셔너리 입력
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from math import radians, pi
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

from ..orbit.kepler import OrbitalElements


# ─── 1. 데이터 구조 ──────────────────────────────────────────────────────────

@dataclass
class CometRecord:
    """혜성·소행성 통합 레코드."""
    name:         str
    spkid:        str = ""        # JPL SPK-ID
    designation:  str = ""        # MPC 임시 명칭
    orbit_class:  str = ""        # HTC, LPC, JFc, Amor, Apollo, Aten...
    H_mag:        float = 0.0     # 절대 등급
    diameter_km:  float = 0.0     # 직경 [km]  (추정치 포함)
    rho_gcm3:     float = 0.5     # 밀도 추정 [g/cm³]
    elements:     Optional[OrbitalElements] = None
    data_epoch:   str = ""
    source:       str = "unknown"


# ─── 2. JPL SBDB JSON 파싱 ───────────────────────────────────────────────────

def parse_sbdb_json(data: Dict[str, Any]) -> CometRecord:
    """JPL SBDB API JSON 응답 → CometRecord.

    Parameters
    ----------
    data : dict  (requests.get(...).json() 결과)

    Example
    -------
    >>> import requests
    >>> r = requests.get("https://ssd-api.jpl.nasa.gov/sbdb.api",
    ...                  params={"sstr": "Halley", "full-prec": 1, "phys-par": 1})
    >>> rec = parse_sbdb_json(r.json())
    """
    obj  = data.get("object", {})
    orb  = data.get("orbit", {})
    phys = data.get("phys_par", [])

    name = (obj.get("shortname") or
            obj.get("fullname") or
            obj.get("des", "Unknown"))
    spkid       = str(obj.get("spkid", ""))
    orbit_class = obj.get("orbit_class", {}).get("code", "")

    # 물리 파라미터
    H_mag = diam = rho = 0.0
    for pp in phys:
        n = pp.get("name", "")
        v = _safe_float(pp.get("value"))
        if "H" in n and "mag" in pp.get("units", ""):
            H_mag = v
        elif n == "diameter":
            diam = v
        elif "density" in n.lower():
            rho  = v

    # 궤도요소 파싱
    elements = _parse_orbit_elements(orb)
    epoch_str = orb.get("epoch", {}).get("value", "")

    return CometRecord(
        name=name,
        spkid=spkid,
        designation=obj.get("des", ""),
        orbit_class=orbit_class,
        H_mag=H_mag,
        diameter_km=diam if diam else _h_to_diameter(H_mag),
        rho_gcm3=rho if rho else 0.5,
        elements=elements,
        data_epoch=epoch_str,
        source="JPL-SBDB",
    )


def _parse_orbit_elements(orb: Dict) -> Optional[OrbitalElements]:
    """JPL SBDB orbit dict → OrbitalElements."""
    try:
        els = {e["name"]: _safe_float(e["value"])
               for e in orb.get("elements", [])}

        a    = els.get("a")
        e    = els.get("e")
        i    = radians(els.get("i",   0.0))
        raan = radians(els.get("om",  0.0))   # Ω
        argp = radians(els.get("w",   0.0))   # ω
        M0   = radians(els.get("ma",  0.0))   # M

        # q 만 있고 a 없는 경우 (포물선에 가까운 혜성)
        if a is None and els.get("q") and e:
            q = els["q"]
            if abs(e - 1.0) < 1e-6:
                a = q * 1e4  # 포물선 근사
            else:
                a = q / (1.0 - e)

        if a is None or e is None:
            return None

        # epoch → JD
        epoch_val = orb.get("epoch", {}).get("value")
        epoch_jd  = _epoch_to_jd(epoch_val) if epoch_val else 2451545.0

        # 비중력 파라미터
        nongrav = orb.get("nongrav", {})
        A1 = _safe_float(nongrav.get("A1", 0)) * 1.495978707e8 / 86400.0**2 / 1e3  # AU/day² 변환
        A2 = _safe_float(nongrav.get("A2", 0)) * 1.495978707e8 / 86400.0**2 / 1e3
        A3 = _safe_float(nongrav.get("A3", 0)) * 1.495978707e8 / 86400.0**2 / 1e3

        return OrbitalElements(
            a=a, e=e, i=i, raan=raan, argp=argp, M0=M0,
            epoch_jd=epoch_jd, A1=A1, A2=A2, A3=A3,
        )
    except Exception:
        return None


# ─── 3. MPC 혜성 궤도요소 파싱 ──────────────────────────────────────────────

def parse_mpc_comet_line(line: str) -> Optional[OrbitalElements]:
    """MPC 혜성 궤도요소 단일 라인 파싱.

    MPC one-line format (80자):
    Cols 1-4:   근일점 통과 연도
    Cols 5-6:   월
    Cols 7-14:  일 (소수 포함)
    Cols 15-24: q [AU]
    Cols 25-34: e
    Cols 35-44: ω [°]
    Cols 45-54: Ω [°]
    Cols 55-64: i [°]
    Cols 66-71: 궤도 등급 코드
    """
    if len(line) < 64:
        return None
    try:
        year   = int(line[0:4])
        month  = int(line[4:6])
        day    = float(line[6:14])
        q      = float(line[14:24])
        e      = float(line[24:34])
        argp   = radians(float(line[34:44]))
        raan   = radians(float(line[44:54]))
        i      = radians(float(line[54:64]))

        # 근일점 통과 JD
        jd_peri = _date_to_jd(year, month, int(day), day % 1)

        if abs(e - 1.0) < 1e-4:
            a = q * 1e5  # 포물선 근사
        elif e < 1.0:
            a = q / (1.0 - e)
        else:
            a = q / (e - 1.0)

        return OrbitalElements(
            a=a, e=e, i=i, raan=raan, argp=argp, M0=0.0,
            epoch_jd=jd_peri,
        )
    except (ValueError, IndexError):
        return None


# ─── 4. 유틸리티 ────────────────────────────────────────────────────────────

def _safe_float(v) -> float:
    try:
        return float(v) if v is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _h_to_diameter(H: float, pV: float = 0.04) -> float:
    """절대 등급 H → 직경 추정 [km].

    D = 1329 / √pV · 10^(-H/5)
    pV = 0.04 혜성 기본값.
    """
    from math import sqrt, pow
    if H <= 0:
        return 0.0
    return 1329.0 / sqrt(pV) * pow(10.0, -H / 5.0)


def _epoch_to_jd(epoch_str: str) -> float:
    """'YYYY-MM-DD' or JD float string → JD float."""
    try:
        return float(epoch_str)
    except ValueError:
        pass
    try:
        dt = datetime.strptime(epoch_str, "%Y-%m-%d")
        # J2000.0 = JD 2451545.0 = 2000-01-01.5
        days = (dt - datetime(2000, 1, 1, 12)).days
        return 2451545.0 + days
    except ValueError:
        return 2451545.0


def _date_to_jd(year: int, month: int, day: int, frac: float = 0.0) -> float:
    """그레고리력 날짜 → JD."""
    if month <= 2:
        year -= 1
        month += 12
    A = int(year / 100)
    B = 2 - A + int(A / 4)
    jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + frac + B - 1524.5
    return jd


# ─── 5. 내장 혜성 데이터베이스 (유명 혜성) ──────────────────────────────────

BUILTIN_COMETS: Dict[str, CometRecord] = {
    "Halley": CometRecord(
        name="1P/Halley",
        orbit_class="HTC",
        H_mag=5.5,
        diameter_km=15.0,
        rho_gcm3=0.6,
        elements=OrbitalElements(
            a=17.834, e=0.9671, i=radians(162.26),
            raan=radians(58.42), argp=radians(111.33),
            M0=radians(38.38), epoch_jd=2446467.0,
        ),
        source="builtin",
    ),
    "Shoemaker-Levy 9": CometRecord(
        name="D/1993 F2 (Shoemaker-Levy 9)",
        orbit_class="LPC",
        H_mag=0.0,
        diameter_km=2.0,     # 핵 파편 추정
        rho_gcm3=0.5,
        elements=OrbitalElements(
            a=6.864, e=0.9986, i=radians(94.2),
            raan=radians(220.6), argp=radians(354.9),
            M0=0.0, epoch_jd=2449108.5,
        ),
        source="builtin",
    ),
    "Chicxulub": CometRecord(
        name="Chicxulub Impactor (K-Pg)",
        orbit_class="unknown",
        H_mag=0.0,
        diameter_km=10.0,
        rho_gcm3=2.5,
        elements=OrbitalElements(
            a=2.5, e=0.7, i=radians(10.0),
            raan=0.0, argp=0.0, M0=pi,
            epoch_jd=2451545.0,
        ),
        source="builtin (모델)",
    ),
    "Lucifer": CometRecord(
        name="Lucifer Impactor (LuciferEngine 기준 천체)",
        orbit_class="LPC",
        H_mag=0.0,
        diameter_km=5.0,
        rho_gcm3=0.6,
        elements=OrbitalElements(
            a=50.0, e=0.98, i=radians(45.0),
            raan=radians(30.0), argp=radians(120.0),
            M0=0.1, epoch_jd=2451545.0,
        ),
        source="builtin (LuciferEngine default)",
    ),
}


def get_comet(name: str) -> Optional[CometRecord]:
    """내장 DB에서 혜성 레코드 반환."""
    for key, rec in BUILTIN_COMETS.items():
        if key.lower() in name.lower() or name.lower() in rec.name.lower():
            return rec
    return None


def load_sbdb_json_file(path: str) -> CometRecord:
    """로컬 SBDB JSON 파일 로드."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return parse_sbdb_json(data)


__all__ = [
    "CometRecord", "parse_sbdb_json", "parse_mpc_comet_line",
    "BUILTIN_COMETS", "get_comet", "load_sbdb_json_file",
    "_h_to_diameter", "_epoch_to_jd",
]
