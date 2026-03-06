"""cli.py — LuciferEngine 커맨드라인 인터페이스.

사용법:
  lucifer-analyze --comet Halley --years 50 --ocean
  lucifer-analyze --name "My Comet" --a 5.0 --e 0.7 --i 30 --raan 0 --argp 90 --M0 0 --D 2.0
"""

from __future__ import annotations

import argparse
import sys
from .engine import LuciferEngine


def main():
    parser = argparse.ArgumentParser(
        prog="lucifer-analyze",
        description="LuciferEngine — 혜성 궤도·충돌 분석 CLI",
    )

    # 혜성 선택
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--comet", metavar="NAME",
                     help="내장 DB 혜성 이름 (Halley, Chicxulub, Lucifer 등)")
    src.add_argument("--sbdb", metavar="JSON_FILE",
                     help="로컬 JPL SBDB JSON 파일 경로")
    src.add_argument("--elements", action="store_true",
                     help="직접 궤도요소 입력 (--a, --e, --i, --raan, --argp, --M0 필요)")

    # 궤도요소 직접 입력
    parser.add_argument("--name",  default="Custom Comet")
    parser.add_argument("--a",     type=float, default=5.0,  help="반장축 [AU]")
    parser.add_argument("--e",     type=float, default=0.7,  help="이심률")
    parser.add_argument("--i",     type=float, default=30.0, help="궤도경사각 [°]")
    parser.add_argument("--raan",  type=float, default=0.0,  help="승교점 적경 [°]")
    parser.add_argument("--argp",  type=float, default=0.0,  help="근점인수 [°]")
    parser.add_argument("--M0",    type=float, default=0.0,  help="평균근점각 [°]")
    parser.add_argument("--D",     type=float, default=1.0,  help="직경 [km]")
    parser.add_argument("--rho",   type=float, default=0.5,  help="밀도 [g/cm³]")

    # 분석 옵션
    parser.add_argument("--years",  type=float, default=100.0, help="충돌까지 시간 [년]")
    parser.add_argument("--n-mc",   type=int,   default=5000,  help="Monte Carlo 샘플 수")
    parser.add_argument("--ocean",  action="store_true", help="해양 충돌 (쓰나미 계산)")
    parser.add_argument("--v",      type=float, default=20.0,  help="충돌 속도 [km/s]")
    parser.add_argument("--theta",  type=float, default=45.0,  help="입사각 [°]")
    parser.add_argument("--json",   action="store_true", help="JSON 형식 출력")

    args = parser.parse_args()

    # 엔진 초기화
    try:
        if args.comet:
            engine = LuciferEngine.from_builtin(args.comet)
        elif args.sbdb:
            engine = LuciferEngine.from_sbdb_json(args.sbdb)
        else:
            engine = LuciferEngine.from_elements(
                name=args.name, a=args.a, e=args.e, i_deg=args.i,
                raan_deg=args.raan, argp_deg=args.argp, M0_deg=args.M0,
                diameter_km=args.D, rho_gcm3=args.rho,
            )
    except ValueError as exc:
        print(f"오류: {exc}", file=sys.stderr)
        sys.exit(1)

    # 전체 분석
    report = engine.full_analysis(
        years_ahead=args.years,
        n_mc=args.n_mc,
        is_ocean=args.ocean,
        v_kms=args.v,
        theta_deg=args.theta,
    )

    if args.json:
        import json
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(report.summary())


if __name__ == "__main__":
    main()
