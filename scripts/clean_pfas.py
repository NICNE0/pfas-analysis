from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_COLS = [
    "PFHpA/375-85-9",
    "PFHxS/355-46-4",
    "PFOA/335-67-1",
    "PFNA/375-95-1",
    "PFOS/1763-23-1",
    "PFDA/335-76-2",
]


def strip_j_suffix(value: object) -> object:
    if not isinstance(value, str):
        return value

    trimmed = value.rstrip()
    if trimmed.endswith(" J"):
        return trimmed[:-2]
    if trimmed.endswith("J") and len(trimmed) >= 2 and trimmed[-2] == " ":
        return trimmed[:-1]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clean PFAS columns by removing trailing ' J' without regex."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to the source Excel file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("cleaned_pfas.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--sheet",
        default=0,
        help="Sheet name or index to read (default: 0).",
    )
    parser.add_argument(
        "--skiprows",
        type=int,
        default=9,
        help="Rows to skip before header (default: 9 for header on line 10).",
    )
    parser.add_argument(
        "--no-coerce",
        action="store_true",
        help="Do not coerce cleaned columns to numeric.",
    )
    args = parser.parse_args()

    df = pd.read_excel(args.input, sheet_name=args.sheet, skiprows=args.skiprows)

    missing = [c for c in DEFAULT_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns: {missing}")

    df[DEFAULT_COLS] = df[DEFAULT_COLS].applymap(strip_j_suffix)

    if not args.no_coerce:
        df[DEFAULT_COLS] = df[DEFAULT_COLS].apply(pd.to_numeric, errors="coerce")

    df.to_csv(args.output, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
