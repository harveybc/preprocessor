
#!/usr/bin/env python3
import argparse
import sys
from typing import List, Tuple, Optional

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate time completeness of a CSV time series (e.g., forex) and report missing periods."
    )
    p.add_argument("-f", "--file", required=True, help="Path to the CSV file to validate.")
    p.add_argument("--datetime-col", default="DATE_TIME", help="Datetime column name. Default: DATE_TIME")
    p.add_argument(
        "--dt-format",
        default="%Y-%m-%d %H:%M:%S",
        help="Strptime format for DATE_TIME parsing. Default: %%Y-%%m-%%d %%H:%%M:%%S",
    )
    p.add_argument(
        "--freq",
        default="H",
        help="Expected frequency (pandas offset alias). Examples: H, 15T, 30T. Default: H",
    )
    p.add_argument(
        "--start",
        default=None,
        help="Optional start datetime to validate (same format as --dt-format). Default: first timestamp in data",
    )
    p.add_argument(
        "--end",
        default=None,
        help="Optional end datetime to validate (same format as --dt-format). Default: last timestamp in data",
    )
    p.add_argument(
        "--include-weekends",
        action="store_true",
        help="Include weekends in expected trading hours (off by default).",
    )
    p.add_argument(
        "--sunday-open-hour",
        type=int,
        default=None,
        help="If set (e.g., 22), includes Sunday hours >= this hour as trading hours. Only applied when weekends are skipped.",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero code if missing timestamps are found.",
    )
    p.add_argument(
        "--show-missing",
        type=int,
        default=50,
        help="How many missing timestamps to print individually (default: 50).",
    )
    return p.parse_args()


def read_datetime_series(
    path: str, dt_col: str, fmt: Optional[str]
) -> pd.Series:
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        print(f"ERROR: Failed to read CSV: {e}", file=sys.stderr)
        sys.exit(2)

    if dt_col not in df.columns:
        print(f"ERROR: Column '{dt_col}' not found. Available columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(2)

    # Parse datetimes
    s = pd.to_datetime(df[dt_col], format=fmt, errors="coerce")
    bad = s.isna().sum()
    if bad > 0:
        print(f"WARNING: {bad} rows had unparseable '{dt_col}' values and will be ignored.", file=sys.stderr)

    s = s.dropna().astype("datetime64[ns]")

    return s


def build_expected_index(
    start: pd.Timestamp,
    end: pd.Timestamp,
    freq: str,
    include_weekends: bool,
    sunday_open_hour: Optional[int],
) -> pd.DatetimeIndex:
    # Normalize start/end to the frequency grid
    try:
        # floor/ceil to frequency
        start_aligned = start.floor(freq)
        end_aligned = end.floor(freq)
    except Exception:
        # If freq not floor-able, fall back to hourly floor when possible
        start_aligned = start.floor("S")  # second floor
        end_aligned = end.floor("S")

    rng = pd.date_range(start=start_aligned, end=end_aligned, freq=freq)

    if include_weekends:
        return rng

    # Forex available hours (default): Monday–Friday only. Optionally allow Sunday evening open.
    def is_trading(ts: pd.Timestamp) -> bool:
        wd = ts.weekday()  # Monday=0 ... Sunday=6
        if wd < 5:
            return True
        if wd == 6 and sunday_open_hour is not None:
            return ts.hour >= sunday_open_hour
        return False

    mask = [is_trading(ts) for ts in rng]
    return rng[mask]


def find_missing_and_duplicates(
    present: pd.DatetimeIndex, expected: pd.DatetimeIndex
) -> Tuple[pd.DatetimeIndex, int]:
    # Duplicates
    dup_count = int(pd.Series(present).duplicated().sum())

    # Unique present timestamps
    present_unique = pd.DatetimeIndex(sorted(pd.unique(present)))

    # Missing relative to expected
    missing = expected.difference(present_unique)

    return missing, dup_count


def group_consecutive_missing(missing: pd.DatetimeIndex, freq: str) -> List[Tuple[pd.Timestamp, pd.Timestamp, int]]:
    if len(missing) == 0:
        return []

    # Sort
    miss = missing.sort_values()
    groups: List[Tuple[pd.Timestamp, pd.Timestamp, int]] = []

    # Expected step as pandas offset; convert to Timedelta by taking diff of first two expected of that freq
    step = pd.tseries.frequencies.to_offset(freq).delta
    if step is None or step == pd.Timedelta(0):
        # Fallback: infer from first difference (assume hourly-like grid)
        if len(miss) >= 2:
            step = miss[1] - miss[0]
        else:
            step = pd.Timedelta(hours=1)

    start = miss[0]
    prev = miss[0]
    count = 1

    for ts in miss[1:]:
        if ts - prev == step:
            count += 1
            prev = ts
        else:
            groups.append((start, prev, count))
            start = ts
            prev = ts
            count = 1
    groups.append((start, prev, count))
    return groups


def fmt_ts(ts: pd.Timestamp) -> str:
    # ISO-like without timezone
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def main():
    args = parse_args()

    s = read_datetime_series(args.file, args.datetime_col, args.dt_format)
    if s.empty:
        print("ERROR: No valid datetime rows found after parsing.", file=sys.stderr)
        sys.exit(2)

    s = s.sort_values().reset_index(drop=True)

    inferred_start = s.iloc[0]
    inferred_end = s.iloc[-1]

    if args.start:
        try:
            start = pd.to_datetime(args.start, format=args.dt_format)
        except Exception:
            start = pd.to_datetime(args.start)  # try flexible parse
    else:
        start = inferred_start

    if args.end:
        try:
            end = pd.to_datetime(args.end, format=args.dt_format)
        except Exception:
            end = pd.to_datetime(args.end)
    else:
        end = inferred_end

    if end < start:
        print(f"ERROR: end < start ({fmt_ts(end)} < {fmt_ts(start)}).", file=sys.stderr)
        sys.exit(2)

    expected = build_expected_index(
        start=start,
        end=end,
        freq=args.freq,
        include_weekends=args.include_weekends,
        sunday_open_hour=args.sunday_open_hour,
    )

    missing, dup_count = find_missing_and_duplicates(pd.DatetimeIndex(s), expected)
    groups = group_consecutive_missing(missing, args.freq)

    # Summary
    print("==== Time Completeness Report ====")
    print(f"File: {args.file}")
    print(f"Datetime column: {args.datetime_col}")
    print(f"Frequency: {args.freq}")
    print(f"Observed range: {fmt_ts(inferred_start)} -> {fmt_ts(inferred_end)}")
    print(f"Validated range: {fmt_ts(expected[0])} -> {fmt_ts(expected[-1])} (n_expected={len(expected)})")
    print(f"Total rows: {len(s)} (unique timestamps: {len(pd.unique(s))})")
    print(f"Duplicates: {dup_count}")
    print(f"Missing timestamps: {len(missing)} ({(len(missing)/max(1,len(expected)))*100:.3f}%)")

    if len(missing) > 0:
        print("\nFirst missing timestamps:")
        for ts in list(missing[: args.show_missing]):
            print(f"  - {fmt_ts(ts)}")
        if len(missing) > args.show_missing:
            print(f"  ... and {len(missing) - args.show_missing} more")

        print("\nMissing ranges (contiguous groups):")
        for start_ts, end_ts, count in groups:
            if count == 1:
                print(f"  - {fmt_ts(start_ts)} (1 period)")
            else:
                print(f"  - {fmt_ts(start_ts)} -> {fmt_ts(end_ts)} ({count} periods)")
    else:
        print("\nNo missing timestamps in the validated range under current trading-hours rules.")

    # Guidance
    print("\nNotes:")
    if not args.include_weekends:
        if args.sunday_open_hour is None:
            print("  - Weekends excluded (Saturday & Sunday). Use --include-weekends to count weekends.")
            print("  - If you have Sunday evening trading hours, pass --sunday-open-hour, e.g., --sunday-open-hour 22")
        else:
            print(f"  - Weekends excluded, but including Sunday hours >= {args.sunday_open_hour}:00.")
    else:
        print("  - Weekends included in expected timeline.")

    if args.strict and len(missing) > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()