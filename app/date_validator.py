#!/usr/bin/env python3
import argparse
import sys
from typing import List, Tuple, Optional

import pandas as pd
from zoneinfo import ZoneInfo

try:
    import holidays as pyholidays  # optional
except Exception:
    pyholidays = None

# Try to import financial market calendars (NYSE, etc.)
try:
    # holidays>=0.31 provides financial calendars here
    from holidays.financial import NYSE as _NYSECal, NASDAQ as _NASDAQCal, USStockMarket as _USSMCal
    _FINANCIAL_CALS = {
        "NYSE": _NYSECal,
        "NASDAQ": _NASDAQCal,
        "USStockMarket": _USSMCal,
        "US_STOCK": _USSMCal,
    }
except Exception:
    _FINANCIAL_CALS = {}


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
        "--trading-profile",
        choices=["24x5", "forex-ny"],
        default="forex-ny",
        help="Trading-hours model. Default: forex-ny (Sun 17:00 NY to Fri 17:00 NY, DST-aware). Use '24x5' for Mon–Fri all hours.",
    )
    p.add_argument(
        "--session-tz",
        default="America/New_York",
        help="Session reference timezone (IANA name). Default: America/New_York",
    )
    p.add_argument(
        "--data-tz",
        default="UTC",
        help="Timezone of the CSV timestamps (assumed naive). Default: UTC",
    )
    p.add_argument(
        "--holiday-cal",
        default="USStockMarket",
        help="Holiday calendar code. Supports financial calendars: NYSE, NASDAQ, USStockMarket; "
             "or country codes for federal holidays (e.g., US). Default: USStockMarket (exclude full holiday dates). "
             "Use '--holiday-cal none' to disable holiday exclusion.",
    )
    p.add_argument(
        "--no-sunday-evening",
        action="store_true",
        default=True,
        help="When using 'forex-ny', do not expect Sunday evening session (treat Sunday as closed). Default: on.",
    )
    p.add_argument(
        "--holidays-file",
        default=None,
        help="Optional file with one YYYY-MM-DD per line to exclude as holidays.",
    )
    p.add_argument(
        "--lenient-yearend",
        dest="lenient_yearend",
        action="store_true",
        default=True,
        help="Treat common FX year-end special hours as closed (Dec 24/31 early close, Dec 26 late open, Jan 1 late open). Default: on.",
    )
    p.add_argument(
        "--no-lenient-yearend",
        dest="lenient_yearend",
        action="store_false",
        help="Disable year-end special-hours handling.",
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
    trading_profile: str = "forex-ny",
    session_tz: str = "America/New_York",
    data_tz: str = "UTC",
    holiday_cal: Optional[str] = "NYSE",
    holidays_file: Optional[str] = None,
    lenient_yearend: bool = True,
    no_sunday_evening: bool = True,
) -> pd.DatetimeIndex:
    # Align boundaries
    try:
        start_aligned = start.floor(freq)
        end_aligned = end.floor(freq)
    except Exception:
        start_aligned = start.floor("S")
        end_aligned = end.floor("S")

    # Build base range in data timezone as tz-aware
    rng_naive = pd.date_range(start=start_aligned, end=end_aligned, freq=freq)
    data_zone = ZoneInfo(data_tz)
    sess_zone = ZoneInfo(session_tz)
    rng = rng_naive.tz_localize(data_zone)

    # Holiday set (by session local date)
    holiday_dates = set()
    # Allow disabling via 'none'
    if holiday_cal and holiday_cal.lower() != "none":
        if pyholidays is None:
            print("WARNING: --holiday-cal set but 'holidays' package not installed. Run: pip install holidays", file=sys.stderr)
        else:
            # Determine years in session local time
            if len(rng) > 0:
                years = list(range(rng[0].tz_convert(sess_zone).year, rng[-1].tz_convert(sess_zone).year + 1))
            else:
                years = []
            try:
                hol = None
                if holiday_cal in _FINANCIAL_CALS:
                    hol = _FINANCIAL_CALS[holiday_cal](years=years)
                else:
                    hol = pyholidays.CountryHoliday(holiday_cal, years=years)
                # use dict-like keys directly
                holiday_dates |= set(hol.keys())
            except Exception as e:
                print(f"WARNING: Could not load holiday cal '{holiday_cal}': {e}", file=sys.stderr)
    if holidays_file:
        try:
            with open(holidays_file, "r") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    holiday_dates.add(pd.to_datetime(line).date())
        except Exception as e:
            print(f"WARNING: Could not read holidays file: {e}", file=sys.stderr)

    def is_trading_24x5(ts_data_tz) -> bool:
        # Simple Mon–Fri filter, optional Sunday evening override
        ts_utc = ts_data_tz.tz_convert("UTC")
        wd = ts_utc.weekday()  # Mon=0..Sun=6
        if wd < 5:
            return True
        if wd == 6 and sunday_open_hour is not None:
            return ts_utc.hour >= sunday_open_hour
        return False

    def is_trading_forex_ny(ts_data_tz) -> bool:
        # Convert to session local time (NY)
        ts_local = ts_data_tz.tz_convert(sess_zone)
        wd = ts_local.weekday()  # Mon=0..Sun=6
        hour = ts_local.hour
        m = ts_local.month
        d = ts_local.day

        # Exclude full holiday days by session local date
        if holiday_dates and ts_local.date() in holiday_dates:
            return False

        # Year-end special hours (broker conventions; DST-safe via session tz)
        if lenient_yearend:
            # Early close on Christmas Eve and New Year's Eve (≈14:00 NY)
            if (m, d) in ((12, 24), (12, 31)) and hour >= 14:
                return False
            # Late open on Boxing Day (Dec 26) (≈07:00 NY)
            if (m, d) == (12, 26) and hour < 7:
                return False
            # Late open on New Year's Day (Jan 1) (≈17:00 NY)
            if (m, d) == (1, 1) and hour < 17:
                return False

        # Sunday handling: optionally treat Sunday as fully closed
        if wd == 6:
            if no_sunday_evening:
                return False
            # Otherwise, open at 17:00 local (first hourly bar opens 17:00)
            return hour >= 17
        # Monday–Thursday: 24h
        if 0 <= wd <= 3:
            return True
        # Friday: trading until 17:00 local; last hourly bar opens at 16:00
        if wd == 4:
            return hour <= 16
        # Saturday: closed
        return False

    if trading_profile == "24x5":
        mask = [is_trading_24x5(ts) for ts in rng]
    else:
        # forex-ny (default)
        mask = [is_trading_forex_ny(ts) for ts in rng]

    expected = rng[mask]

    # Return naive timestamps back in data tz for comparison with CSV
    return expected.tz_convert(data_zone).tz_localize(None)


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
            start = pd.to_datetime(args.start)
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
        trading_profile=args.trading_profile,
        session_tz=args.session_tz,
        data_tz=args.data_tz,
        holiday_cal=args.holiday_cal,
        holidays_file=args.holidays_file,
        lenient_yearend=args.lenient_yearend,
        no_sunday_evening=args.no_sunday_evening,
    )

    missing, dup_count = find_missing_and_duplicates(pd.DatetimeIndex(s), expected)
    groups = group_consecutive_missing(missing, args.freq)

    # Summary
    print("==== Time Completeness Report ====")
    print(f"File: {args.file}")
    print(f"Datetime column: {args.datetime_col}")
    print(f"Frequency: {args.freq}")
    print(f"Observed range: {fmt_ts(inferred_start)} -> {fmt_ts(inferred_end)}")
    if len(expected) == 0:
        print("Validated range: (no expected timestamps after filters). Check trading profile/holidays settings.", file=sys.stderr)
        if args.strict:
            sys.exit(1)
        return
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
    print(f"  - Holiday calendar: {args.holiday_cal}")
    print(f"  - Year-end special hours: {'ON' if args.lenient_yearend else 'OFF'}")

    if args.strict and len(missing) > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()