"""Data loading and transformation utilities for the online spread dataset."""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, Iterable, Optional

import pandas as pd

LOGGER = logging.getLogger(__name__)

DEFAULT_SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1ijAq9nDO4XK0xnQ0HRBzQp8HLXzijPe3NodvrSDcF5c/export?format=csv&gid=0"
)

BETSLIPS_TARGET_MAPPING: Dict[str, str] = {
    "shots_total_for_team": "Shots",
    "shots_on_target_for_team": "Shots on target",
    "throw_ins_for_team": "Throw-ins",
    "goal_kicks_for_team": "Goal kicks",
    "fouls_for_team": "Fouls",
    "free_kicks_for_team": "Free kicks",
    "possession_for_team": "Possession",
    "tackles_for_team": "Tackles",
    "offsides_for_team": "Offsides",
    "cards_yellow_for_team": "Cards",
    "passes_total_for_team": "Passes",
    "corner_kicks_for_team": "Corners",
    "shots_total_for_player": "Player shots",
    "shots_on_target_for_player": "Player shots on target",
    "fouls_for_player": "Player fouls",
    "passes_for_player": "Player passes",
    "tackles_for_player": "Player tackles",
}


def normalise_sheet_url(raw_url: str) -> str:
    """Return a direct-download CSV URL for a Google Sheet (or the original URL)."""
    stripped = raw_url.strip()
    if not stripped:
        raise ValueError("Google Sheet URL is empty.")
    match = re.search(
        r"https://docs\.google\.com/spreadsheets/d/([a-zA-Z0-9\-_]+)", stripped
    )
    if not match:
        return stripped
    sheet_id = match.group(1)
    gid_match = re.search(r"(?:gid=)(\d+)", stripped)
    gid = gid_match.group(1) if gid_match else "0"
    return (
        f"https://docs.google.com/spreadsheets/d/{sheet_id}/export"
        f"?format=csv&gid={gid}"
    )


def read_online_spread_from_sheet(sheet_url: str) -> pd.DataFrame:
    """Download the online spread dataset directly from a Google Sheet."""
    download_url = normalise_sheet_url(sheet_url)
    LOGGER.info("Downloading online spread dataset from %s", download_url)
    frame = pd.read_csv(
        download_url,
        encoding="utf-8",
    )
    return frame


def determine_category_from_bet(
    bet: Any, betslips_target_mapping: Optional[Dict[str, str]] = None
) -> Optional[str]:
    """Return the target key best matching the bet description."""
    mapping = betslips_target_mapping or BETSLIPS_TARGET_MAPPING
    if not isinstance(bet, str):
        return None

    matches = {
        key: bool(re.search(rf"\b{value}\b", bet, re.IGNORECASE))
        for key, value in mapping.items()
    }
    if any(matches.values()):
        matched_lengths = {
            key: len(mapping[key]) for key, matched in matches.items() if matched
        }
        return max(matched_lengths.items(), key=lambda item: item[1])[0]
    return None


def normalise_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce dtypes, enrich fields, and keep only the required columns."""
    expected_columns = [
        "Match",
        "League",
        "Date",
        "Bet",
        "Odds",
        "Stake",
        "Bookmaker",
        "Result",
    ]
    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    normalised = df.copy()
    normalised["Date"] = pd.to_datetime(normalised["Date"], errors="coerce")
    normalised["Stake"] = pd.to_numeric(normalised["Stake"], errors="coerce")
    normalised["Result"] = pd.to_numeric(normalised["Result"], errors="coerce")
    normalised["Odds"] = pd.to_numeric(normalised["Odds"], errors="coerce")

    normalised = normalised.dropna(
        subset=["Match", "League", "Bookmaker", "Bet"], how="any"
    )
    normalised = normalised.sort_values("Date", na_position="last").reset_index(
        drop=True
    )

    normalised["market_key"] = normalised["Bet"].apply(
        lambda bet: determine_category_from_bet(bet, BETSLIPS_TARGET_MAPPING)
    )
    normalised["Market"] = normalised["market_key"].map(BETSLIPS_TARGET_MAPPING)

    def _infer_bet_type(row: pd.Series) -> Optional[str]:
        match = row.get("Match", "")
        bet = row.get("Bet", "")
        if not isinstance(match, str) or not isinstance(bet, str):
            return None
        home, _, away = match.partition(" - ")
        bet_lower = bet.lower()
        home_lower = home.strip().lower()
        away_lower = away.strip().lower()
        contains_home = bool(home_lower and home_lower in bet_lower)
        contains_away = bool(away_lower and away_lower in bet_lower)
        if contains_home and not contains_away:
            return "Home"
        if contains_away and not contains_home:
            return "Away"
        if ("total over" in bet_lower) or ("total under" in bet_lower):
            return "Total"
        return None

    def _infer_line_type(bet: Any) -> Optional[str]:
        if not isinstance(bet, str):
            return None
        bet_lower = bet.lower()
        if " over " in bet_lower:
            return "Over"
        if " under " in bet_lower:
            return "Under"
        return None

    normalised["BetType"] = normalised.apply(_infer_bet_type, axis=1)
    normalised["LineType"] = normalised["Bet"].apply(_infer_line_type)

    return normalised


def build_cumulative_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Create cumulative profit dataframe."""
    if df.empty:
        return pd.DataFrame(
            columns=["bet_number", "cumulative_result", "flat_cumulative", "Date"]
        )

    prepared = df.copy()
    prepared = prepared.sort_values("Date").reset_index(drop=True)
    prepared["bet_number"] = range(1, len(prepared) + 1)
    prepared["cumulative_result"] = prepared["Result"].cumsum()

    prepared["flat_result"] = 0.0
    win_mask = prepared["Result"] > 0
    loss_mask = prepared["Result"] < 0

    safe_stake = prepared["Stake"].replace(0, pd.NA)
    win_flat = prepared.loc[win_mask, "Result"] / safe_stake.loc[win_mask]
    odds_adjustment = (prepared.loc[win_mask, "Odds"] - 1).fillna(0.0)
    win_flat = win_flat.fillna(odds_adjustment)
    prepared.loc[win_mask, "flat_result"] = win_flat.astype(float)
    prepared.loc[loss_mask, "flat_result"] = -1.0
    prepared["flat_result"] = prepared["flat_result"].fillna(0.0)
    prepared["flat_cumulative"] = prepared["flat_result"].cumsum()

    return prepared[
        ["bet_number", "cumulative_result", "flat_cumulative", "Date"]
    ].copy()


def apply_filters(
    df: pd.DataFrame,
    leagues: Optional[Iterable[str]],
    bookmakers: Optional[Iterable[str]],
    markets: Optional[Iterable[str]],
    min_stake: Optional[float],
    start_date: Optional[str],
    end_date: Optional[str],
    bet_types: Optional[Iterable[str]],
    line_types: Optional[Iterable[str]],
) -> pd.DataFrame:
    """Filter dataframe according to optional selections."""
    filtered = df
    if leagues:
        leagues_lower = {league.lower() for league in leagues}
        filtered = filtered[filtered["League"].str.lower().isin(leagues_lower)]
    if bookmakers:
        bookmakers_lower = {bookmaker.lower() for bookmaker in bookmakers}
        filtered = filtered[filtered["Bookmaker"].str.lower().isin(bookmakers_lower)]
    if markets:
        markets_lower = {market.lower() for market in markets}
        filtered = filtered[
            filtered["Market"].fillna("").str.lower().isin(markets_lower)
        ]
    if min_stake is not None:
        filtered = filtered[filtered["Stake"].fillna(0.0) >= float(min_stake)]
    if start_date:
        try:
            start_ts = pd.to_datetime(start_date, errors="coerce")
            if pd.notna(start_ts):
                filtered = filtered[
                    filtered["Date"].fillna(pd.Timestamp.min) >= start_ts
                ]
        except Exception:  # pragma: no cover
            pass
    if end_date:
        try:
            end_ts = pd.to_datetime(end_date, errors="coerce")
            if pd.notna(end_ts):
                filtered = filtered[
                    filtered["Date"].fillna(pd.Timestamp.max) <= end_ts
                ]
        except Exception:  # pragma: no cover
            pass
    if bet_types:
        bet_types_set = {bt.lower() for bt in bet_types}
        filtered = filtered[
            filtered["BetType"].fillna("").str.lower().isin(bet_types_set)
        ]
    if line_types:
        line_types_set = {lt.lower() for lt in line_types}
        filtered = filtered[
            filtered["LineType"].fillna("").str.lower().isin(line_types_set)
        ]
    return filtered.copy()


def load_online_spread_dataset(sheet_url: Optional[str] = None) -> pd.DataFrame:
    """Load, concatenate, and normalise online spread data from a Google Sheet."""
    if sheet_url is None:
        candidate: Optional[str] = os.getenv("ONLINE_SPREAD_SHEET_URL")
    else:
        candidate = sheet_url

    if candidate is None:
        candidate = DEFAULT_SHEET_URL

    if not isinstance(candidate, str):
        raise ValueError("Google Sheet URL must be a string.")
    candidate = candidate.strip()
    if not candidate:
        raise ValueError(
            "ONLINE_SPREAD_SHEET_URL is empty. Provide a Google Sheet URL to load data."
        )
    frame = read_online_spread_from_sheet(candidate)
    LOGGER.info("Loaded online spread dataset from Google Sheet.")
    return normalise_dataset(frame)


__all__ = [
    "BETSLIPS_TARGET_MAPPING",
    "DEFAULT_SHEET_URL",
    "apply_filters",
    "build_cumulative_frame",
    "determine_category_from_bet",
    "load_online_spread_dataset",
    "normalise_dataset",
    "normalise_sheet_url",
    "read_online_spread_from_sheet",
]
