"""Streamlit dashboard for exploring the online spread dataset."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from spread_analytics import apply_filters, build_cumulative_frame, load_online_spread_dataset  # noqa: E402

BACKGROUND_COLOR = "#1a1d2e"
AXES_COLOR = "#FFFFFF"
COLOR_ACTUAL = "#FF8A3D"
COLOR_FLAT = "#FF4FA0"
NEUTRAL_COLOR = "#B0B7C3"
CURRENT_SEASON_START_DATE = "2025-09-13"


@st.cache_data(show_spinner=True)
def get_dataset(sheet_url: Optional[str]) -> pd.DataFrame:
    """Load and cache the online spread dataset from a Google Sheet."""
    return load_online_spread_dataset(sheet_url)


def build_filter_cache(df: pd.DataFrame) -> Dict[str, Any]:
    """Mirror the API filter cache structure for local use."""
    min_date = (
        df["Date"].dropna().min().date().isoformat()
        if "Date" in df.columns and not df["Date"].dropna().empty
        else None
    )
    max_date = (
        df["Date"].dropna().max().date().isoformat()
        if "Date" in df.columns and not df["Date"].dropna().empty
        else None
    )
    leagues = sorted([league for league in df["League"].dropna().unique()])
    bookmakers = sorted([bk for bk in df["Bookmaker"].dropna().unique()])
    markets = sorted(
        [
            market
            for market in df["Market"].dropna().unique()
            if isinstance(market, str) and market.strip()
        ]
    )
    bet_types_raw = [
        bet_type
        for bet_type in df["BetType"].dropna().unique()
        if isinstance(bet_type, str) and bet_type.strip()
    ]
    bet_type_preference = ["Home", "Away", "Total"]
    bet_types = [
        bt for bt in bet_type_preference if bt in bet_types_raw
    ] + sorted({bt for bt in bet_types_raw if bt not in bet_type_preference})

    line_types_raw = [
        line_type
        for line_type in df["LineType"].dropna().unique()
        if isinstance(line_type, str) and line_type.strip()
    ]
    line_type_preference = ["Over", "Under"]
    line_types = [
        lt for lt in line_type_preference if lt in line_types_raw
    ] + sorted({lt for lt in line_types_raw if lt not in line_type_preference})

    return {
        "leagues": leagues,
        "bookmakers": bookmakers,
        "markets": markets,
        "bet_types": bet_types,
        "line_types": line_types,
        "date_range": {"min": min_date, "max": max_date},
    }


def calculate_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Compute summary metrics for the filtered dataset."""
    metrics_df = df.dropna(subset=["Result", "Stake"])
    total_bets = int(len(df))
    total_result = float(metrics_df["Result"].sum())
    total_stake = float(metrics_df["Stake"].sum())
    roi = float((total_result / total_stake) * 100.0) if total_stake else 0.0
    return {
        "total_bets": total_bets,
        "total_result": round(total_result, 2),
        "total_stake": round(total_stake, 2),
        "roi": round(roi, 2),
    }


def build_minimum_unit_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return cumulative metrics for each minimum stake threshold."""
    empty_columns = [
        "Minimum Unit",
        "Bets",
        "Units Staked",
        "Units Returned",
        "ROI",
    ]
    if df.empty or "Stake" not in df.columns:
        return pd.DataFrame(columns=empty_columns)

    stake_values = df["Stake"].dropna().astype(float)
    if stake_values.empty:
        return pd.DataFrame(columns=empty_columns)

    thresholds = sorted(
        {round(float(value), 2) for value in stake_values if float(value) >= 0.0}
    )
    summary_rows = []
    for threshold in thresholds:
        tier_df = df[df["Stake"].fillna(0.0) >= threshold]
        if tier_df.empty:
            continue
        metrics = calculate_metrics(tier_df)
        summary_rows.append(
            {
                "Minimum Unit": round(threshold, 2),
                "Bets": metrics["total_bets"],
                "Units Staked": metrics["total_stake"],
                "Units Returned": metrics["total_result"],
                "ROI": metrics["roi"],
            }
        )

    return pd.DataFrame(summary_rows, columns=empty_columns)


def render_minimum_unit_table(df: pd.DataFrame) -> None:
    """Display metrics for each minimum unit threshold."""
    min_unit_summary = build_minimum_unit_summary(df)
    st.subheader("Results by Minimum Unit")
    if min_unit_summary.empty:
        st.info("No data available to build the minimum units table.")
        return

    display_summary = min_unit_summary.copy()
    display_summary["Minimum Unit"] = display_summary["Minimum Unit"].map(
        lambda value: f"{value:.2f} u"
    )
    display_summary["Bets"] = display_summary["Bets"].map(lambda value: f"{value:d}")
    display_summary["Units Staked"] = display_summary["Units Staked"].map(
        lambda value: f"{value:.2f}"
    )
    display_summary["Units Returned"] = display_summary["Units Returned"].map(
        lambda value: f"{value:.2f}"
    )
    display_summary["ROI"] = display_summary["ROI"].map(
        lambda value: f"{value:.2f}%"
    )

    st.dataframe(
        display_summary,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Minimum Unit": st.column_config.TextColumn("Minimum Unit"),
            "Bets": st.column_config.TextColumn("Bets"),
            "Units Staked": st.column_config.TextColumn("Units Staked"),
            "Units Returned": st.column_config.TextColumn("Units Returned"),
            "ROI": st.column_config.TextColumn("ROI"),
        },
    )


def plot_cumulative_chart(cumulative_df: pd.DataFrame) -> None:
    """Render cumulative profits chart."""
    if cumulative_df.empty:
        st.info("No settled bets available to build a cumulative chart.")
        return

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Inter",
                "Segoe UI",
                "Helvetica Neue",
                "Helvetica",
                "Arial",
                "sans-serif",
            ],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )

    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)

    ax.plot(
        cumulative_df["bet_number"],
        cumulative_df["cumulative_result"],
        color=COLOR_ACTUAL,
        linewidth=1.6,
        alpha=0.95,
        marker="",
        label="Cumulative Profit",
    )
    ax.plot(
        cumulative_df["bet_number"],
        cumulative_df["flat_cumulative"],
        color=COLOR_FLAT,
        linewidth=1.4,
        alpha=0.85,
        marker="",
        label="Cumulative Profit Flat Stakes",
    )

    ax.axhline(y=0, color=NEUTRAL_COLOR, linestyle="--", alpha=0.7, linewidth=2)

    ax.set_xlabel("Bets placed", fontweight="semibold", fontsize=9, color=AXES_COLOR)
    ax.set_ylabel("Units", fontweight="semibold", fontsize=9, color=AXES_COLOR)

    ax.grid(
        True,
        axis="y",
        color=NEUTRAL_COLOR,
        linestyle=":",
        linewidth=0.6,
        alpha=0.35,
    )
    ax.spines["left"].set_color(AXES_COLOR)
    ax.spines["bottom"].set_color(AXES_COLOR)
    ax.tick_params(axis="x", colors=AXES_COLOR, labelsize=8)
    ax.tick_params(axis="y", colors=AXES_COLOR, labelsize=8)

    legend = ax.legend(
        loc="upper left",
        frameon=True,
        fancybox=True,
        shadow=False,
        fontsize=6.75,
        borderpad=0.45,
        handlelength=1.95,
    )
    legend.get_frame().set_facecolor("#f5f5f5")
    legend.get_frame().set_alpha(0.8)

    st.pyplot(fig, use_container_width=True)


def render_summary(metrics: Dict[str, float]) -> None:
    """Display key metrics cards for the selected filters."""
    st.markdown(
        """
        <style>
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 8px;
            margin-bottom: 16px;
        }
        .summary-grid__cell {
            border: 1px solid rgba(255, 255, 255, 0.35);
            border-radius: 6px;
            padding: 10px 12px;
            display: flex;
            flex-direction: column;
            gap: 4px;
            background: rgba(255, 255, 255, 0.02);
        }
        .summary-grid__label,
        .summary-grid__value {
            font-size: 1.45rem;
            font-weight: 600;
            margin: 0;
            line-height: 1.15;
        }
        .summary-grid__label {
            color: rgba(255, 255, 255, 0.9);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.subheader("Summary")
    total_bets = metrics.get("total_bets", 0)
    total_result = metrics.get("total_result", 0.0)
    total_stake = metrics.get("total_stake", 0.0)
    roi = metrics.get("roi", 0.0)

    st.markdown(
        f"""
        <div class="summary-grid">
            <div class="summary-grid__cell">
                <p class="summary-grid__label">Bets</p>
                <p class="summary-grid__value">{total_bets}</p>
            </div>
            <div class="summary-grid__cell">
                <p class="summary-grid__label">Units Staked</p>
                <p class="summary-grid__value">{total_stake:.2f}</p>
            </div>
            <div class="summary-grid__cell">
                <p class="summary-grid__label">Units Returned</p>
                <p class="summary-grid__value">{total_result:.2f}</p>
            </div>
            <div class="summary-grid__cell">
                <p class="summary-grid__label">ROI</p>
                <p class="summary-grid__value">{roi:.2f}%</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_table(filtered_df: pd.DataFrame) -> None:
    """Render the filtered rows in a tabular layout."""
    if filtered_df.empty:
        st.warning("No bets match the current filter selection.")
        return

    display_frame = filtered_df.drop(columns=["market_key"], errors="ignore").rename(
        columns={
            "BetType": "Bet Type",
            "LineType": "Line Type",
        }
    )
    if "Date" in display_frame.columns:
        display_frame["Date"] = pd.to_datetime(
            display_frame["Date"], errors="coerce"
        ).dt.date

    preferred_order = [
        "Date",
        "Match",
        "Bet",
        "Market",
        "Bet Type",
        "Line Type",
        "Odds",
        "Stake",
        "Bookmaker",
        "Result",
        "League",
    ]
    columns_present = [col for col in preferred_order if col in display_frame.columns]
    display_frame = display_frame.loc[:, columns_present]

    st.dataframe(
        display_frame,
        hide_index=True,
        use_container_width=True,
    )


def main() -> None:
    """Entry point for the Streamlit dashboard."""
    st.set_page_config(
        page_title="Online Spread Analytics Dashboard",
        page_icon="📊",
        layout="wide",
    )
    st.title("Online Spread Analytics Dashboard")

    try:
        dataset = get_dataset(os.getenv("ONLINE_SPREAD_SHEET_URL"))
    except Exception as exc:  # pylint: disable=broad-except
        st.error(f"Unable to load dataset from Google Sheet: {exc}")
        return

    filters = build_filter_cache(dataset)
    league_order = [
        "Premier League",
        "La Liga",
        "Serie A",
        "Bundesliga",
        "Ligue 1",
        "Championship",
        "Eredivisie",
        "Scottish Premiership",
        "Brasileiro Serie A",
        "MLS",
    ]
    raw_leagues = filters.get("leagues", [])
    league_options = [
        league for league in league_order if league in raw_leagues
    ] + sorted({league for league in raw_leagues if league not in league_order})
    bookmaker_options = filters.get("bookmakers", [])
    market_options = filters.get("markets", [])
    bet_type_options = filters.get("bet_types", [])
    line_type_options = filters.get("line_types", [])
    date_range = filters.get("date_range", {})
    start_date_default = date_range.get("min") or ""
    default_end = date_range.get("max") or ""

    logo_path = Path(__file__).resolve().parents[2] / "telegram_logo.png"
    if logo_path.exists():
        logo_bytes = logo_path.read_bytes()
        logo_base64 = base64.b64encode(logo_bytes).decode("utf-8")
        st.sidebar.markdown(
            f"""
            <div style="display:flex;justify-content:center;padding:8px 0;">
              <img src="data:image/png;base64,{logo_base64}" style="width:50%;border:3px solid #ffffff;border-radius:50%;box-shadow:0 0 4px rgba(0,0,0,0.25);">
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.sidebar.header("Filters")
    selected_leagues = st.sidebar.multiselect(
        "Leagues",
        options=league_options,
        default=[],
        placeholder="Leave empty to include all leagues.",
    )
    selected_bookmakers = st.sidebar.multiselect(
        "Bookmakers",
        options=bookmaker_options,
        default=[],
        placeholder="Leave empty to include all bookmakers.",
    )
    selected_markets = st.sidebar.multiselect(
        "Markets",
        options=market_options,
        default=[],
        placeholder="Leave empty to include all markets.",
    )
    min_stake_value = st.sidebar.number_input(
        "Units limit (stake ≥)",
        min_value=0.0,
        step=0.25,
        value=0.5,
        format="%.2f",
        help="Only include bets with stake greater than or equal to this value.",
    )
    start_date_key = "start_date_input"
    end_date_key = "end_date_input"
    end_date_default_key = "end_date_default"
    current_season_key = "current_season_toggle"
    if start_date_key not in st.session_state:
        st.session_state[start_date_key] = start_date_default or ""
    if end_date_key not in st.session_state:
        st.session_state[end_date_key] = default_end or ""
    if end_date_default_key not in st.session_state:
        st.session_state[end_date_default_key] = default_end or ""
    if current_season_key not in st.session_state:
        st.session_state[current_season_key] = False
    if st.session_state[current_season_key]:
        st.session_state[start_date_key] = CURRENT_SEASON_START_DATE
    elif st.session_state[start_date_key] == CURRENT_SEASON_START_DATE:
        st.session_state[start_date_key] = start_date_default or ""
    previous_default = st.session_state[end_date_default_key]
    if previous_default != (default_end or ""):
        current_end_value = st.session_state[end_date_key]
        if current_end_value in ("", previous_default):
            st.session_state[end_date_key] = default_end or ""
        st.session_state[end_date_default_key] = default_end or ""
    col_start_date, col_end_date = st.sidebar.columns(2)
    col_start_date.text_input(
        "Start date",
        key=start_date_key,
    )
    col_end_date.text_input(
        "End date",
        key=end_date_key,
    )
    st.sidebar.checkbox(
        "Current season",
        key=current_season_key,
    )
    start_date_input = (st.session_state[start_date_key] or "").strip()
    end_date_input = (st.session_state[end_date_key] or "").strip()
    selected_bet_types: List[str] = []
    selected_line_types: List[str] = []
    with st.sidebar.expander("Additional filters", expanded=False):
        if bet_type_options:
            st.markdown("**Bet type**")
            for option in bet_type_options:
                option_key = option.replace(" ", "_").lower()
                if st.checkbox(
                    option,
                    key=f"bet_type_{option_key}",
                ):
                    selected_bet_types.append(option)
        if line_type_options:
            st.markdown("**Line type**")
            for option in line_type_options:
                option_key = option.replace(" ", "_").lower()
                if st.checkbox(
                    option,
                    key=f"line_type_{option_key}",
                ):
                    selected_line_types.append(option)
    min_stake_param = min_stake_value if min_stake_value > 0 else None
    start_date_param = start_date_input.strip() or None
    end_date_param = end_date_input.strip() or None

    filtered_without_min_stake = apply_filters(
        dataset,
        selected_leagues or None,
        selected_bookmakers or None,
        selected_markets or None,
        None,
        start_date_param,
        end_date_param,
        selected_bet_types or None,
        selected_line_types or None,
    )

    if min_stake_param is not None:
        working_df = filtered_without_min_stake[
            filtered_without_min_stake["Stake"].fillna(0.0) >= float(min_stake_param)
        ]
    else:
        working_df = filtered_without_min_stake

    metrics = calculate_metrics(working_df)
    render_summary(metrics)

    cumulative_input = working_df.dropna(subset=["Date", "Result", "Stake"])
    cumulative_df = build_cumulative_frame(cumulative_input)
    if not cumulative_df.empty:
        cumulative_df["bet_number"] = pd.to_numeric(
            cumulative_df["bet_number"], errors="coerce"
        )
        cumulative_df["cumulative_result"] = pd.to_numeric(
            cumulative_df["cumulative_result"], errors="coerce"
        )
        cumulative_df["flat_cumulative"] = pd.to_numeric(
            cumulative_df["flat_cumulative"], errors="coerce"
        )
        cumulative_df = cumulative_df.dropna(
            subset=["bet_number", "cumulative_result", "flat_cumulative"]
        )

    plot_cumulative_chart(cumulative_df)

    render_minimum_unit_table(filtered_without_min_stake)

    st.subheader("Selections")
    render_table(working_df)


if __name__ == "__main__":
    main()
