"""Streamlit dashboard for exploring the online spread dataset."""

from __future__ import annotations

import os
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


def plot_cumulative_chart(cumulative_df: pd.DataFrame) -> None:
    """Render cumulative profits chart."""
    if cumulative_df.empty:
        st.info("No settled bets available to build a cumulative chart.")
        return

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
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

    ax.set_title(
        "Cumulative Profit",
        fontweight="bold",
        color=AXES_COLOR,
        fontsize=14,
        pad=20,
    )
    ax.set_xlabel("Bet Number", fontweight="bold", fontsize=12, color=AXES_COLOR)
    ax.set_ylabel("Units", fontweight="bold", fontsize=12, color=AXES_COLOR)

    ax.grid(False)
    ax.spines["left"].set_color(AXES_COLOR)
    ax.spines["bottom"].set_color(AXES_COLOR)
    ax.tick_params(axis="x", colors=AXES_COLOR)
    ax.tick_params(axis="y", colors=AXES_COLOR)

    legend = ax.legend(
        loc="upper left",
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=10,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.85)

    st.pyplot(fig, use_container_width=True)


def render_summary(metrics: Dict[str, float]) -> None:
    """Display key metrics cards for the selected filters."""
    st.subheader("Summary")
    total_bets = metrics.get("total_bets", 0)
    total_result = metrics.get("total_result", 0.0)
    total_stake = metrics.get("total_stake", 0.0)
    roi = metrics.get("roi", 0.0)

    col_bets, col_units, col_stake, col_roi = st.columns(4)
    col_bets.metric("Bets", f"{total_bets:,}")
    col_units.metric("Total Units", f"{total_result:.2f}")
    col_stake.metric("Total Stake", f"{total_stake:.2f}")
    col_roi.metric("ROI", f"{roi:.2f}%")


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
        page_title="Online Spread Dashboard",
        page_icon="📊",
        layout="wide",
    )
    st.title("Online Spread Dashboard")

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
    default_start = date_range.get("min") or ""
    default_end = date_range.get("max") or ""

    st.sidebar.header("Filters")
    excluded_leagues = {"MLS", "Brasileiro Serie A"}
    default_leagues = [
        league for league in league_options if league not in excluded_leagues
    ]
    selected_leagues = st.sidebar.multiselect(
        "Leagues",
        options=league_options,
        default=default_leagues or league_options,
    )
    selected_bookmakers = st.sidebar.multiselect(
        "Bookmakers",
        options=bookmaker_options,
        default=bookmaker_options,
    )
    excluded_markets = {"Possession", "Corners", "Shots on target"}
    default_markets = [
        market for market in market_options if market not in excluded_markets
    ]
    selected_markets = st.sidebar.multiselect(
        "Markets",
        options=market_options,
        default=default_markets or market_options,
    )
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
    min_stake_value = st.sidebar.number_input(
        "Units limit (stake ≥)",
        min_value=0.0,
        step=0.25,
        value=0.5,
        help="Only include bets with stake greater than or equal to this value.",
    )
    start_date_input = st.sidebar.text_input(
        "Start date (YYYY-MM-DD)",
        value=default_start,
    )
    end_date_input = st.sidebar.text_input(
        "End date (YYYY-MM-DD)",
        value=default_end,
    )
    min_stake_param = min_stake_value if min_stake_value > 0 else None
    start_date_param = start_date_input.strip() or None
    end_date_param = end_date_input.strip() or None
    show_results = st.sidebar.button("Display results")

    if show_results:
        working_df = apply_filters(
            dataset,
            selected_leagues or None,
            selected_bookmakers or None,
            selected_markets or None,
            min_stake_param,
            start_date_param,
            end_date_param,
            selected_bet_types or None,
            selected_line_types or None,
        )
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

        st.subheader("Selections")
        render_table(working_df)
    else:
        st.info("Select filters and click 'Display results' to view metrics and charts.")


if __name__ == "__main__":
    main()
