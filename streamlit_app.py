from io import BytesIO

import math

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


def clean_analyte_column(series: pd.Series) -> pd.Series:
    series = series.astype("string").str.replace(" J", "", regex=False)
    series = series.astype("string").str.replace("<", "", regex=False)
    return series


def round_sigfigs(value: float, sig_figs: int = 3) -> float:
    if value == 0:
        return 0.0
    return round(value, sig_figs - int(math.floor(math.log10(abs(value)))) - 1)


def format_value(value: float) -> str:
    if pd.isna(value):
        return ""
    rounded = round_sigfigs(float(value), 3)
    abs_val = abs(rounded)
    if abs_val >= 1000:
        return f"{rounded:,.0f}"
    if abs_val >= 100:
        return f"{rounded:.0f}"
    if abs_val >= 10:
        return f"{rounded:.1f}"
    if abs_val >= 1:
        return f"{rounded:.2f}"
    return f"{rounded:.3f}"


@st.cache_data
def load_data(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_excel(BytesIO(file_bytes))

    analyte_cols = [
        "PFHpA",
        "PFHxS",
        "PFOA",
        "PFNA",
        "PFOS",
        "PFDA",
    ]

    for col in analyte_cols:
        if col in df.columns:
            df[col] = clean_analyte_column(df[col])

    if "total" in df.columns:
        df["total"] = df["total"].astype("string").replace("ND", pd.NA).astype("Float64")

    for col in analyte_cols + ["total"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "sample_date" in df.columns:
        df["sample_date"] = pd.to_datetime(df["sample_date"], errors="coerce")

    return df


st.set_page_config(page_title="PFAS Analysis", layout="wide")

st.title("PFAS Analysis")

with st.sidebar:
    st.header("Data Source")
    uploaded = st.file_uploader("Excel file", type=["xlsx", "xls"])

    st.header("Filters")
    min_value = st.number_input("Minimum analyte value", min_value=0.0, value=2.0, step=0.1)

    st.header("Chart")
    bar_width = st.slider("Bar width", min_value=0.2, max_value=0.9, value=0.6, step=0.05)

if uploaded is None:
    st.info("Upload an Excel file to begin.")
    st.stop()

df = load_data(uploaded.getvalue())

required_cols = {
    "sample_date",
    "location",
    "total",
    "PFHpA",
    "PFHxS",
    "PFOA",
    "PFNA",
    "PFOS",
    "PFDA",
}
missing = sorted(required_cols - set(df.columns))
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

locations = sorted(df["location"].dropna().unique().tolist())

analyte_cols = [
    "PFHpA",
    "PFHxS",
    "PFOA",
    "PFNA",
    "PFOS",
    "PFDA",
]

color_map = {
    "PFHpA": "#0B4F6C",
    "PFHxS": "#E67E22",
    "PFOA": "#1B5E20",
    "PFNA": "#C2185B",
    "PFOS": "#7B1FA2",
    "PFDA": "#2E86DE",
}

if not locations:
    st.warning("No locations found in the uploaded file.")
    st.stop()

for location in locations:
    st.subheader(location)
    filtered = df[df["location"] == location].copy()

    display_df = filtered.copy()
    if "lab_id" in display_df.columns:
        display_df["lab_id"] = display_df["lab_id"].astype("string")
    st.dataframe(display_df, width="stretch", height=240)

    plot_df = filtered[["sample_date", "total"] + analyte_cols].copy()
    plot_df[analyte_cols] = plot_df[analyte_cols].where(plot_df[analyte_cols] >= min_value)
    plot_df = (
        plot_df.groupby("sample_date", as_index=False)[analyte_cols + ["total"]]
               .sum(min_count=1)
    )
    plot_df = plot_df.sort_values("sample_date")

    if plot_df.empty:
        st.warning("No data after filtering.")
        continue

    x = range(len(plot_df))
    date_labels = plot_df["sample_date"].dt.strftime("%Y-%m-%d")

    fig, ax = plt.subplots(figsize=(12, 6))

    bottom = pd.Series([0.0] * len(plot_df))
    for col in analyte_cols:
        values = plot_df[col].fillna(0)
        ax.bar(x, values, bottom=bottom, label=col, color=color_map.get(col), width=bar_width)
        bottom += values

    max_height = bottom.max() if len(bottom) else 0
    offset = max_height * 0.02 if max_height else 1
    for xi, total in zip(x, plot_df["total"]):
        if pd.notna(total):
            ax.text(
                xi,
                bottom.iloc[xi] + offset,
                format_value(total),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(list(x))
    ax.set_xticklabels(date_labels, rotation=45, ha="right")
    ax.set_ylim(0, max_height + offset * 3)
    ax.set_title(f"{location} — PRIVATE WELL SAMPLING RESULTS")
    ax.set_ylabel("Value")
    ax.legend(title="", loc="upper left", bbox_to_anchor=(1.02, 1))

    st.pyplot(fig, width="stretch")
