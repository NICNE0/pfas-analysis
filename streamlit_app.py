import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


def clean_analyte_column(series: pd.Series) -> pd.Series:
    series = series.astype("string").str.replace(" J", "", regex=False)
    series = series.astype("string").str.replace("<", "", regex=False)
    return series


@st.cache_data
def load_data(path: str, skiprows: int) -> pd.DataFrame:
    df = pd.read_excel(path, skiprows=skiprows)

    analyte_cols = [
        "PFHpA/375-85-9",
        "PFHxS/355-46-4",
        "PFOA/335-67-1",
        "PFNA/375-95-1",
        "PFOS/1763-23-1",
        "PFDA/335-76-2",
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
    default_path = "files/clean_format.xlsx"
    use_upload = st.checkbox("Upload a file", value=False)
    uploaded = st.file_uploader("Excel file", type=["xlsx", "xls"]) if use_upload else None
    skiprows = st.number_input("Skip rows", min_value=0, max_value=50, value=8, step=1)

    st.header("Filters")
    min_value = st.number_input("Minimum analyte value", min_value=0.0, value=2.0, step=0.1)

    st.header("Chart")
    bar_width = st.slider("Bar width", min_value=0.2, max_value=0.9, value=0.6, step=0.05)

if uploaded is not None:
    data_path = uploaded
else:
    data_path = default_path

try:
    df = load_data(data_path, skiprows)
except FileNotFoundError:
    st.error(f"File not found: {default_path}")
    st.stop()

required_cols = {
    "sample_date",
    "location",
    "total",
    "PFHpA/375-85-9",
    "PFHxS/355-46-4",
    "PFOA/335-67-1",
    "PFNA/375-95-1",
    "PFOS/1763-23-1",
    "PFDA/335-76-2",
}
missing = sorted(required_cols - set(df.columns))
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

locations = sorted(df["location"].dropna().unique().tolist())
location = st.selectbox("Location", locations, index=locations.index("1 Amber Road") if "1 Amber Road" in locations else 0)

analyte_cols = [
    "PFHpA/375-85-9",
    "PFHxS/355-46-4",
    "PFOA/335-67-1",
    "PFNA/375-95-1",
    "PFOS/1763-23-1",
    "PFDA/335-76-2",
]

color_map = {
    "PFHpA/375-85-9": "#0B4F6C",
    "PFHxS/355-46-4": "#E67E22",
    "PFOA/335-67-1": "#1B5E20",
    "PFNA/375-95-1": "#C2185B",
    "PFOS/1763-23-1": "#7B1FA2",
    "PFDA/335-76-2": "#2E86DE",
}

filtered = df[df["location"] == location].copy()

plot_df = filtered[["sample_date", "total"] + analyte_cols].copy()
plot_df[analyte_cols] = plot_df[analyte_cols].where(plot_df[analyte_cols] >= min_value)
plot_df = (
    plot_df.groupby("sample_date", as_index=False)[analyte_cols + ["total"]]
           .sum(min_count=1)
)
plot_df = plot_df.sort_values("sample_date")

st.subheader("Preview")
st.dataframe(filtered.head(10), use_container_width=True)

if plot_df.empty:
    st.warning("No data after filtering.")
    st.stop()

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
            f"{total:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

ax.set_xticks(list(x))
ax.set_xticklabels(date_labels, rotation=45, ha="right")
ax.set_ylim(0, max_height + offset * 3)
ax.set_title("PRIVATE WELL SAMPLING RESULTS")
ax.set_ylabel("Value")
ax.legend(title="", loc="upper left", bbox_to_anchor=(1.02, 1))

st.pyplot(fig, use_container_width=True)
