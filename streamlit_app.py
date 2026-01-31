from io import BytesIO
from pathlib import Path

import math

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import font_manager as fm
try:
    from PIL import Image
except Exception:  # Pillow is optional; fall back to raw PNG
    Image = None


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


def plot_location_chart(
    ax,
    plot_df,
    title,
    analyte_cols,
    color_map,
    bar_width,
    legend_outside=False,
):
    x = range(len(plot_df))
    date_labels = plot_df["sample_date"].dt.strftime("%Y-%m-%d")

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
    ax.set_title(title)
    ax.set_ylabel("Value")
    if legend_outside:
        ax.legend(title="", loc="upper left", bbox_to_anchor=(1.02, 1))
    else:
        ax.legend(title="", loc="upper left", frameon=False)


def optimize_png_bytes(png_bytes: bytes, compression_level: int = 9) -> bytes:
    if Image is None:
        return png_bytes
    try:
        img = Image.open(BytesIO(png_bytes))
        buf = BytesIO()
        img.save(buf, format="PNG", optimize=True, compress_level=compression_level)
        return buf.getvalue()
    except Exception:
        return png_bytes


def render_chart_png(
    plot_df,
    title,
    analyte_cols,
    color_map,
    bar_width,
    fig_width_in,
    fig_height_in,
    dpi=450,
):
    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in))
    plot_location_chart(
        ax,
        plot_df,
        title,
        analyte_cols,
        color_map,
        bar_width,
        legend_outside=True,
    )
    fig.tight_layout(pad=0.2)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    return optimize_png_bytes(buf.getvalue(), compression_level=PNG_COMPRESSION_LEVEL)


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

st.markdown(
    """
<style>
section[data-testid="stSidebar"] div[data-testid="stDownloadButton"] > button {
    background-color: #d64541;
    color: #ffffff;
    border: 1px solid #d64541;
}
section[data-testid="stSidebar"] div[data-testid="stDownloadButton"] > button:hover {
    background-color: #b83b36;
    border-color: #b83b36;
}
section[data-testid="stSidebar"] div[data-testid="stDownloadButton"] > button:focus {
    box-shadow: 0 0 0 0.2rem rgba(214, 69, 65, 0.35);
}
</style>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Data Source")
    uploaded = st.file_uploader("Excel file", type=["xlsx", "xls"])

    st.header("Filters")
    min_value = st.number_input("Minimum Reporting Limit (MRL)", min_value=0.0, value=2.0, step=0.1)

    st.header("Chart")
    bar_width = st.slider("Bar width", min_value=0.2, max_value=0.9, value=0.6, step=0.05)

    st.header("Export")
    sidebar_placeholder = st.empty()

if uploaded is None:
    st.info("Upload an Excel file to begin.")
    st.stop()

df = load_data(uploaded.getvalue())
test_name = Path(uploaded.name).stem.replace("_", " ").strip()

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

LETTER_WIDTH_IN = 8.5
LETTER_HEIGHT_IN = 11.0
WORD_NARROW_MARGIN_IN = 0.5
CAPTION_GAP_IN = 0.2
CAPTION_BOX_IN = 0.25
INTER_CHART_GAP_IN = 0.6
CONTENT_WIDTH_IN = LETTER_WIDTH_IN - (WORD_NARROW_MARGIN_IN * 2)
CHART_WIDTH_IN = CONTENT_WIDTH_IN
USABLE_HEIGHT_IN = LETTER_HEIGHT_IN - (WORD_NARROW_MARGIN_IN * 2)
CHART_HEIGHT_IN = max(
    (USABLE_HEIGHT_IN - (2 * CAPTION_GAP_IN + (2 * CAPTION_BOX_IN) + INTER_CHART_GAP_IN)) / 2,
    3.5,
)
INSIGHTS_CHART_WIDTH_IN = 12.0
INSIGHTS_CHART_HEIGHT_IN = 6.0
PDF_RENDER_WIDTH_IN = INSIGHTS_CHART_WIDTH_IN
PDF_RENDER_HEIGHT_IN = INSIGHTS_CHART_HEIGHT_IN
PDF_RENDER_DPI = 450
PNG_COMPRESSION_LEVEL = 9
PREVIEW_SCALE = 3.0

layout_width_in = CHART_WIDTH_IN + (WORD_NARROW_MARGIN_IN * 2)
layout_height_in = (
    (CHART_HEIGHT_IN * 2)
    + (CAPTION_GAP_IN * 2)
    + (CAPTION_BOX_IN * 2)
    + INTER_CHART_GAP_IN
    + (WORD_NARROW_MARGIN_IN * 2)
)
if layout_width_in > LETTER_WIDTH_IN or layout_height_in > LETTER_HEIGHT_IN:
    st.error("Layout exceeds letter size. Reduce chart height or gap.")
    st.stop()

if (
    (WORD_NARROW_MARGIN_IN + CAPTION_BOX_IN + CAPTION_GAP_IN) * 2
    + INTER_CHART_GAP_IN
    > (LETTER_HEIGHT_IN - (WORD_NARROW_MARGIN_IN * 2))
):
    st.error("Caption/gap settings exceed available page height.")
    st.stop()

location_items = []
for location in locations:
    filtered = df[df["location"] == location].copy()

    display_df = filtered.copy()
    if "lab_id" in display_df.columns:
        display_df["lab_id"] = display_df["lab_id"].astype("string")

    plot_df = filtered[["sample_date", "total"] + analyte_cols].copy()
    plot_df[analyte_cols] = plot_df[analyte_cols].where(plot_df[analyte_cols] >= min_value)
    plot_df = (
        plot_df.groupby("sample_date", as_index=False)[analyte_cols + ["total"]]
               .sum(min_count=1)
    )
    plot_df = plot_df.sort_values("sample_date")

    title = f"{test_name}\n{location}"
    location_items.append(
        {
            "location": location,
            "display_df": display_df,
            "plot_df": plot_df,
            "title": title,
            "caption": "",
        }
    )

report_items = []
figure_number = 1
for item in location_items:
    if item["plot_df"].empty:
        continue
    item["caption"] = f"Figure 1-{figure_number}. {test_name} – {item['location']}"
    report_items.append(item)
    figure_number += 1

pdf_bytes = None
preview_images = []
preview_error = None
if report_items:
    import fitz
    import matplotlib

    italic_font_path = str(
        (Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Oblique.ttf")
    )

    page_width_pt = LETTER_WIDTH_IN * 72.0
    page_height_pt = LETTER_HEIGHT_IN * 72.0
    margin_pt = WORD_NARROW_MARGIN_IN * 72.0
    content_width_pt = CONTENT_WIDTH_IN * 72.0
    chart_height_pt = CHART_HEIGHT_IN * 72.0
    caption_gap_pt = CAPTION_GAP_IN * 72.0
    caption_box_pt = CAPTION_BOX_IN * 72.0
    inter_gap_pt = INTER_CHART_GAP_IN * 72.0

    top_chart_top = page_height_pt - margin_pt
    top_chart_bottom = top_chart_top - chart_height_pt
    top_caption_top = top_chart_bottom - caption_gap_pt
    top_caption_bottom = top_caption_top - caption_box_pt

    bottom_chart_top = top_caption_bottom - inter_gap_pt
    bottom_chart_bottom = bottom_chart_top - chart_height_pt
    bottom_caption_top = bottom_chart_bottom - caption_gap_pt
    bottom_caption_bottom = bottom_caption_top - caption_box_pt
    pdf_doc = fitz.open()
    pdf_buffer = BytesIO()

    for i in range(0, len(report_items), 2):
        slots = report_items[i:i + 2]
        positions = [
            (top_chart_bottom, top_caption_bottom, top_caption_top),
            (bottom_chart_bottom, bottom_caption_bottom, bottom_caption_top),
        ]

        page = pdf_doc.new_page(width=page_width_pt, height=page_height_pt)

        for item, (chart_y, caption_bottom, caption_top) in zip(slots, positions):
            img_bytes = render_chart_png(
                item["plot_df"],
                item["title"],
                analyte_cols,
                color_map,
                bar_width,
                PDF_RENDER_WIDTH_IN,
                PDF_RENDER_HEIGHT_IN,
                dpi=PDF_RENDER_DPI,
            )
            img_arr = mpimg.imread(BytesIO(img_bytes))
            img_h, img_w = img_arr.shape[:2]
            box_w = content_width_pt
            box_h = chart_height_pt
            img_ratio = img_w / img_h
            box_ratio = box_w / box_h
            if img_ratio >= box_ratio:
                draw_w = box_w
                draw_h = box_w / img_ratio
            else:
                draw_h = box_h
                draw_w = box_h * img_ratio
            draw_x = margin_pt + (box_w - draw_w) / 2
            draw_y = chart_y + (box_h - draw_h) / 2
            rect = fitz.Rect(draw_x, draw_y, draw_x + draw_w, draw_y + draw_h)
            page.insert_image(rect, stream=img_bytes, keep_proportion=False)
            caption_rect = fitz.Rect(
                margin_pt,
                caption_bottom,
                margin_pt + content_width_pt,
                caption_top,
            )
            page.insert_textbox(
                caption_rect,
                item["caption"],
                fontsize=10,
                fontname="DejaVuSans-Oblique",
                fontfile=italic_font_path,
                align=1,
            )

    pdf_doc.save(pdf_buffer)
    pdf_doc.close()
    pdf_bytes = pdf_buffer.getvalue()
    sidebar_placeholder.download_button(
        "Download report (PDF)",
        pdf_bytes,
        file_name=f"{test_name}_report.pdf",
        mime="application/pdf",
    )

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(PREVIEW_SCALE, PREVIEW_SCALE), alpha=False)
            preview_images.append(pix.tobytes("png"))
    except Exception as exc:
        preview_error = str(exc)

print_tab, insights_tab = st.tabs(["Print Preview", "Insights"])

with print_tab:
    if not pdf_bytes:
        st.info("PDF preview will appear after data is processed.")
    else:
        if preview_error:
            st.error(f"PDF preview failed: {preview_error}")
        elif not preview_images:
            st.warning("PDF preview is unavailable.")
        else:
            for img in preview_images:
                st.image(img, width="stretch")

with insights_tab:
    for item in location_items:
        st.subheader(item["location"])
        st.dataframe(item["display_df"], width="stretch", height=240)

        if item["plot_df"].empty:
            st.warning("No data after filtering.")
            continue

        fig, ax = plt.subplots(figsize=(INSIGHTS_CHART_WIDTH_IN, INSIGHTS_CHART_HEIGHT_IN))
        plot_location_chart(
            ax,
            item["plot_df"],
            item["title"],
            analyte_cols,
            color_map,
            bar_width,
            legend_outside=True,
        )
        st.pyplot(fig, width="stretch")
