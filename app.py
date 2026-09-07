# app.py
# RheaVita vial temperature viewer
# Run with:
#   streamlit run app.py

import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import pyodbc

try:
    from scipy.signal import savgol_filter
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Vial Temperature Viewer",
    page_icon="🌡️",
    layout="wide",
)

st.title("🌡️ Vial Temperature Viewer")
st.caption(
    "Select a SQL database, choose vials, apply smoothing/alignment, "
    "compare temperature profiles and export the results."
)


# ============================================================
# HELPERS
# ============================================================

def safe_identifier(name: str) -> str:
    """
    Validate a SQL Server identifier that was retrieved from SQL Server itself.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Invalid SQL identifier.")
    return "[" + name.replace("]", "]]") + "]"


def get_sql_driver():
    """
    Use the same legacy driver as the original Spyder script.
    """
    return "SQL Server"



def connection_string(server, database):
    """
    Match the original Spyder/pyodbc connection style.
    """
    return (
        r"driver={SQL Server};"
        f"server={server};"
        f"database={database};"
        r"trusted_connection=YES;"
    )





def test_connection(server, database="GMP_fast_roche"):
    """
    Test the exact SQL connection style used in the original Spyder script.
    """
    try:
        conx = pyodbc.connect(
            connection_string(server, database),
            timeout=10,
        )
        cursor = conx.cursor()
        cursor.execute(
            "SELECT @@SERVERNAME AS ServerName, "
            "DB_NAME() AS DatabaseName, "
            "SYSTEM_USER AS LoginName"
        )
        row = cursor.fetchone()
        conx.close()

        return True, {
            "server": row[0] if row else server,
            "database": row[1] if row else database,
            "login": row[2] if row else "",
        }

    except Exception as exc:
        return False, {
            "error_type": type(exc).__name__,
            "error_repr": repr(exc),
            "error_args": [str(x) for x in getattr(exc, "args", [])],
        }


@st.cache_data(ttl=60, show_spinner=False)
def list_databases(server):
    conx = pyodbc.connect(
        connection_string(server, "master"),
        timeout=10,
    )
    query = """
    SELECT name
    FROM sys.databases
    WHERE state_desc = 'ONLINE'
    ORDER BY name;
    """
    dbs = pd.read_sql(query, conx)["name"].tolist()
    conx.close()
    return dbs


@st.cache_data(ttl=60, show_spinner=False)
def get_vial_ids(server, database):
    db = safe_identifier(database)

    conx = pyodbc.connect(
        connection_string(server, database),
        timeout=10,
    )

    query = f"""
    SELECT DISTINCT vd.[VialId]
    FROM {db}.[dbo].[VialData] vd
    WHERE vd.[VialId] IS NOT NULL
    ORDER BY vd.[VialId];
    """

    values = pd.read_sql(query, conx)["VialId"].dropna().astype(int).tolist()
    conx.close()
    return values


@st.cache_data(ttl=60, show_spinner=False)
def load_temperature_data(
    server,
    database,
    vial_ids_tuple,
    sample_every,
    device_filter_mode,
    device_ids_tuple,
):
    if not vial_ids_tuple:
        return pd.DataFrame()

    db = safe_identifier(database)
    vial_ids = list(vial_ids_tuple)
    vial_placeholders = ",".join("?" for _ in vial_ids)

    device_clause = ""
    params = list(vial_ids)

    if device_ids_tuple:
        device_placeholders = ",".join("?" for _ in device_ids_tuple)
        if device_filter_mode == "Include only":
            device_clause = f" AND sd.[DeviceData_Id] IN ({device_placeholders}) "
        else:
            device_clause = f" AND sd.[DeviceData_Id] NOT IN ({device_placeholders}) "
        params.extend(list(device_ids_tuple))

    # ROW_NUMBER is partitioned by vial so downsampling is performed
    # consistently within each vial rather than globally.
    query = f"""
    WITH SourceData AS (
        SELECT
            sd.[Id],
            sd.[DeviceData_Id],
            TRY_CONVERT(float, sd.[Value]) AS [Value],
            vd.[VialId],
            vd.[SignalDataId],
            vd.[Timestamp],
            MIN(vd.[Timestamp]) OVER (
                PARTITION BY vd.[VialId]
            ) AS MinTimestamp,
            ROW_NUMBER() OVER (
                PARTITION BY vd.[VialId]
                ORDER BY vd.[Timestamp], sd.[Id]
            ) AS VialRowNum
        FROM {db}.[dbo].[SignalData] sd
        INNER JOIN {db}.[dbo].[VialData] vd
            ON sd.[Id] = vd.[SignalDataId]
        WHERE vd.[VialId] IN ({vial_placeholders})
        {device_clause}
    )
    SELECT
        [Id],
        [DeviceData_Id],
        [Value],
        [VialId],
        [SignalDataId],
        [Timestamp],
        DATEDIFF(SECOND, MinTimestamp, [Timestamp]) AS RelativeTimestamp
    FROM SourceData
    WHERE [Value] IS NOT NULL
      AND ((VialRowNum - 1) % ? = 0)
    ORDER BY [VialId], [Timestamp];
    """

    params.append(int(sample_every))

    conx = pyodbc.connect(
        connection_string(server, database),
        timeout=20,
    )

    df = pd.read_sql(query, conx, params=params)
    conx.close()

    if not df.empty:
        df["VialId"] = df["VialId"].astype(int)
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
        df["RelativeTimestamp"] = pd.to_numeric(
            df["RelativeTimestamp"], errors="coerce"
        )
        df = df.dropna(subset=["Value", "RelativeTimestamp"])

    return df


def robust_rolling_mean(series, window=15, mad_factor=2.0, min_value=None):
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    output = np.full(len(values), np.nan)

    for i in range(len(values)):
        start = max(0, i - window + 1)
        x = values[start:i + 1]
        x = x[np.isfinite(x)]

        if min_value is not None:
            x = x[x >= min_value]

        if len(x) == 0:
            continue

        median = np.median(x)
        mad = np.median(np.abs(x - median))

        if mad <= 1e-12:
            output[i] = np.mean(x)
            continue

        keep = np.abs(x - median) <= mad_factor * mad
        output[i] = np.mean(x[keep]) if np.any(keep) else median

    return pd.Series(output, index=series.index)


def smooth_series(series, method, window, robust_factor, min_value, polyorder):
    s = pd.to_numeric(series, errors="coerce").astype(float)

    if method == "None":
        return s

    if method == "Rolling mean":
        return s.rolling(window=window, min_periods=1, center=True).mean()

    if method == "Rolling median":
        return s.rolling(window=window, min_periods=1, center=True).median()

    if method == "Robust rolling mean":
        return robust_rolling_mean(
            s,
            window=window,
            mad_factor=robust_factor,
            min_value=min_value,
        )

    if method == "Savitzky-Golay":
        if not SCIPY_AVAILABLE:
            return s.rolling(window=window, min_periods=1, center=True).mean()

        valid = s.interpolate(limit_direction="both").to_numpy()
        n = len(valid)

        if n < 5:
            return s

        win = min(window, n if n % 2 == 1 else n - 1)
        win = max(win, polyorder + 2)

        if win % 2 == 0:
            win -= 1

        if win <= polyorder or win < 3:
            return s

        return pd.Series(
            savgol_filter(valid, window_length=win, polyorder=polyorder),
            index=s.index,
        )

    return s


def calculate_alignment_time(vial_df, mode, threshold, manual_offset_min):
    t = vial_df["RelativeTimestamp"].to_numpy(dtype=float)
    y = vial_df["SmoothedValue"].to_numpy(dtype=float)

    if len(t) == 0:
        return 0.0

    if mode == "No alignment":
        return 0.0

    if mode == "Minimum temperature":
        if np.all(np.isnan(y)):
            return 0.0
        return float(t[np.nanargmin(y)])

    if mode == "First below threshold":
        idx = np.where(y <= threshold)[0]
        return float(t[idx[0]]) if len(idx) else 0.0

    if mode == "Manual offset":
        return float(manual_offset_min) * 60.0

    return 0.0


def prepare_data(
    raw_df,
    smoothing_method,
    smoothing_window,
    robust_factor,
    min_value,
    polyorder,
    alignment_mode,
    threshold,
    manual_offsets,
    plot_time_offset_min,
):
    frames = []

    for vial_id, vial_df in raw_df.groupby("VialId"):
        vial_df = vial_df.sort_values("RelativeTimestamp").copy()

        vial_df["SmoothedValue"] = smooth_series(
            vial_df["Value"],
            method=smoothing_method,
            window=smoothing_window,
            robust_factor=robust_factor,
            min_value=min_value,
            polyorder=polyorder,
        )

        align_time = calculate_alignment_time(
            vial_df,
            mode=alignment_mode,
            threshold=threshold,
            manual_offset_min=manual_offsets.get(int(vial_id), 0.0),
        )

        vial_df["AlignedTime_min"] = (
            (vial_df["RelativeTimestamp"] - align_time) / 60.0
            + plot_time_offset_min
        )

        frames.append(vial_df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def make_figure(
    processed_df,
    show_raw,
    show_mean,
    show_std,
    pd_setpoint,
    sd_setpoint,
    freezing_start,
    freezing_end,
    x_min,
    x_max,
    y_min,
    y_max,
    linewidth,
    alpha,
):
    fig, ax = plt.subplots(figsize=(12, 7))

    vial_ids = sorted(processed_df["VialId"].unique())

    for vial_id in vial_ids:
        d = processed_df[processed_df["VialId"] == vial_id].sort_values(
            "AlignedTime_min"
        )

        if show_raw:
            ax.plot(
                d["AlignedTime_min"],
                d["Value"],
                linewidth=0.6,
                alpha=0.18,
            )

        ax.plot(
            d["AlignedTime_min"],
            d["SmoothedValue"],
            label=f"Vial {vial_id}",
            linewidth=linewidth,
            alpha=alpha,
        )

    if show_mean and len(vial_ids) > 1:
        min_t = max(
            processed_df.loc[
                processed_df["VialId"] == v, "AlignedTime_min"
            ].min()
            for v in vial_ids
        )
        max_t = min(
            processed_df.loc[
                processed_df["VialId"] == v, "AlignedTime_min"
            ].max()
            for v in vial_ids
        )

        if min_t < max_t:
            grid = np.linspace(min_t, max_t, 1500)
            curves = []

            for vial_id in vial_ids:
                d = processed_df[
                    processed_df["VialId"] == vial_id
                ].sort_values("AlignedTime_min")

                x = d["AlignedTime_min"].to_numpy()
                y = d["SmoothedValue"].to_numpy()

                valid = np.isfinite(x) & np.isfinite(y)
                x = x[valid]
                y = y[valid]

                if len(x) >= 2:
                    curves.append(np.interp(grid, x, y))

            if curves:
                arr = np.vstack(curves)
                mean = np.nanmean(arr, axis=0)
                std = np.nanstd(arr, axis=0, ddof=1)

                ax.plot(
                    grid,
                    mean,
                    linewidth=3,
                    label="Mean",
                )

                if show_std and len(curves) > 1:
                    ax.fill_between(
                        grid,
                        mean - std,
                        mean + std,
                        alpha=0.18,
                        label="±1 SD",
                    )

    if pd_setpoint is not None:
        ax.axhline(
            pd_setpoint,
            linestyle="--",
            linewidth=1.2,
            label=f"PD setpoint ({pd_setpoint:g} K)",
        )

    if sd_setpoint is not None:
        ax.axhline(
            sd_setpoint,
            linestyle="--",
            linewidth=1.2,
            label=f"SD setpoint ({sd_setpoint:g} K)",
        )

    if freezing_start < freezing_end:
        ax.axvspan(
            freezing_start,
            freezing_end,
            alpha=0.08,
            label="Freezing",
        )

    ax.set_xlabel("Aligned Time (min)")
    ax.set_ylabel("Temperature (K)")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    fig.tight_layout()

    return fig


# ============================================================
# SESSION STATE
# ============================================================

if "databases" not in st.session_state:
    st.session_state.databases = []

if "available_vials" not in st.session_state:
    st.session_state.available_vials = []

if "raw_data" not in st.session_state:
    st.session_state.raw_data = pd.DataFrame()


# ============================================================
# SIDEBAR - CONNECTION
# ============================================================

st.sidebar.header("1. Database")

SQL_DRIVER = "SQL Server"
SQL_SERVER = r"RV-PC-15\SQLEXPRESS"

st.sidebar.text_input(
    "SQL driver",
    value=SQL_DRIVER,
    disabled=True,
)

st.sidebar.text_input(
    "SQL Server",
    value=SQL_SERVER,
    disabled=True,
)

st.sidebar.checkbox(
    "Use Windows authentication",
    value=True,
    disabled=True,
)

database_hint = st.sidebar.text_input(
    "Default database",
    value="GMP_fast_roche",
)

if st.sidebar.button("Connect / refresh databases", use_container_width=True):
    with st.spinner("Testing SQL Server connection..."):
        ok, details = test_connection(
            SQL_SERVER,
            database=database_hint,
        )

    if ok:
        st.sidebar.success(
            f"Connected to {details['server']} / {details['database']}"
        )
        if details.get("login"):
            st.sidebar.caption(f"Login: {details['login']}")

        try:
            with st.spinner("Reading databases..."):
                st.session_state.databases = list_databases(
                    SQL_SERVER,
                )
            st.sidebar.success(
                f"{len(st.session_state.databases)} databases found."
            )
        except Exception as e:
            st.sidebar.warning(
                "Direct connection works, but the database list could not be read."
            )
            st.sidebar.code(repr(e))
            st.session_state.databases = [database_hint]

    else:
        st.sidebar.error("SQL Server could not be reached.")
        st.sidebar.code(
            f"Type: {details['error_type']}\n"
            f"repr: {details['error_repr']}\n"
            f"args: {details['error_args']}"
        )
        st.sidebar.info(
            "This app is now using the exact same driver/server style "
            "as the original Spyder script."
        )

if not st.session_state.databases:
    st.info(
        "Use **Connect / refresh databases** in the sidebar to start."
    )
    st.stop()

database = st.sidebar.selectbox(
    "Database",
    st.session_state.databases,
)

if st.sidebar.button("Load vial list", use_container_width=True):
    try:
        with st.spinner("Reading vial IDs..."):
            st.session_state.available_vials = get_vial_ids(
                SQL_SERVER,
                database,
            )
        st.sidebar.success(
            f"{len(st.session_state.available_vials)} vial IDs found."
        )
    except Exception as e:
        st.sidebar.error(f"Could not load vial IDs: {e}")


# ============================================================
# VIAL SELECTION
# ============================================================

st.sidebar.header("2. Vials")

available_vials = st.session_state.available_vials

selection_mode = st.sidebar.radio(
    "Selection mode",
    ["Range", "Individual vials"],
    horizontal=True,
)

selected_vials = []

if selection_mode == "Range":
    if available_vials:
        default_start = int(min(available_vials))
        default_end = int(min(default_start + 29, max(available_vials)))
    else:
        default_start = 1
        default_end = 30

    c1, c2 = st.sidebar.columns(2)
    start_vial = c1.number_input(
        "First vial",
        value=default_start,
        step=1,
    )
    end_vial = c2.number_input(
        "Last vial",
        value=default_end,
        step=1,
    )

    if end_vial >= start_vial:
        selected_vials = list(
            range(int(start_vial), int(end_vial) + 1)
        )

        if available_vials:
            available_set = set(available_vials)
            selected_vials = [
                v for v in selected_vials if v in available_set
            ]

    st.sidebar.caption(
        f"{len(selected_vials)} vial(s) selected"
    )

else:
    if not available_vials:
        st.sidebar.warning("Load the vial list first.")
        selected_vials = []
    else:
        selected_vials = st.sidebar.multiselect(
            "Choose vials",
            available_vials,
            default=available_vials[: min(10, len(available_vials))],
        )


# ============================================================
# DATA FILTER
# ============================================================

st.sidebar.header("3. Data loading")

sample_every = st.sidebar.slider(
    "Use every Nth row",
    min_value=1,
    max_value=50,
    value=5,
    help="Higher values make loading and plotting faster.",
)

with st.sidebar.expander("DeviceData filter"):
    device_filter_mode = st.radio(
        "Filter mode",
        ["Exclude", "Include only"],
    )

    default_excluded = (
        "1296,1298,1290,1292,1291,1294,1486,1229,1227,1224,"
        "1226,1228,1222,1285,1289,1293,1295,1223,1225,1204,"
        "1206,1209,1276,1211,1219,1201,1203,1205,1207,1213,"
        "1215,1221,1217,1274,1200,1278,1287,1283,1281,1258,"
        "1272,1282,1212,1280,1284,1220,1218,1214,12,85,1288,"
        "1210,1286,1208,1216,1202"
    )

    device_text = st.text_area(
        "DeviceData IDs",
        value=default_excluded if device_filter_mode == "Exclude" else "",
        height=100,
        help="Comma-separated IDs.",
    )

    device_ids = tuple(
        int(x)
        for x in re.findall(r"\d+", device_text)
    )

if st.sidebar.button("Load selected vial data", type="primary", use_container_width=True):
    if not selected_vials:
        st.sidebar.warning("Select at least one vial.")
    else:
        try:
            with st.spinner(
                f"Loading data for {len(selected_vials)} vial(s)..."
            ):
                st.session_state.raw_data = load_temperature_data(
                    server=SQL_SERVER,
                    database=database,
                    vial_ids_tuple=tuple(selected_vials),
                    sample_every=sample_every,
                    device_filter_mode=device_filter_mode,
                    device_ids_tuple=device_ids,
                )
            st.sidebar.success(
                f"Loaded {len(st.session_state.raw_data):,} rows."
            )
        except Exception as e:
            st.sidebar.error(f"Data loading failed: {e}")


# ============================================================
# MAIN OPTIONS
# ============================================================

raw_df = st.session_state.raw_data

if raw_df.empty:
    st.warning("No data loaded yet.")
    st.stop()

loaded_vials = sorted(raw_df["VialId"].unique())

m1, m2, m3 = st.columns(3)
m1.metric("Loaded vials", len(loaded_vials))
m2.metric("Data rows", f"{len(raw_df):,}")
m3.metric(
    "Time span",
    f"{raw_df['RelativeTimestamp'].max()/60:.1f} min"
)

tab_plot, tab_data, tab_raw = st.tabs(
    ["📈 Plot", "📋 Processed data", "🔎 Raw data"]
)

with tab_plot:

    col_options, col_plot = st.columns([0.30, 0.70])

    with col_options:
        st.subheader("Processing")

        smoothing_method = st.selectbox(
            "Smoothing",
            [
                "None",
                "Rolling mean",
                "Rolling median",
                "Robust rolling mean",
                "Savitzky-Golay",
            ],
            index=1,
        )

        smoothing_window = 3
        robust_factor = 2.0
        min_value = 20.0
        polyorder = 2

        if smoothing_method != "None":
            smoothing_window = st.slider(
                "Smoothing window",
                min_value=3,
                max_value=101,
                value=15 if smoothing_method == "Robust rolling mean" else 5,
                step=2,
            )

        if smoothing_method == "Robust rolling mean":
            robust_factor = st.slider(
                "MAD outlier factor",
                min_value=0.5,
                max_value=5.0,
                value=2.0,
                step=0.1,
            )
            min_value = st.number_input(
                "Ignore values below (K)",
                value=20.0,
            )

        if smoothing_method == "Savitzky-Golay":
            polyorder = st.slider(
                "Polynomial order",
                min_value=1,
                max_value=5,
                value=2,
            )
            if not SCIPY_AVAILABLE:
                st.warning(
                    "SciPy is not installed. Rolling mean will be used instead."
                )

        st.divider()

        alignment_mode = st.selectbox(
            "Alignment",
            [
                "Minimum temperature",
                "First below threshold",
                "No alignment",
                "Manual offset",
            ],
        )

        threshold = 250.0
        manual_offsets = {}

        if alignment_mode == "First below threshold":
            threshold = st.number_input(
                "Temperature threshold (K)",
                value=250.0,
                step=1.0,
            )

        if alignment_mode == "Manual offset":
            st.caption(
                "Enter the zero point in minutes after the first recorded point."
            )
            for vial_id in loaded_vials:
                manual_offsets[int(vial_id)] = st.number_input(
                    f"Vial {vial_id}",
                    value=0.0,
                    step=0.1,
                    key=f"offset_{vial_id}",
                )

        plot_time_offset_min = st.number_input(
            "Additional time shift (min)",
            value=0.0,
            step=0.1,
            help="Useful for matching another reference profile.",
        )

        st.divider()
        st.subheader("Display")

        show_raw = st.checkbox("Show raw data", value=False)
        show_mean = st.checkbox("Show mean profile", value=False)
        show_std = st.checkbox(
            "Show ±1 SD",
            value=False,
            disabled=not show_mean,
        )

        linewidth = st.slider(
            "Line width",
            0.5,
            4.0,
            1.2,
            0.1,
        )

        alpha = st.slider(
            "Line opacity",
            0.1,
            1.0,
            0.8,
            0.05,
        )

    processed_df = prepare_data(
        raw_df=raw_df,
        smoothing_method=smoothing_method,
        smoothing_window=smoothing_window,
        robust_factor=robust_factor,
        min_value=min_value,
        polyorder=polyorder,
        alignment_mode=alignment_mode,
        threshold=threshold,
        manual_offsets=manual_offsets,
        plot_time_offset_min=plot_time_offset_min,
    )

    with col_plot:
        st.subheader("Temperature profiles")

        with st.expander("Axes, setpoints and process regions", expanded=False):
            a1, a2 = st.columns(2)

            x_min = a1.number_input(
                "X minimum (min)",
                value=-15.0,
                step=1.0,
            )
            x_max = a2.number_input(
                "X maximum (min)",
                value=210.0,
                step=1.0,
            )

            y_min = a1.number_input(
                "Y minimum (K)",
                value=220.0,
                step=1.0,
            )
            y_max = a2.number_input(
                "Y maximum (K)",
                value=330.0,
                step=1.0,
            )

            use_pd = a1.checkbox("Show PD setpoint", value=True)
            pd_setpoint = a1.number_input(
                "PD setpoint (K)",
                value=236.0,
                step=1.0,
                disabled=not use_pd,
            )

            use_sd = a2.checkbox("Show SD setpoint", value=True)
            sd_setpoint = a2.number_input(
                "SD setpoint (K)",
                value=313.0,
                step=1.0,
                disabled=not use_sd,
            )

            freezing_start = a1.number_input(
                "Freezing region start (min)",
                value=-13.0,
                step=0.5,
            )
            freezing_end = a2.number_input(
                "Freezing region end (min)",
                value=-1.0,
                step=0.5,
            )

        fig = make_figure(
            processed_df=processed_df,
            show_raw=show_raw,
            show_mean=show_mean,
            show_std=show_std,
            pd_setpoint=pd_setpoint if use_pd else None,
            sd_setpoint=sd_setpoint if use_sd else None,
            freezing_start=freezing_start,
            freezing_end=freezing_end,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            linewidth=linewidth,
            alpha=alpha,
        )

        st.pyplot(fig, use_container_width=True)

        # Export PNG
        png_buffer = io.BytesIO()
        fig.savefig(
            png_buffer,
            format="png",
            dpi=300,
            bbox_inches="tight",
        )
        png_buffer.seek(0)

        # Export processed CSV
        export_columns = [
            "VialId",
            "Timestamp",
            "RelativeTimestamp",
            "AlignedTime_min",
            "Value",
            "SmoothedValue",
            "DeviceData_Id",
        ]

        csv_buffer = processed_df[
            [c for c in export_columns if c in processed_df.columns]
        ].to_csv(index=False).encode("utf-8")

        d1, d2 = st.columns(2)
        d1.download_button(
            "⬇️ Export figure (PNG)",
            data=png_buffer,
            file_name=f"{database}_vial_temperature_plot.png",
            mime="image/png",
            use_container_width=True,
        )

        d2.download_button(
            "⬇️ Export processed data (CSV)",
            data=csv_buffer,
            file_name=f"{database}_processed_vial_data.csv",
            mime="text/csv",
            use_container_width=True,
        )

with tab_data:
    st.dataframe(
        processed_df,
        use_container_width=True,
        hide_index=True,
    )

with tab_raw:
    st.dataframe(
        raw_df,
        use_container_width=True,
        hide_index=True,
    )
