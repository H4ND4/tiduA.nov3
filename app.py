from __future__ import annotations

from io import BytesIO
from pathlib import Path
import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE SETUP
# ============================================================
# Native Streamlit components only.
# This avoids the fragile custom CSS problems from the earlier version.
st.set_page_config(
    page_title="Audit Workbench",
    page_icon="📊",
    layout="wide",
)

PREVIEW_ROWS = 100


# ============================================================
# FILE LOADING
# ============================================================
@st.cache_data(show_spinner=False)
def get_excel_sheet_names(file_bytes: bytes) -> list[str]:
    """
    Read only the workbook structure and return worksheet names.

    Caching means Streamlit does not repeatedly reopen the workbook
    every time the user changes a widget.
    """
    excel_file = pd.ExcelFile(BytesIO(file_bytes), engine="openpyxl")
    return excel_file.sheet_names


@st.cache_data(show_spinner=False)
def load_file(
    file_bytes: bytes,
    filename: str,
    header_row: int = 1,
    sheet_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load the uploaded CSV or XLSX file into a pandas DataFrame.

    Important:
    - We do NOT sort the data.
    - We do NOT drop duplicates.
    - We do NOT alter the source DataFrame after loading.
    - header_row is user-facing (1 means the first row in the file).
    """
    suffix = Path(filename).suffix.lower()
    pandas_header = header_row - 1

    if suffix == ".csv":
        # low_memory=False gives pandas more consistent type inference
        # for larger CSV files containing mixed data types.
        return pd.read_csv(
            BytesIO(file_bytes),
            header=pandas_header,
            low_memory=False,
        )

    if suffix == ".xlsx":
        return pd.read_excel(
            BytesIO(file_bytes),
            sheet_name=sheet_name if sheet_name is not None else 0,
            header=pandas_header,
            engine="openpyxl",
        )

    raise ValueError("Unsupported file type. Please upload a CSV or XLSX file.")


# ============================================================
# BASIC ROW SAMPLING METHODS
# ============================================================
def sample_odd_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select user-facing row positions 1, 3, 5, 7, ...

    pandas uses zero-based positions:
        user row 1 -> pandas position 0
        user row 3 -> pandas position 2

    iloc[::2] therefore selects exactly the required rows.
    """
    return df.iloc[::2].copy()


def sample_top_n(df: pd.DataFrame, n: int) -> Tuple[pd.DataFrame, Optional[str]]:
    """Return the first N rows without changing their order."""
    if n <= 0:
        raise ValueError("N must be greater than 0.")

    if n > len(df):
        return (
            df.copy(),
            f"You requested {n:,} rows, but only {len(df):,} rows are available. "
            "All available rows were returned.",
        )

    return df.head(n).copy(), None


def sample_random(
    df: pd.DataFrame,
    n: int,
    seed: int,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Random sample WITHOUT replacement.

    random_state=seed makes the sample reproducible.
    sort_index() does not change which rows were selected; it simply
    shows selected rows in their original source order.
    """
    if n <= 0:
        raise ValueError("The requested sample size must be greater than 0.")

    warning = None
    effective_n = n

    if n > len(df):
        effective_n = len(df)
        warning = (
            f"You requested {n:,} rows, but only {len(df):,} rows are available. "
            "All available rows were returned."
        )

    result = df.sample(
        n=effective_n,
        replace=False,
        random_state=seed,
    ).sort_index()

    return result.copy(), warning


def sample_systematic(
    df: pd.DataFrame,
    sample_size: int,
    seed: int,
) -> Tuple[pd.DataFrame, dict, Optional[str]]:
    """
    Systematic random sampling.

    Population = number of rows in the uploaded dataset.
    Interval   = Population / Sample size.

    A reproducible random starting point is generated inside the first
    interval. Then we move forward by one interval for every sample item.

    When sample_size <= population, this returns exactly sample_size rows.
    """
    population = len(df)

    if sample_size <= 0:
        raise ValueError("Sample size must be greater than 0.")

    if population == 0:
        raise ValueError("The uploaded dataset contains no data rows.")

    warning = None

    if sample_size > population:
        warning = (
            f"You requested {sample_size:,} rows, but the population contains "
            f"only {population:,}. All available rows were returned."
        )
        metadata = {
            "population": population,
            "sample_size": population,
            "interval": 1.0,
            "random_start": 1.0,
        }
        return df.copy(), metadata, warning

    interval = population / sample_size

    # NumPy's random generator is fast and reproducible with the same seed.
    rng = np.random.default_rng(seed)

    # start_zero_based is a continuous position between 0 and one interval.
    start_zero_based = rng.uniform(0, interval)

    # Vectorized calculation - no row-by-row Python loop.
    positions = np.floor(
        start_zero_based + np.arange(sample_size) * interval
    ).astype(int)

    # Safety clip for floating-point edge cases.
    positions = np.clip(positions, 0, population - 1)

    result = df.iloc[positions].copy()

    metadata = {
        "population": population,
        "sample_size": len(result),
        "interval": interval,
        # Convert to a user-facing 1-based position for display.
        "random_start": start_zero_based + 1,
    }

    return result, metadata, warning


# ============================================================
# AUDIT-SPECIFIC HELPERS
# ============================================================
def numeric_series_from_columns(
    df: pd.DataFrame,
    mode: str,
    column_1: str,
    column_2: Optional[str] = None,
) -> pd.Series:
    """
    Create a temporary numeric amount series for testing.

    The original DataFrame is NOT modified.

    mode:
    - "One amount column"
    - "Add two columns"
    """
    first = pd.to_numeric(df[column_1], errors="coerce")

    if mode == "One amount column":
        return first

    if column_2 is None:
        raise ValueError("Please select the second amount column.")

    second = pd.to_numeric(df[column_2], errors="coerce")

    # This matches the example journal workbook's Debit + Credit logic.
    return first.fillna(0) + second.fillna(0)


def benford_filter(
    df: pd.DataFrame,
    amounts: pd.Series,
    threshold_percentage_points: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, list[int], int]:
    """
    Perform a first-digit Benford test.

    The example audit workbook calculates:
        first digit = LEFT(amount, 1)
        expected    = LOG10(1 + 1/digit)
        actual      = count / total

    Its colored rows correspond to over-represented digits.
    A default threshold of 0.25 percentage points reproduces the
    pattern seen in the provided example workbook.

    We calculate the first significant digit without changing the source data.
    """
    numeric = pd.to_numeric(amounts, errors="coerce")

    # The provided Excel example uses LEFT(amount, 1).
    # Therefore values from 0 to <1 start with "0" and are NOT part of
    # the 1-9 Benford distribution. Negative values start with "-" and
    # are also excluded. Using >= 1 reproduces the example workbook.
    valid_mask = numeric.notna() & (numeric >= 1)
    valid = numeric.loc[valid_mask]

    if valid.empty:
        raise ValueError("No numeric amounts of 1 or greater were available for Benford testing.")

    # Vectorized first-significant-digit calculation.
    magnitude = np.floor(np.log10(valid))
    first_digits_valid = np.floor(valid / np.power(10.0, magnitude)).astype(int)
    first_digits_valid = first_digits_valid.clip(1, 9)

    # Keep the first digit aligned with the original DataFrame index.
    first_digits = pd.Series(pd.NA, index=df.index, dtype="Int64")
    first_digits.loc[valid.index] = first_digits_valid.astype("int64")

    digits = np.arange(1, 10)
    counts = first_digits_valid.value_counts().reindex(digits, fill_value=0).sort_index()
    actual = counts / len(first_digits_valid)
    expected = np.log10(1 + (1 / digits))
    difference_pp = (actual.to_numpy() - expected) * 100

    # Select only digits whose ACTUAL frequency is above expected by
    # more than the chosen threshold.
    selected_mask = difference_pp > threshold_percentage_points
    flagged_digits = digits[selected_mask].tolist()

    summary = pd.DataFrame(
        {
            "Digit": digits,
            "Count": counts.to_numpy(),
            "Expected %": expected * 100,
            "Actual %": actual.to_numpy() * 100,
            "Actual - Expected (pp)": difference_pp,
            "Flagged": ["Yes" if d in flagged_digits else "No" for d in digits],
        }
    )

    result = df.loc[first_digits.isin(flagged_digits)].copy()

    return result, summary, flagged_digits, len(valid)


def weekend_filter(df: pd.DataFrame, date_column: str) -> Tuple[pd.DataFrame, int]:
    """
    Select Saturday and Sunday transactions.

    A temporary parsed date series is created.
    The original column is left untouched.
    """
    parsed = pd.to_datetime(df[date_column], errors="coerce")
    valid_dates = int(parsed.notna().sum())

    mask = parsed.dt.dayofweek.isin([5, 6])
    return df.loc[mask].copy(), valid_dates


def public_holiday_filter(
    df: pd.DataFrame,
    date_column: str,
    holiday_text: str,
) -> Tuple[pd.DataFrame, list[str], int]:
    """
    Select transactions whose date matches one of the user-provided holidays.

    Holiday input format:
        YYYY-MM-DD
        one date per line
    """
    raw_lines = [line.strip() for line in holiday_text.splitlines() if line.strip()]

    if not raw_lines:
        raise ValueError("Please enter at least one public holiday date.")

    parsed_holidays = pd.to_datetime(raw_lines, errors="coerce")
    invalid_lines = [
        raw_lines[i]
        for i, value in enumerate(parsed_holidays)
        if pd.isna(value)
    ]

    holidays = pd.DatetimeIndex(parsed_holidays.dropna()).normalize().unique()

    if len(holidays) == 0:
        raise ValueError("None of the holiday dates could be read.")

    parsed_dates = pd.to_datetime(df[date_column], errors="coerce")
    valid_dates = int(parsed_dates.notna().sum())

    mask = parsed_dates.dt.normalize().isin(holidays)
    return df.loc[mask].copy(), invalid_lines, valid_dates


def round_million_filter(
    df: pd.DataFrame,
    amounts: pd.Series,
) -> Tuple[pd.DataFrame, int]:
    """
    Match the example workbook's '000000 ending' test.

    Examples flagged:
        1,000,000
        25,000,000
        -3,000,000

    Examples not flagged:
        1,500,000
        1,000,000.50
        0

    The source data is not changed.
    """
    numeric = pd.to_numeric(amounts, errors="coerce")
    valid_numeric = int(numeric.notna().sum())

    # np.isclose protects against tiny floating-point representation errors.
    remainder = np.mod(numeric.abs(), 1_000_000)
    mask = (
        numeric.notna()
        & (numeric != 0)
        & np.isclose(remainder, 0, atol=1e-6)
    )

    return df.loc[mask].copy(), valid_numeric


# ============================================================
# DOWNLOAD HELPERS
# ============================================================
def sanitize_filename(filename: str) -> str:
    """Make a safe base filename for downloads."""
    stem = Path(filename).stem
    stem = re.sub(r"[^\w.-]+", "_", stem, flags=re.UNICODE).strip("._")
    return stem or "sample"


@st.cache_data(show_spinner=False)
def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Create a UTF-8 CSV in memory."""
    return df.to_csv(index=False).encode("utf-8-sig")


@st.cache_data(show_spinner=False)
def dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    """Create an XLSX file in memory without writing a temporary file."""
    buffer = BytesIO()

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Selected Data")

    return buffer.getvalue()


def preview_with_original_positions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a DISPLAY-ONLY preview.

    The left-side index is the original position in the loaded dataset:
        first data row = position 1

    This helper does NOT add a column to the downloadable result.
    """
    preview = df.head(PREVIEW_ROWS).copy()

    # The loaded source DataFrame has a normal 0..N-1 RangeIndex.
    # All our filters/samplers keep those original index values.
    preview.index = preview.index + 1
    preview.index.name = "Original row position"

    return preview


def result_slug(
    method: str,
    selected_count: int,
    seed: Optional[int] = None,
) -> str:
    """Build a readable filename suffix."""
    if method == "Odd-numbered rows":
        return "odd_rows"
    if method == "Top N rows":
        return f"top_{selected_count}"
    if method == "Random rows":
        return f"random_{selected_count}_seed{seed}"
    if method == "Systematic random":
        return f"systematic_{selected_count}_seed{seed}"
    if method == "Benford test":
        return "benford_flagged"
    if method == "Public holiday transactions":
        return "public_holidays"
    if method == "Weekend transactions":
        return "weekends"
    if method == "000'000 ending transactions":
        return "round_million"
    return "selected"


# ============================================================
# UI HELPERS
# ============================================================
def amount_controls(
    df: pd.DataFrame,
    key_prefix: str,
) -> Tuple[str, str, Optional[str]]:
    """
    Render simple controls for choosing the transaction amount.

    Some audit files already have one amount column.
    Others, like the provided journal example, use Debit + Credit.
    """
    mode = st.radio(
        "Amount source",
        options=["One amount column", "Add two columns"],
        horizontal=True,
        key=f"{key_prefix}_amount_mode",
        help=(
            "Choose one amount column, or add two columns such as Debit + Credit. "
            "This creates a temporary calculation only; your uploaded data is not changed."
        ),
    )

    columns = list(df.columns)

    column_1 = st.selectbox(
        "Amount column" if mode == "One amount column" else "First amount column",
        options=columns,
        key=f"{key_prefix}_amount_1",
    )

    column_2 = None

    if mode == "Add two columns":
        default_second = 1 if len(columns) > 1 else 0
        column_2 = st.selectbox(
            "Second amount column",
            options=columns,
            index=default_second,
            key=f"{key_prefix}_amount_2",
        )

    return mode, column_1, column_2


# ============================================================
# MAIN APP
# ============================================================
def main() -> None:
    st.title("Audit Workbench")
    st.caption("Simple local spreadsheet testing and sampling")

    # --------------------------------------------------------
    # 1. Upload
    # --------------------------------------------------------
    with st.container(border=True):
        st.subheader("1 · Upload data")

        uploaded_file = st.file_uploader(
            "Upload a CSV or Excel file",
            type=["csv", "xlsx"],
            help="The file is processed in this Streamlit session. CSV and XLSX are supported.",
        )

        if uploaded_file is None:
            st.info("Upload a CSV or XLSX file to begin.")
            return

        file_bytes = uploaded_file.getvalue()
        suffix = Path(uploaded_file.name).suffix.lower()

        # File options stay compact and only appear after upload.
        option_cols = st.columns([2, 1])

        sheet_name = None

        with option_cols[0]:
            if suffix == ".xlsx":
                try:
                    sheets = get_excel_sheet_names(file_bytes)
                except Exception as exc:
                    st.error(f"Could not open this Excel workbook: {exc}")
                    return

                sheet_name = st.selectbox(
                    "Worksheet",
                    options=sheets,
                    index=0,
                    help="The first worksheet is selected by default.",
                )
            else:
                st.caption("CSV file")

        with option_cols[1]:
            header_row = int(
                st.number_input(
                    "Header row",
                    value=1,
                    step=1,
                    min_value=1,
                    help=(
                        "Row containing the column names. "
                        "Use 1 for normal files. "
                        "Your example journal workbook uses row 3."
                    ),
                )
            )

        try:
            with st.spinner("Reading file..."):
                df = load_file(
                    file_bytes=file_bytes,
                    filename=uploaded_file.name,
                    header_row=header_row,
                    sheet_name=sheet_name,
                )
        except Exception as exc:
            st.error(f"Could not read the uploaded file: {exc}")
            return

        if len(df.columns) == 0:
            st.error("The uploaded file does not contain readable columns.")
            return

        # Basic information required by the assignment.
        m1, m2, m3 = st.columns(3)
        m1.metric("Rows", f"{len(df):,}")
        m2.metric("Columns", f"{len(df.columns):,}")
        m3.metric("File", uploaded_file.name)

        with st.expander("Column names"):
            st.write(list(df.columns))

        with st.expander("Preview source data"):
            st.caption(
                "The first data row is shown as Original row position 1. "
                "This is the position inside the loaded dataset, not necessarily "
                "the physical Excel worksheet row number."
            )
            st.dataframe(
                preview_with_original_positions(df),
                use_container_width=True,
            )

    if df.empty:
        st.warning("The file contains column headers but no data rows.")
        return

    # --------------------------------------------------------
    # 2. Choose ONE method
    # --------------------------------------------------------
    with st.container(border=True):
        st.subheader("2 · Choose one method")

        method = st.selectbox(
            "Sampling / filtering method",
            options=[
                "Odd-numbered rows",
                "Top N rows",
                "Random rows",
                "Systematic random",
                "Benford test",
                "Public holiday transactions",
                "Weekend transactions",
                "000'000 ending transactions",
            ],
        )

        st.caption(
            "The original DataFrame is never sorted or modified. "
            "Each method creates a separate result."
        )

    # --------------------------------------------------------
    # 3. Configure + run selected method
    # --------------------------------------------------------
    result: Optional[pd.DataFrame] = None
    warning_message: Optional[str] = None
    extra_info: dict = {}
    seed_for_filename: Optional[int] = None

    with st.container(border=True):
        st.subheader("3 · Configure")

        try:
            if method == "Odd-numbered rows":
                st.write(
                    "Selects data positions **1, 3, 5, 7, ...**. "
                    "This is based on row position, not on values in a column."
                )
                result = sample_odd_rows(df)

            elif method == "Top N rows":
                n = int(
                    st.number_input(
                        "Number of rows (N)",
                        value=10,
                        step=1,
                    )
                )

                if n <= 0:
                    st.error("N must be greater than 0.")
                else:
                    result, warning_message = sample_top_n(df, n)

            elif method == "Random rows":
                c1, c2 = st.columns(2)

                with c1:
                    n = int(
                        st.number_input(
                            "Number of rows",
                            value=10,
                            step=1,
                            key="random_n",
                        )
                    )

                with c2:
                    seed = int(
                        st.number_input(
                            "Random seed",
                            value=42,
                            step=1,
                            key="random_seed",
                            help=(
                                "Using the same seed and the same dataset gives "
                                "the same random sample again."
                            ),
                        )
                    )

                st.caption(
                    "Sampling is without replacement, so the same source row "
                    "cannot be selected twice."
                )

                if n <= 0:
                    st.error("The number of rows must be greater than 0.")
                else:
                    result, warning_message = sample_random(df, n, seed)
                    seed_for_filename = seed

            elif method == "Systematic random":
                c1, c2 = st.columns(2)

                with c1:
                    sample_size = int(
                        st.number_input(
                            "Sample size",
                            value=5,
                            step=1,
                            key="systematic_n",
                        )
                    )

                with c2:
                    seed = int(
                        st.number_input(
                            "Random seed",
                            value=42,
                            step=1,
                            key="systematic_seed",
                            help=(
                                "The seed makes the random starting point reproducible."
                            ),
                        )
                    )

                if sample_size <= 0:
                    st.error("Sample size must be greater than 0.")
                else:
                    result, systematic_info, warning_message = sample_systematic(
                        df,
                        sample_size,
                        seed,
                    )
                    extra_info.update(systematic_info)
                    seed_for_filename = seed

            elif method == "Benford test":
                mode, column_1, column_2 = amount_controls(df, "benford")

                threshold_pp = float(
                    st.number_input(
                        "Over-representation threshold (percentage points)",
                        min_value=0.0,
                        value=0.25,
                        step=0.05,
                        format="%.2f",
                        help=(
                            "A digit is flagged when its actual frequency is above "
                            "Benford's expected frequency by more than this amount. "
                            "0.25 matches the pattern in your example workbook."
                        ),
                    )
                )

                amounts = numeric_series_from_columns(
                    df,
                    mode,
                    column_1,
                    column_2,
                )

                result, benford_summary, flagged_digits, valid_amount_count = benford_filter(
                    df,
                    amounts,
                    threshold_pp,
                )

                extra_info["benford_summary"] = benford_summary
                extra_info["flagged_digits"] = flagged_digits
                extra_info["valid_amount_count"] = valid_amount_count

            elif method == "Public holiday transactions":
                date_column = st.selectbox(
                    "Date column",
                    options=list(df.columns),
                    key="holiday_date_column",
                )

                # These are the holiday dates listed in the example workbook.
                # They are editable because another audit year will have different dates.
                default_holidays = """2025-01-01
2025-01-29
2025-01-30
2025-01-31
2025-03-08
2025-06-01
2025-07-11
2025-07-12
2025-07-13
2025-07-14
2025-07-15
2025-11-21
2025-11-26
2025-12-29"""

                holiday_text = st.text_area(
                    "Public holiday dates — one per line (YYYY-MM-DD)",
                    value=default_holidays,
                    height=180,
                    help=(
                        "The defaults are the dates contained in your example workbook. "
                        "Replace them when testing a different year."
                    ),
                )

                result, invalid_holidays, valid_date_count = public_holiday_filter(
                    df,
                    date_column,
                    holiday_text,
                )

                extra_info["invalid_holidays"] = invalid_holidays
                extra_info["valid_date_count"] = valid_date_count

            elif method == "Weekend transactions":
                date_column = st.selectbox(
                    "Date column",
                    options=list(df.columns),
                    key="weekend_date_column",
                )

                result, valid_date_count = weekend_filter(df, date_column)
                extra_info["valid_date_count"] = valid_date_count

            elif method == "000'000 ending transactions":
                mode, column_1, column_2 = amount_controls(df, "round")

                amounts = numeric_series_from_columns(
                    df,
                    mode,
                    column_1,
                    column_2,
                )

                result, valid_numeric_count = round_million_filter(df, amounts)
                extra_info["valid_numeric_count"] = valid_numeric_count

        except ValueError as exc:
            st.error(str(exc))
            result = None
        except Exception as exc:
            st.error(f"The selected method could not be completed: {exc}")
            result = None

    # Stop here if validation failed.
    if result is None:
        return

    if warning_message:
        st.warning(warning_message)

    # --------------------------------------------------------
    # 4. Results
    # --------------------------------------------------------
    with st.container(border=True):
        st.subheader("4 · Results")

        r1, r2, r3 = st.columns(3)
        r1.metric("Method", method)
        r2.metric("Rows in original file", f"{len(df):,}")
        r3.metric("Rows selected", f"{len(result):,}")

        if method == "Systematic random":
            st.write(
                f"**Population:** {extra_info['population']:,}  \n"
                f"**Sample size:** {extra_info['sample_size']:,}  \n"
                f"**Interval:** {extra_info['interval']:.4f}  \n"
                f"**Random start (1-based):** {extra_info['random_start']:.4f}"
            )

        if method == "Benford test":
            flagged = extra_info["flagged_digits"]

            st.write(
                "**Flagged first digits:** "
                + (", ".join(map(str, flagged)) if flagged else "None")
            )
            st.caption(
                f"Numeric amounts >= 1 included in the Benford calculation: "
                f"{extra_info['valid_amount_count']:,}"
            )

            st.dataframe(
                extra_info["benford_summary"].style.format(
                    {
                        "Expected %": "{:.2f}%",
                        "Actual %": "{:.2f}%",
                        "Actual - Expected (pp)": "{:.2f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

        if method == "Public holiday transactions":
            st.caption(
                f"Readable dates in selected date column: "
                f"{extra_info['valid_date_count']:,}"
            )

            invalid = extra_info["invalid_holidays"]
            if invalid:
                st.warning(
                    "These holiday entries could not be read and were ignored: "
                    + ", ".join(invalid)
                )

        if method == "Weekend transactions":
            st.caption(
                f"Readable dates in selected date column: "
                f"{extra_info['valid_date_count']:,}"
            )

        if method == "000'000 ending transactions":
            st.caption(
                f"Readable numeric amounts used for the test: "
                f"{extra_info['valid_numeric_count']:,}"
            )

        if len(result) == 0:
            st.info("No rows matched the selected method.")
        else:
            st.caption(
                f"Previewing up to {PREVIEW_ROWS} selected rows. "
                "The download contains the complete result."
            )

            st.dataframe(
                preview_with_original_positions(result),
                use_container_width=True,
            )

        st.caption(
            "The preview's index is the row's original position in the loaded dataset "
            "(first data row = 1). This helper index is NOT added to the downloaded files."
        )

    # --------------------------------------------------------
    # 5. Download
    # --------------------------------------------------------
    with st.container(border=True):
        st.subheader("5 · Download")

        base_name = sanitize_filename(uploaded_file.name)
        slug = result_slug(
            method,
            selected_count=len(result),
            seed=seed_for_filename,
        )

        csv_name = f"{base_name}_{slug}.csv"
        excel_name = f"{base_name}_{slug}.xlsx"

        d1, d2 = st.columns(2)

        with d1:
            st.download_button(
                "Download CSV",
                data=dataframe_to_csv_bytes(result),
                file_name=csv_name,
                mime="text/csv",
                use_container_width=True,
            )

        with d2:
            st.download_button(
                "Download Excel",
                data=dataframe_to_excel_bytes(result),
                file_name=excel_name,
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                ),
                use_container_width=True,
            )

        st.caption(
            "Downloads contain only the selected rows and the original data columns. "
            "Pandas exports data values; original Excel formatting, colors, and formulas "
            "are not reproduced in the new output workbook."
        )


if __name__ == "__main__":
    main()