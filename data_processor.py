
import pandas as pd
import re


# --------------------------------------------------
# 1. LOAD CSV
# --------------------------------------------------

def load_csv(file):

    try:
        df = pd.read_csv(file)

    except Exception as e:
        raise ValueError(f"Unable to read CSV: {e}")

    if df.empty:
        raise ValueError("Dataset is empty.")

    if len(df.columns) < 2:
        raise ValueError(
            "Dataset must contain at least two columns."
        )

    # Clean column names
    df.columns = (
        df.columns.astype(str)
        .str.strip()
    )

    # Check duplicate column names
    if df.columns.duplicated().any():
        raise ValueError(
            "Duplicate column names found."
        )

    return df


# --------------------------------------------------
# 2. DETECT DATA TYPE
# --------------------------------------------------

def detect_type(series):

    values = series.dropna()

    if values.empty:
        return "Empty"

    if pd.api.types.is_numeric_dtype(series):
        return "Numeric"

    # Detect date columns
    dates = pd.to_datetime(
        values,
        errors="coerce"
    )

    if dates.notna().mean() > 0.85:
        return "Date"

    return "Categorical"


# --------------------------------------------------
# 3. CLASSIFY COLUMN ROLE
# --------------------------------------------------

def classify_column(col, series, dtype, row_count):

    name = str(col).strip().lower()

    # Normalize column name for matching
    normalized_name = re.sub(
        r"[^a-z0-9]+",
        "_",
        name
    ).strip("_")

    unique = int(series.nunique())

    # ----------------------------------------------
    # IDENTIFIER AND CONTACT FIELDS
    # ----------------------------------------------

    identifier_names = {
        "id",
        "hcp_id",
        "npi",
        "name",
        "hcp_name",
        "doctor_name",
        "physician_name",
        "provider_name",
        "phone",
        "phone_number",
        "mobile",
        "mobile_number",
        "email",
        "email_address",
        "address",
        "contact",
        "contact_number",
        "fax",
        "fax_number"
    }

    identifier_pattern = re.compile(
        r"(^|_)(id|npi|name|phone|mobile|email|"
        r"address|contact|fax)(_|$)",
        re.I
    )

    # ----------------------------------------------
    # EXISTING BUSINESS LABELS
    # ----------------------------------------------

    label_pattern = re.compile(
        r"(^|_)(tier|segment|category|class|label|"
        r"target|outcome|classification)(_|$)",
        re.I
    )

    # ----------------------------------------------
    # REFERENCE FIELD CLASSIFICATION
    # ----------------------------------------------

    if (
        normalized_name in identifier_names
        or identifier_pattern.search(normalized_name)
    ):

        return {
            "role": "Reference Field",

            "reason": (
                "Identifier or contact field; retained "
                "for identification and reporting, "
                "not used for clustering."
            ),

            "selectable": False
        }

    # ----------------------------------------------
    # EXISTING BUSINESS LABEL CLASSIFICATION
    # ----------------------------------------------

    if label_pattern.search(normalized_name):

        return {
            "role": "Excluded Business Label",

            "reason": (
                "Existing business classification; "
                "excluded by default to avoid influencing "
                "newly generated segments."
            ),

            "selectable": False
        }

    # ----------------------------------------------
    # NUMERIC FEATURE
    # ----------------------------------------------

    if dtype == "Numeric" and unique > 1:

        return {
            "role": "Potential Feature",

            "reason": (
                "Numeric variable that may help "
                "distinguish HCP behaviour."
            ),

            "selectable": True
        }

    # ----------------------------------------------
    # CATEGORICAL FEATURE
    # ----------------------------------------------

    if (
        dtype == "Categorical"
        and 1 < unique <= max(
            20,
            int(row_count * 0.5)
        )
    ):

        return {
            "role": "Potential Feature",

            "reason": (
                "Categorical variable that may help "
                "identify similar HCP groups."
            ),

            "selectable": True
        }

    # ----------------------------------------------
    # FIELDS REQUIRING MANUAL REVIEW
    # ----------------------------------------------

    return {
        "role": "Review",

        "reason": (
            "Not automatically included. "
            "Review data quality and business relevance."
        ),

        "selectable": False
    }


# --------------------------------------------------
# 4. PROFILE DATASET
# --------------------------------------------------

def profile_data(df):

    columns = []

    features = []

    reference_fields = []

    excluded_labels = []

    review_fields = []

    # ----------------------------------------------
    # PROCESS EVERY COLUMN
    # ----------------------------------------------

    for col in df.columns:

        series = df[col]

        dtype = detect_type(series)

        missing = int(series.isna().sum())

        unique = int(series.nunique())

        classification = classify_column(
            col,
            series,
            dtype,
            len(df)
        )

        role = classification["role"]

        # ------------------------------------------
        # COLUMN INFORMATION
        # ------------------------------------------

        column_info = {

            "column": col,

            "type": dtype,

            "missing": missing,

            "unique": unique,

            "role": role,

            "reason": classification["reason"]

        }

        columns.append(column_info)

        # ------------------------------------------
        # GROUP COLUMNS BY ROLE
        # ------------------------------------------

        if classification["selectable"]:

            features.append({

                "column": col,

                "type": dtype

            })

        elif role == "Reference Field":

            reference_fields.append(col)

        elif role == "Excluded Business Label":

            excluded_labels.append(col)

        elif role == "Review":

            review_fields.append(col)


    # ----------------------------------------------
    # DATA QUALITY STATISTICS
    # ----------------------------------------------

    missing_cells = int(
        df.isna().sum().sum()
    )

    duplicate_rows = int(
        df.duplicated().sum()
    )


    # ----------------------------------------------
    # EXCLUSION COUNTS
    # ----------------------------------------------

    reference_count = len(reference_fields)

    excluded_label_count = len(excluded_labels)

    excluded_count = (
        reference_count + excluded_label_count
    )


    # ----------------------------------------------
    # FINAL PROFILE RESPONSE
    # ----------------------------------------------

    return {

        "rows": len(df),

        "total_columns": len(df.columns),

        "missing_cells": missing_cells,

        "duplicate_rows": duplicate_rows,

        "columns": columns,

        "features": features,

        "reference_fields": reference_fields,

        "excluded_labels": excluded_labels,

        "review_fields": review_fields,

        "excluded_count": excluded_count,

        "reference_count": reference_count,

        "excluded_label_count": excluded_label_count

    }