import pandas as pd


def clean_gdelt_chunk(df, dtype_map, valid_fips):
    """Clean a chunk of GDELT records and create quality flags."""

    df = df.copy()

    # Standardize text columns
    for col in dtype_map:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace("", pd.NA)

    # Convert dates
    df["EventDate"] = pd.to_datetime(
        df["SQLDATE"].astype("string"),
        format="%Y%m%d",
        errors="coerce"
    )

    df["DateAdded"] = pd.to_datetime(
        df["DATEADDED"].astype("string"),
        format="%Y%m%d",
        errors="coerce"
    )

    # Calculate reporting delay
    df["ReportingDelayDays"] = (
        df["DateAdded"] - df["EventDate"]
    ).dt.days

    # Date quality flags
    df["Flag_LongDelay"] = df["ReportingDelayDays"] > 30
    df["Flag_FutureEvent"] = df["ReportingDelayDays"] < 0
    df["Flag_InvalidDate"] = (
        df["EventDate"].isna() | df["DateAdded"].isna()
    )

    # Missing actor information
    df["Flag_MissingActor1"] = df["Actor1Code"].isna()
    df["Flag_MissingActor2"] = df["Actor2Code"].isna()

    df["Flag_MissingActor1Country"] = (
        df["Actor1CountryCode"].isna()
    )
    df["Flag_MissingActor2Country"] = (
        df["Actor2CountryCode"].isna()
    )

    # Missing event geography
    df["Flag_MissingActionCountry"] = (
        df["ActionGeo_CountryCode"].isna()
    )
    df["Flag_MissingActionCoordinates"] = (
        df["ActionGeo_Lat"].isna()
        | df["ActionGeo_Long"].isna()
    )

    # Geographic validation
    for prefix in ["Actor1Geo", "Actor2Geo", "ActionGeo"]:
        lat = df[f"{prefix}_Lat"]
        lon = df[f"{prefix}_Long"]

        df[f"Flag_Invalid{prefix}Coordinates"] = (
            (lat.notna() & ~lat.between(-90, 90))
            | (lon.notna() & ~lon.between(-180, 180))
            | (lat.isna() ^ lon.isna())
        )

        country_col = f"{prefix}_CountryCode"

        df[f"Flag_Unrecognized{prefix}Country"] = (
            df[country_col].notna()
            & ~df[country_col].isin(valid_fips)
        )

    # Numerical validation
    numeric_checks = {
        "IsRootEvent": ~df["IsRootEvent"].isin([0, 1]),
        "QuadClass": ~df["QuadClass"].isin([1, 2, 3, 4]),
        "GoldsteinScale": ~df["GoldsteinScale"].between(-10, 10),
        "AvgTone": ~df["AvgTone"].between(-100, 100),
        "NumMentions": df["NumMentions"] < 0,
        "NumSources": df["NumSources"] < 0,
        "NumArticles": df["NumArticles"] < 0
    }

    for col, invalid in numeric_checks.items():
        df[f"Flag_Invalid_{col}"] = (
            invalid | df[col].isna()
        )

    return df


def validate_cameo(df, valid_cameo_event_codes):
    """Validate CAMEO classifications and their hierarchy."""

    df = df.copy()

    for col in ["EventCode", "EventBaseCode", "EventRootCode"]:
        codes = df[col].astype("string")

        df[f"Flag_Unrecognized_{col}"] = (
            codes.isna()
            | ~codes.isin(valid_cameo_event_codes)
        )

    df["Flag_EventBaseMismatch"] = pd.Series(
        [
            not event.startswith(base)
            if pd.notna(event) and pd.notna(base)
            else True
            for event, base in zip(
                df["EventCode"], df["EventBaseCode"]
            )
        ],
        index=df.index
    )

    df["Flag_EventRootMismatch"] = pd.Series(
        [
            not base.startswith(root)
            if pd.notna(base) and pd.notna(root)
            else True
            for base, root in zip(
                df["EventBaseCode"], df["EventRootCode"]
            )
        ],
        index=df.index
    )

    return df


def adjust_goldstein(df):
    """Preserve original Goldstein scores and impute code 1213."""

    df = df.copy()

    df["GoldsteinScale_Adjusted"] = df["GoldsteinScale"].copy()

    impute_mask = (
        df["EventCode"].eq("1213")
        & df["GoldsteinScale"].isna()
    ).fillna(False)

    # Parent code 121 has a Goldstein score of -4.0
    df.loc[impute_mask, "GoldsteinScale_Adjusted"] = -4.0

    df["Flag_GoldsteinImputed"] = impute_mask

    return df


def clean_gdelt(df, dtype_map, valid_fips, valid_cameo_event_codes):
    """Apply all GDELT cleaning and validation steps."""

    df = clean_gdelt_chunk(df, dtype_map, valid_fips)
    df = validate_cameo(df, valid_cameo_event_codes)
    df = adjust_goldstein(df)

    return df