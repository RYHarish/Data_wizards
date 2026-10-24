"""GDELT daily geopolitical feature engineering."""

from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

OIL_EXPORTERS = {'SAU': 'Saudi Arabia',
 'RUS': 'Russia',
 'KWT': 'Kuwait',
 'IRN': 'Iran',
 'IRQ': 'Iraq',
 'NGA': 'Nigeria',
 'ARE': 'United Arab Emirates',
 'AGO': 'Angola',
 'VEN': 'Venezuela',
 'NOR': 'Norway'}

OIL_IMPORTERS = {'USA': 'United States',
 'CHN': 'China',
 'IND': 'India',
 'JPN': 'Japan',
 'KOR': 'South Korea',
 'DEU': 'Germany',
 'ITA': 'Italy',
 'FRA': 'France',
 'NLD': 'Netherlands',
 'SGP': 'Singapore'}

CAMEO_ROOTS = {'01': 'Make public statement',
 '02': 'Appeal',
 '03': 'Express intent to cooperate',
 '04': 'Consult',
 '05': 'Engage in diplomatic cooperation',
 '06': 'Engage in material cooperation',
 '07': 'Provide aid',
 '08': 'Yield',
 '09': 'Investigate',
 '10': 'Demand',
 '11': 'Disapprove',
 '12': 'Reject',
 '13': 'Threaten',
 '14': 'Protest',
 '15': 'Exhibit military posture',
 '16': 'Reduce relations',
 '17': 'Coerce',
 '18': 'Assault',
 '19': 'Fight',
 '20': 'Use unconventional mass violence'}

FEATURE_GROUPS = {'general': ['date',
             'global_total_events',
             'exporter_total_events',
             'importer_total_events',
             'oil_country_event_share'],
 'global_cameo': ['global_root_01',
                  'global_root_02',
                  'global_root_03',
                  'global_root_04',
                  'global_root_05',
                  'global_root_06',
                  'global_root_07',
                  'global_root_08',
                  'global_root_09',
                  'global_root_10',
                  'global_root_11',
                  'global_root_12',
                  'global_root_13',
                  'global_root_14',
                  'global_root_15',
                  'global_root_16',
                  'global_root_17',
                  'global_root_18',
                  'global_root_19',
                  'global_root_20'],
 'oil_country_cameo': ['exporter_root_01',
                       'exporter_root_04',
                       'exporter_root_05',
                       'exporter_root_06',
                       'exporter_root_10',
                       'exporter_root_11',
                       'exporter_root_13',
                       'exporter_root_14',
                       'exporter_root_15',
                       'exporter_root_16',
                       'exporter_root_17',
                       'exporter_root_19',
                       'importer_root_04',
                       'importer_root_05',
                       'importer_root_06',
                       'importer_root_11',
                       'importer_root_13',
                       'importer_root_14',
                       'importer_root_17',
                       'importer_root_19'],
 'directed_relationships': ['exp_to_exp_events',
                            'exp_to_exp_goldstein_mean',
                            'exp_to_exp_tone_mean',
                            'exp_to_imp_events',
                            'exp_to_imp_goldstein_mean',
                            'exp_to_imp_tone_mean',
                            'imp_to_exp_events',
                            'imp_to_exp_goldstein_mean',
                            'imp_to_exp_tone_mean',
                            'imp_to_imp_events',
                            'imp_to_imp_goldstein_mean',
                            'imp_to_imp_tone_mean'],
 'goldstein': ['global_goldstein_mean',
               'exporter_goldstein_mean',
               'importer_goldstein_mean',
               'global_goldstein_std',
               'exporter_goldstein_impact_index',
               'importer_goldstein_impact_index'],
 'tone': ['global_tone_mean',
          'exporter_tone_mean',
          'importer_tone_mean',
          'global_negative_tone_share'],
 'media_attention': ['global_total_mentions',
                     'global_total_articles',
                     'exporter_total_mentions',
                     'exporter_total_articles',
                     'global_mentions_per_event'],
 'internal_and_quality': ['exporter_internal_event_share',
                          'importer_internal_event_share',
                          'actor1_country_missing_share',
                          'actor2_country_missing_share'],
 'major_countries': ['iran_total_events',
                     'saudi_total_events',
                     'russia_total_events',
                     'iraq_total_events',
                     'usa_total_events',
                     'china_total_events',
                     'iran_goldstein_mean',
                     'russia_goldstein_mean']}

MAJOR_COUNTRIES = {'iran': 'IRN',
 'saudi': 'SAU',
 'russia': 'RUS',
 'iraq': 'IRQ',
 'usa': 'USA',
 'china': 'CHN'}

REQUIRED_COLUMNS = ['GLOBALEVENTID',
 'DATEADDED',
 'Actor1CountryCode',
 'Actor2CountryCode',
 'EventRootCode',
 'GoldsteinScale_Adjusted',
 'AvgTone',
 'NumMentions',
 'NumArticles']


EXPORTER_CODES = set(OIL_EXPORTERS)
IMPORTER_CODES = set(OIL_IMPORTERS)
OIL_COUNTRIES = EXPORTER_CODES | IMPORTER_CODES

FEATURE_COLUMNS = [
    name
    for features in FEATURE_GROUPS.values()
    for name in features
]

RELATIONSHIPS = [
    "exp_to_exp",
    "exp_to_imp",
    "imp_to_exp",
    "imp_to_imp",
]


def aggregate_feature_batch(df):
    """Return additive daily statistics for one GDELT batch."""

    dates = pd.to_datetime(
        df["DATEADDED"].astype(str), format="%Y%m%d"
    )

    a1 = df["Actor1CountryCode"]
    a2 = df["Actor2CountryCode"]
    root = df["EventRootCode"]

    gold = df["GoldsteinScale_Adjusted"].astype("float64")
    tone = df["AvgTone"].astype("float64")
    mentions = df["NumMentions"].astype("float64")
    articles = df["NumArticles"].astype("float64")

    exporter = (
        a1.isin(EXPORTER_CODES) | a2.isin(EXPORTER_CODES)
    ).fillna(False)

    importer = (
        a1.isin(IMPORTER_CODES) | a2.isin(IMPORTER_CODES)
    ).fillna(False)

    oil_country = exporter | importer

    internal = (
        a1.notna() & a2.notna() & a1.eq(a2)
    ).fillna(False)

    # Intermediate statistics are additive across batches
    data = {
        "global_total_events": np.ones(len(df), dtype="int64"),
        "exporter_total_events": exporter.astype("int64"),
        "importer_total_events": importer.astype("int64"),
        "_oil_country_events": oil_country.astype("int64"),
        "_global_gold_sum": gold.fillna(0),
        "_global_gold_count": gold.notna().astype("int64"),
        "_global_gold_sq_sum": gold.pow(2).fillna(0),
        "_exporter_gold_sum": gold.where(exporter, 0).fillna(0),
        "_exporter_gold_count": (exporter & gold.notna()).astype("int64"),
        "_importer_gold_sum": gold.where(importer, 0).fillna(0),
        "_importer_gold_count": (importer & gold.notna()).astype("int64"),
        "_global_tone_sum": tone.fillna(0),
        "_global_tone_count": tone.notna().astype("int64"),
        "_exporter_tone_sum": tone.where(exporter, 0).fillna(0),
        "_exporter_tone_count": (exporter & tone.notna()).astype("int64"),
        "_importer_tone_sum": tone.where(importer, 0).fillna(0),
        "_importer_tone_count": (importer & tone.notna()).astype("int64"),
        "_global_negative_tone_count": tone.lt(0).fillna(False).astype("int64"),
        "global_total_mentions": mentions,
        "global_total_articles": articles,
        "exporter_total_mentions": mentions.where(exporter, 0),
        "exporter_total_articles": articles.where(exporter, 0),
        "_exporter_internal_count": (exporter & internal).astype("int64"),
        "_importer_internal_count": (importer & internal).astype("int64"),
        "_actor1_missing_count": a1.isna().astype("int64"),
        "_actor2_missing_count": a2.isna().astype("int64"),
        "_exporter_gii_sum": (gold * mentions).where(exporter, 0).fillna(0),
        "_importer_gii_sum": (gold * mentions).where(importer, 0).fillna(0),
        "_exporter_gii_count": (
            exporter & gold.notna() & mentions.notna()
        ).astype("int64"),
        "_importer_gii_count": (
            importer & gold.notna() & mentions.notna()
        ).astype("int64"),
    }

    # Global CAMEO root counts
    for code in CAMEO_ROOTS:
        data[f"global_root_{code}"] = root.eq(code).fillna(False).astype("int64")

    # Selected exporter/importer root counts
    for feature in FEATURE_GROUPS["oil_country_cameo"]:
        group, _, code = feature.partition("_root_")
        mask = exporter if group == "exporter" else importer
        data[feature] = (mask & root.eq(code)).fillna(False).astype("int64")

    # Four mutually exclusive directed relationship categories
    a1_exp, a1_imp = a1.isin(EXPORTER_CODES), a1.isin(IMPORTER_CODES)
    a2_exp, a2_imp = a2.isin(EXPORTER_CODES), a2.isin(IMPORTER_CODES)

    relationship_masks = {
        "exp_to_exp": a1_exp & a2_exp,
        "exp_to_imp": a1_exp & a2_imp,
        "imp_to_exp": a1_imp & a2_exp,
        "imp_to_imp": a1_imp & a2_imp,
    }

    for name, mask in relationship_masks.items():
        mask = mask.fillna(False)
        data[f"{name}_events"] = mask.astype("int64")

        for metric, values in [("goldstein", gold), ("tone", tone)]:
            data[f"_{name}_{metric}_sum"] = values.where(mask, 0).fillna(0)
            data[f"_{name}_{metric}_count"] = (
                mask & values.notna()
            ).astype("int64")

    # Major-country features
    for name, code in MAJOR_COUNTRIES.items():
        mask = (a1.eq(code) | a2.eq(code)).fillna(False)
        data[f"{name}_total_events"] = mask.astype("int64")

        if name in ("iran", "russia"):
            data[f"_{name}_gold_sum"] = gold.where(mask, 0).fillna(0)
            data[f"_{name}_gold_count"] = (
                mask & gold.notna()
            ).astype("int64")

    # Aggregate intermediate statistics by reporting date
    batch_data = pd.DataFrame(data)
    batch_data.insert(0, "date", dates.to_numpy())

    return batch_data.groupby("date", sort=True).sum()


def finalize_daily_features(stats):
    """Convert combined daily statistics into the final 84 features."""

    s = stats.copy()
    result = pd.DataFrame(index=s.index)

    def ratio(numerator, denominator):
        return numerator.div(denominator.replace(0, np.nan))

    # 1. General activity
    for col in [
        "global_total_events",
        "exporter_total_events",
        "importer_total_events",
    ]:
        result[col] = s[col]

    result["oil_country_event_share"] = ratio(
        s["_oil_country_events"], s["global_total_events"]
    )

    # 2. CAMEO category counts
    for group in ["global_cameo", "oil_country_cameo"]:
        for col in FEATURE_GROUPS[group]:
            result[col] = s[col]

    # 3. Directed relationship counts and averages
    for relationship in RELATIONSHIPS:
        result[f"{relationship}_events"] = s[f"{relationship}_events"]

        for metric in ["goldstein", "tone"]:
            result[f"{relationship}_{metric}_mean"] = ratio(
                s[f"_{relationship}_{metric}_sum"],
                s[f"_{relationship}_{metric}_count"]
            )

    # 4. Global/exporter/importer Goldstein means
    for group in ["global", "exporter", "importer"]:
        result[f"{group}_goldstein_mean"] = ratio(
            s[f"_{group}_gold_sum"],
            s[f"_{group}_gold_count"]
        )

    # Sample standard deviation (ddof=1)
    n = s["_global_gold_count"]
    variance = (
        s["_global_gold_sq_sum"]
        - s["_global_gold_sum"].pow(2).div(n.replace(0, np.nan))
    ) / (n - 1).replace(0, np.nan)

    result["global_goldstein_std"] = np.sqrt(variance.clip(lower=0))

    # Original Goldstein Impact Index formula from Paper 1
    for group in ["exporter", "importer"]:
        result[f"{group}_goldstein_impact_index"] = ratio(
            s[f"_{group}_gii_sum"],
            s[f"_{group}_gii_count"]
        )

    # 5. News tone
    for group in ["global", "exporter", "importer"]:
        result[f"{group}_tone_mean"] = ratio(
            s[f"_{group}_tone_sum"],
            s[f"_{group}_tone_count"]
        )

    result["global_negative_tone_share"] = ratio(
        s["_global_negative_tone_count"],
        s["global_total_events"]
    )

    # 6. Media attention
    for col in [
        "global_total_mentions",
        "global_total_articles",
        "exporter_total_mentions",
        "exporter_total_articles",
    ]:
        result[col] = s[col]

    result["global_mentions_per_event"] = ratio(
        s["global_total_mentions"],
        s["global_total_events"]
    )

    # 7. Internal events and missingness
    for group in ["exporter", "importer"]:
        result[f"{group}_internal_event_share"] = ratio(
            s[f"_{group}_internal_count"],
            s[f"{group}_total_events"]
        )

    for actor in ["actor1", "actor2"]:
        result[f"{actor}_country_missing_share"] = ratio(
            s[f"_{actor}_missing_count"],
            s["global_total_events"]
        )

    # 8. Major countries
    for country in MAJOR_COUNTRIES:
        result[f"{country}_total_events"] = s[f"{country}_total_events"]

        if country in ["iran", "russia"]:
            result[f"{country}_goldstein_mean"] = ratio(
                s[f"_{country}_gold_sum"],
                s[f"_{country}_gold_count"]
            )

    # Final 84-column structure
    result = result.reset_index()
    result = result[FEATURE_COLUMNS]

    return result


def run_feature_pipeline(input_dir, output_dir, batch_size=100_000):
    """Build daily GDELT geopolitical features from cleaned Parquet files."""

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    parquet_files = sorted(input_dir.rglob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(
            f"No Parquet files found in {input_dir}"
        )

    if batch_size < 1:
        raise ValueError("batch_size must be a positive integer")

    daily_stats = None
    processed_rows = 0

    for file_number, file in enumerate(parquet_files, start=1):
        parquet = pq.ParquetFile(file)
        file_rows = 0

        for batch in parquet.iter_batches(
            batch_size=batch_size,
            columns=REQUIRED_COLUMNS
        ):
            df = batch.to_pandas()

            batch_stats = aggregate_feature_batch(df)

            if daily_stats is None:
                daily_stats = batch_stats
            else:
                daily_stats = daily_stats.add(
                    batch_stats,
                    fill_value=0
                )

            file_rows += len(df)
            processed_rows += len(df)

            del df, batch_stats

        print(
            f"[{file_number}/{len(parquet_files)}] "
            f"{file.parent.name}: {file_rows:,} records",
            flush=True
        )

    daily_features = finalize_daily_features(
        daily_stats.sort_index()
    )

    # Validate resulting dataset
    if daily_features["global_total_events"].sum() != processed_rows:
        raise AssertionError("Event counts do not match input records")

    if list(daily_features.columns) != FEATURE_COLUMNS:
        raise AssertionError("Feature columns do not match specification")

    if daily_features["date"].duplicated().any():
        raise AssertionError("Duplicate reporting dates detected")

    if not np.isfinite(
        daily_features.select_dtypes(include="number").to_numpy()
    ).all():
        raise AssertionError("Non-finite feature values detected")

    daily_features = (
        daily_features
        .sort_values("date")
        .reset_index(drop=True)
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_output = output_dir / "gdelt_daily_features.parquet"
    csv_output = output_dir / "gdelt_daily_features.csv"

    daily_features.to_parquet(
        parquet_output,
        index=False,
        engine="pyarrow",
        compression="snappy"
    )

    daily_features.to_csv(csv_output, index=False)

    print("\nFEATURE ENGINEERING COMPLETE")
    print("Records processed:", f"{processed_rows:,}")
    print("Daily observations:", len(daily_features))
    print("Feature columns:", len(daily_features.columns))
    print("Parquet:", parquet_output)
    print("CSV:", csv_output)

    return daily_features

