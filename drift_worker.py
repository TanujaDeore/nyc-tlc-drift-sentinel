import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from sqlalchemy import create_engine

# --- DATABASE CONFIGURATION ---
DB_USER = "postgres"
DB_PASS = "postgres"  # <-- Set your pgAdmin master password
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "drift_sentinel"

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

MONITORED_FEATURES = ["fare_amount", "trip_distance", "tip_amount"]


def calculate_psi(reference: np.ndarray, current: np.ndarray, num_bins: int = 10) -> float:
    """Computes the Population Stability Index using reference quantile bins."""
    ref_clean = reference[~np.isnan(reference)]
    curr_clean = current[~np.isnan(current)]

    if len(ref_clean) == 0 or len(curr_clean) == 0:
        return 0.0

    # Create quantile-based bins using the baseline reference
    quantiles = np.linspace(0, 1, num_bins + 1)
    bin_edges = np.quantile(ref_clean, quantiles)
    bin_edges = np.unique(bin_edges)  # remove duplicate edges

    if len(bin_edges) < 2:
        return 0.0

    # Expand boundaries slightly to prevent out-of-bounds drops
    bin_edges[0] -= 1e-5
    bin_edges[-1] += 1e-5

    ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
    curr_counts, _ = np.histogram(curr_clean, bins=bin_edges)

    # Use Laplace smoothing to prevent division by zero / log(0)
    ref_pct = np.where(ref_counts == 0, 1e-4, ref_counts) / len(ref_clean)
    curr_pct = np.where(curr_counts == 0, 1e-4, curr_counts) / len(curr_clean)

    psi_value = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
    return float(psi_value)


def run_drift_analysis():
    print("\n--- Starting Continuous Drift & Degradation Sentinel ---")

    # 1. Load Reference Data
    print("Loading baseline reference dataset...")
    ref_df = pd.read_sql(
        f"SELECT {', '.join(MONITORED_FEATURES)} FROM reference_trips", engine
    )

    # 2. Identify Unique Production Batches
    batches = pd.read_sql(
        "SELECT DISTINCT batch_id FROM inference_stream ORDER BY batch_id", engine
    )["batch_id"].tolist()

    summary_records = []
    incident_records = []

    # 3. Analyze each batch
    for batch_id in batches:
        print(f"\nAnalyzing production batch: [{batch_id}]")
        curr_df = pd.read_sql(
            f"SELECT {', '.join(MONITORED_FEATURES)} FROM inference_stream WHERE batch_id = %(batch)s",
            engine,
            params={"batch": batch_id},
        )

        for feature in MONITORED_FEATURES:
            ref_col = ref_df[feature].to_numpy()
            curr_col = curr_df[feature].to_numpy()

            # Null Analysis
            curr_null_pct = float(np.isnan(curr_col).mean() * 100)
            ref_null_pct = float(np.isnan(ref_col).mean() * 100)
            null_delta = curr_null_pct - ref_null_pct

            # Clean slices for statistical tests
            ref_clean = ref_col[~np.isnan(ref_col)]
            curr_clean = curr_col[~np.isnan(curr_col)]

            # Statistical Distance Calculations
            psi_score = calculate_psi(ref_clean, curr_clean)
            w_dist = float(wasserstein_distance(ref_clean, curr_clean)) if len(curr_clean) > 0 else 0.0
            ks_stat, ks_p_val = ks_2samp(ref_clean, curr_clean) if len(curr_clean) > 0 else (0.0, 1.0)

            # Severity classification based on industry-standard PSI thresholds
            if psi_score >= 0.25 or curr_null_pct >= 20.0:
                severity = "SEVERE"
            elif psi_score >= 0.10 or curr_null_pct >= 5.0:
                severity = "MODERATE"
            else:
                severity = "STABLE"

            summary_records.append({
                "batch_id": batch_id,
                "feature_name": feature,
                "psi_score": round(psi_score, 4),
                "wasserstein_dist": round(w_dist, 4),
                "ks_p_value": round(float(ks_p_val), 6),
                "null_percentage": round(curr_null_pct, 2),
                "drift_severity": severity,
            })

            # Automated Diagnostic Worker: Root-Cause Triage
            if null_delta >= 15.0:
                incident_records.append({
                    "batch_id": batch_id,
                    "feature_name": feature,
                    "root_cause": "UPSTREAM_NULL_SPIKE",
                    "summary": (
                        f"Data pipeline fault detected in {feature}. Null rate jumped by {null_delta:.1f}%. "
                        "Action: Investigate upstream ingestion extraction transformations before blaming ML models."
                    ),
                })
            elif psi_score >= 0.25:
                incident_records.append({
                    "batch_id": batch_id,
                    "feature_name": feature,
                    "root_cause": "COVARIATE_SHIFT",
                    "summary": (
                        f"Substantial population drift in {feature} (PSI={psi_score:.3f}, W-Dist={w_dist:.2f}). "
                        "Action: Inspect macroeconomic shifts or trigger scheduled model retraining."
                    ),
                })

    # 4. Save results back into PostgreSQL
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_sql(
        "drift_metrics_summary", engine, if_exists="append", index=False
    )
    print(f"\n[+] Successfully logged {len(summary_df)} feature metrics to 'drift_metrics_summary'.")

    if incident_records:
        incident_df = pd.DataFrame(incident_records)
        incident_df.to_sql(
            "drift_incidents", engine, if_exists="append", index=False
        )
        print(f"[!] Logged {len(incident_df)} critical alerts to 'drift_incidents'.")

    print("\nSentinel audit run completed successfully.")


if __name__ == "__main__":
    run_drift_analysis()