import os
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MIXED_ICE_COLS = ['0-10', '10-20', '20-30', '30-40', '40-50',
                  '50-60', '60-70', '70-80', '80-90', '90-100']

MIN_MIXED_FRACTION = 0.10   # at least 10% of pixels must be in transition zones
MAX_PURE_FRACTION  = 0.90   # reject if >90% of pixels are val_0 or val_100


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_stats(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y%m%dT%H%M%S', errors='coerce')
    df['year']  = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    return df


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def remove_corrupt(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where the error column is non-null (corrupt AMSR2 or SIC file)."""
    mask = df['error'].notna()
    n    = mask.sum()
    if n:
        print(f'  Removed {n} corrupt rows (non-null error column)')
    return df[~mask].copy()


def compute_fractions(df: pd.DataFrame) -> pd.DataFrame:
    """Add fraction columns for mixed ice, pure open water, and pure ice."""
    bin_cols = ['val_0'] + MIXED_ICE_COLS + ['val_100']
    total    = df[bin_cols].sum(axis=1).replace(0, np.nan)

    df['frac_mixed']      = df[MIXED_ICE_COLS].sum(axis=1) / total
    df['frac_pure_water'] = df['val_0']   / total
    df['frac_pure_ice']   = df['val_100'] / total
    df['frac_pure']       = df['frac_pure_water'] + df['frac_pure_ice']
    return df


def filter_scenes(df: pd.DataFrame,
                  min_mixed: float = MIN_MIXED_FRACTION,
                  max_pure:  float = MAX_PURE_FRACTION) -> pd.DataFrame:
    """
    Keep only scenes with meaningful ice/water transition zones.
    Rejects:
      - Scenes that are nearly all open water  (val_0 dominated)
      - Scenes that are nearly all sea ice     (val_100 dominated)
      - Scenes with too little mixed-concentration area
    """
    before = len(df)
    mask   = (df['frac_mixed'] >= min_mixed) & (df['frac_pure'] <= max_pure)
    df     = df[mask].copy()
    print(f'  Kept {len(df)}/{before} scenes after ice-mix filter '
          f'(min_mixed={min_mixed:.0%}, max_pure={max_pure:.0%})')
    return df


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------

def stratified_sample(df: pd.DataFrame,
                      n_samples:        int  = 5000,
                      per_year_balance: bool = True) -> pd.DataFrame:
    """
    Sample with stratification across:
      - Season     (winter / spring / summer / autumn)
      - Year       (optional balance so no single year dominates)
      - Ice class  (low / mid / high mixed-ice fraction)
    """
    df = df.copy()

    df['ice_class'] = pd.cut(
        df['frac_mixed'],
        bins   = [0,    0.3,  0.6,  1.01],
        labels = ['low_mix', 'mid_mix', 'high_mix'],
        right  = False,
    )

    df['stratum'] = df['month'].astype(str).str.zfill(2) + '_' + df['ice_class'].astype(str)

    n_strata         = df['stratum'].nunique()
    base_per_stratum = max(1, n_samples // n_strata)

    sampled_parts = []
    for stratum, group in df.groupby('stratum'):
        if per_year_balance and 'year' in group.columns:
            years    = group['year'].unique()
            per_year = max(1, base_per_stratum // len(years))
            for yr, yg in group.groupby('year'):
                n = min(per_year, len(yg))
                sampled_parts.append(yg.sample(n, random_state=42))
        else:
            n = min(base_per_stratum, len(group))
            sampled_parts.append(group.sample(n, random_state=42))

    sampled = pd.concat(sampled_parts).drop_duplicates()

    if len(sampled) > n_samples:
        sampled = sampled.sample(n_samples, random_state=42)

    print(f'  Sampled {len(sampled)} scenes across {n_strata} strata')
    print(sampled.groupby(['month', 'ice_class']).size().to_string())
    return sampled.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Integration with SICDataLoader
# ---------------------------------------------------------------------------

def filter_loader_pairs(matched_pairs: list[dict],
                        training_index: pd.DataFrame) -> list[dict]:
    """
    Filter SICDataLoader matched_pairs down to only those in the training index.

    Usage:
        loader = SICDataLoader(...)
        index  = pd.read_csv('training_index.csv')
        pairs  = filter_loader_pairs(loader.get_matched_pairs_info(), index)
    """
    allowed = set(zip(training_index['amsr2_file'], training_index['sic_file']))

    filtered = []
    for pair in matched_pairs:
        amsr2_basenames = [os.path.basename(f) for f in pair['amsr2_files']]
        sic_basenames   = [os.path.basename(f) for f in pair['sic_files']]
        if any((a, s) in allowed for a in amsr2_basenames for s in sic_basenames):
            filtered.append(pair)

    print(f'  Loader pairs after index filter: {len(filtered)}/{len(matched_pairs)}')
    return filtered


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_training_index(csv_path:    str,
                         output_path: str   = 'training_index.csv',
                         n_samples:   int   = 5000,
                         min_mixed:   float = MIN_MIXED_FRACTION,
                         max_pure:    float = MAX_PURE_FRACTION) -> pd.DataFrame:
    """
    Full pipeline: load → remove corrupt → filter → sample → save index CSV.
    The saved CSV is the training index consumed by SICDataLoader.
    """
    print(f'Loading {csv_path}')
    df = load_stats(csv_path)
    print(f'  Total rows: {len(df)}')

    df = remove_corrupt(df)
    df = compute_fractions(df)
    df = filter_scenes(df, min_mixed=min_mixed, max_pure=max_pure)
    df = stratified_sample(df, n_samples=n_samples)

    df.to_csv(output_path, index=False)
    print(f'\nSaved training index → {output_path}')
    return df


if __name__ == '__main__':
    df = build_training_index(
        csv_path    = 'sic_amsr2_metadata_stats_all_years.csv',
        output_path = 'training_index.csv',
        n_samples   = 5000,
        min_mixed   = 0.10,
        max_pure    = 0.90,
    )

    print(df[['timestamp', 'year', 'month', 'ice_class',
              'frac_mixed', 'frac_pure', 'amsr2_file', 'sic_file']].head(20).to_string())