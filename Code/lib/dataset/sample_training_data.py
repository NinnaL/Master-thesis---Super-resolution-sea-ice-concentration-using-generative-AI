import os
import numpy as np
import pandas as pd

seed = 42
# interestin columns
mixed_ice_cols = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100']

# min fraction of valid pixels in ice bins
min_mixed_frac = 0.5 # at least 10% must be in ice bins
# max fraction of valid pixels of pure ice and open water
max_pure_frac = 0.8 # at most 90% must be in pure ice and open water bins
# number of samples to draw for training
n_samples = 1000


def load_stats(file_path):
    """ Load data and convert timestamp to datetime, extract year and month. """
    # Avoid mixed-type inference warnings in large CSV files.
    df = pd.read_csv(file_path, low_memory=False)
    df.columns = df.columns.str.strip()

    # Parse timestamps like 20140101T123456.
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(str).str.strip(), format='%Y%m%dT%H%M%S', errors='coerce')
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    return df

def remove_corrupt(df):
    """ Remove rows with errors in either AMSR2 or SIC data. """
    mask = df['error'].notna()
    return df[~mask].copy()

def compute_fractions(df):
    """ Add fraction columns for mixed ice, open water and pure ice. """
    bin_cols = ['val_0'] + mixed_ice_cols + ['val_100']
    total = df[bin_cols].sum(axis=1).replace(0, np.nan) # avoid division by zero

    df['frac_mixed'] = df[mixed_ice_cols].sum(axis=1) / total
    df['frac_pure'] = (df['val_0'] + df['val_100']) / total

    return df

def filter_samples(df, min_mixed=min_mixed_frac, max_pure=max_pure_frac):
    """ Filter samples based on mixed ice and pure fractions. """
    before = len(df)
    mask = (df['frac_mixed'] >= min_mixed) & (df['frac_pure'] <= max_pure)
    df = df[mask].copy()
    print(f'Samples before: {before} \nFiltered samples: {len(df)} \n(min_mixed={min_mixed}, max_pure={max_pure})')
    return df


# Temporal filtering functions
def temporal_filter(df, n_samples=50000, per_year_balance = False):
    """ Sample the data to ensure temporal balance across years and months along with the ice concentration. """
    df = df.copy()
    df['ice_class'] = pd.cut(df['frac_mixed'], bins=[0, 0.3, 0.6, 1.01], labels=['low_mix', 'mid_mix', 'high_mix'])

    df['stratum'] = df['month'].astype(str).str.zfill(2) + '_' + df['ice_class'].astype(str)
    n_strata = df['stratum'].nunique()
    base_per_stratum = max(1, n_samples // n_strata)

    samples_init = []
    for stratum, group in df.groupby('stratum'):
        if per_year_balance and 'year' in group.columns:
            years = group['year'].unique()
            per_year = max(1, base_per_stratum // len(years))
            for year, year_group in group.groupby('year'):
                n = min(per_year, len(year_group))
                samples_init.append(year_group.sample(n=n, random_state=42))
        else:
            n = min(base_per_stratum, len(group))
            samples_init.append(group.sample(n=n, random_state=42))
    
    samples = pd.concat(samples_init).drop_duplicates()

    if len(samples) > n_samples:
        samples = samples.sample(n=n_samples, random_state=42)
    
    print(f'Sampled {len(samples)} scenes with temporal and ice concentration balance across {n_strata} strata.')
    print(samples.groupby(['month', 'ice_class']).size().to_string())
    return samples.reset_index(drop=True)

# ---------------------------------------------------------------------------
# Filtering the loaded pairs

def filter_loader_pairs(matched_pairs, training_idx):
    """ Filter the matched pairs to only include those in the training index. """
    mask = set(zip(training_idx['amsr2_file'], training_idx['sic_file']))

    filtered = []
    for pair in matched_pairs:
        amsr2_basenames = [os.path.basename(f) for f in pair['amsr2_files']]
        sic_basenames = [os.path.basename(f) for f in pair['sic_files']]
        if any((a, s) in mask for a in amsr2_basenames for s in sic_basenames):
            filtered.append(pair)
    
    print(f'Loaded pairs after index filter: {len(filtered)}/{len(matched_pairs)}')
    return filtered

# ---------------------------------------------------------------------------
# Main sampling function
def sample_training_data(input_path, output_path='Training_index.csv', n_samples=100000, min_mixed=min_mixed_frac, max_pure=max_pure_frac):
    """ Combination of above functions: load csv -> remove error columns -> filter samples -> temporal filter -> save index. """
    df = load_stats(input_path)
    df = remove_corrupt(df)
    df = compute_fractions(df)
    df = filter_samples(df, min_mixed=min_mixed, max_pure=max_pure)
    sampled_df = temporal_filter(df, n_samples=n_samples)

    # Save the training index
    sampled_df.to_csv(output_path, index=False)
    print(f'Training index saved to {output_path} with {len(sampled_df)} samples.')
    return sampled_df

if __name__ == '__main__':
    df = sample_training_data(
        input_path    = '/dmidata/users/nili/Master/Master-thesis---Super-resolution-sea-ice-concentration-using-generative-AI/Data/meta/sic_amsr2_metadata_stats_all_years.csv',
        output_path = '/dmidata/users/nili/Master/Master-thesis---Super-resolution-sea-ice-concentration-using-generative-AI/training_index_short.csv',
        n_samples   = n_samples,
        min_mixed   = min_mixed_frac,
        max_pure    = max_pure_frac,
    )
 
    print(df[['timestamp', 'year', 'month', 'ice_class',
              'frac_mixed', 'frac_pure', 'amsr2_file', 'sic_file']].head(20).to_string())