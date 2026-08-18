"""
Multi-dataset processor: Harmonizes UNSW-NB15 and CIC-IDS2017 into compatible format.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def normalize_cicids2017(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert CIC-IDS2017 columns to UNSW-NB15-compatible schema.
    """
    df = df.copy()
    
    # Clean column names (remove leading/trailing spaces)
    df.columns = df.columns.str.strip()
    
    # Create standardized label (0=Normal, 1=Attack)
    if 'Label' in df.columns:
        df['label'] = (df['Label'].str.lower() != 'benign').astype(int)
        df['attack_cat'] = df['Label'].str.strip()
    
    # Map CIC-IDS columns to UNSW-compatible names
    column_mapping = {
        'Protocol': 'proto',
        'Source Port': 'sport',
        'Destination Port': 'dport',
        'Flow Duration': 'dur',
        'Total Fwd Packets': 'spkts',
        'Total Backward Packets': 'dpkts',
        'Total Length of Fwd Packets': 'sbytes',
        'Total Length of Bwd Packets': 'dbytes',
        'Flow Bytes/s': 'rate',
        'Fwd Packet Length Mean': 'smean',
        'Bwd Packet Length Mean': 'dmean',
        'Flow IAT Mean': 'sinpkt',
        'Bwd IAT Mean': 'dinpkt',
        'SYN Flag Count': 'ct_state_ttl',
        'ACK Flag Count': 'ct_srv_src',
        'FIN Flag Count': 'ct_dst_ltm',
    }
    
    for cic_col, unsw_col in column_mapping.items():
        if cic_col in df.columns:
            df[unsw_col] = pd.to_numeric(df[cic_col], errors='coerce').fillna(0)
    
    # Extract protocol as string
    if 'Protocol' in df.columns:
        df['proto'] = df['Protocol'].astype(str).str.lower()
        df.loc[df['proto'] == '6', 'proto'] = 'tcp'
        df.loc[df['proto'] == '17', 'proto'] = 'udp'
        df.loc[~df['proto'].isin(['tcp', 'udp']), 'proto'] = 'tcp'
    
    # Infer service from destination port
    common_services = {
        21: 'ftp', 22: 'ssh', 23: 'telnet', 25: 'smtp', 53: 'dns',
        80: 'http', 110: 'pop3', 143: 'imap', 443: 'https',
        3306: 'mysql', 5432: 'postgres', 8080: 'http'
    }
    if 'Destination Port' in df.columns:
        # Some CIC files contain NaN/inf in port columns; coerce safely before mapping.
        dst_port = pd.to_numeric(df['Destination Port'], errors='coerce')
        dst_port = dst_port.replace([np.inf, -np.inf], np.nan).fillna(-1).astype(int)
        df['service'] = dst_port.map(lambda x: common_services.get(x, '-'))
    else:
        df['service'] = '-'
    
    # Infer connection state
    if 'SYN Flag Count' in df.columns and 'FIN Flag Count' in df.columns:
        syn_count = pd.to_numeric(df['SYN Flag Count'], errors='coerce').fillna(0)
        fin_count = pd.to_numeric(df['FIN Flag Count'], errors='coerce').fillna(0)
        df['state'] = 'CON'
        df.loc[syn_count > 0, 'state'] = 'SYN'
        df.loc[fin_count > 0, 'state'] = 'FIN'
    else:
        df['state'] = 'CON'
    
    # Source IP and basic ID
    if 'Source IP' in df.columns:
        df['srcip'] = df['Source IP'].astype(str)
    else:
        df['srcip'] = 'unknown'
    
    df['id'] = range(len(df))
    
    return df


def normalize_unswnb15(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure UNSW-NB15 format consistency.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()
    
    if 'label' not in df.columns:
        df['label'] = 0
    if 'attack_cat' not in df.columns:
        df['attack_cat'] = 'Unknown'
    
    return df


def standardize_datasets(df: pd.DataFrame, dataset_type: str = 'auto') -> pd.DataFrame:
    """
    Standardize any dataset to common schema.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()
    
    # Auto-detect dataset type
    if dataset_type == 'auto':
        if 'Source IP' in df.columns and 'Destination Port' in df.columns:
            dataset_type = 'cicids2017'
        elif 'srcip' in df.columns or 'proto' in df.columns:
            dataset_type = 'unswnb15'
        else:
            dataset_type = 'unswnb15'
    
    if dataset_type.lower() in ['cicids2017', 'cicids', 'cic']:
        df = normalize_cicids2017(df)
    else:
        df = normalize_unswnb15(df)
    
    # Ensure all required columns exist
    required_cols = ['label', 'attack_cat', 'proto', 'service', 'state', 'id']
    for col in required_cols:
        if col not in df.columns:
            if col == 'label':
                df[col] = 0
            elif col == 'attack_cat':
                df[col] = 'Unknown'
            elif col == 'id':
                df[col] = range(len(df))
            else:
                df[col] = 'unknown'
    
    # Handle NaNs in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    return df


def process_all_raw_datasets(raw_dir: str = 'data/raw', output_dir: str = 'data/processed') -> str:
    """
    Process all CSV files in raw directory and create combined cleaned dataset.
    """
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    csv_files = list(raw_path.glob('*.csv'))
    if not csv_files:
        print("No CSV files found in data/raw/")
        return None
    
    dfs = []
    
    for csv_file in csv_files:
        print(f"Processing {csv_file.name}...", end=" ")
        try:
            # Detect dataset type from filename
            if 'ISCX' in csv_file.name or 'cicids' in csv_file.name.lower():
                dataset_type = 'cicids2017'
            else:
                dataset_type = 'unswnb15'
            
            # Read with error handling
            try:
                df = pd.read_csv(csv_file, low_memory=False)
            except UnicodeDecodeError:
                df = pd.read_csv(csv_file, encoding='iso-8859-1', low_memory=False)
            
            # Standardize
            df = standardize_datasets(df, dataset_type=dataset_type)
            
            # Remove exact duplicates
            df = df.drop_duplicates()
            
            dfs.append(df)
            print(f"✓ ({len(df)} rows)")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    if not dfs:
        print("Failed to process any files!")
        return None
    
    # Combine all datasets
    print(f"\nCombining {len(dfs)} datasets...", end=" ")
    combined = pd.concat(dfs, ignore_index=True)
    print(f"✓ ({len(combined)} total rows)")
    
    # Remove duplicates across all datasets.
    # For very large merged frames this can exceed memory; switch to a compact subset.
    print("Removing duplicates...", end=" ")
    initial_rows = len(combined)
    if initial_rows > 1_500_000:
        dedup_subset = [
            c
            for c in [
                'label',
                'proto',
                'service',
                'state',
                'dur',
                'spkts',
                'dpkts',
                'sbytes',
                'dbytes',
                'rate',
            ]
            if c in combined.columns
        ]
        combined = combined.drop_duplicates(subset=dedup_subset, keep='first')
    else:
        combined = combined.drop_duplicates(
            subset=[c for c in combined.columns if c not in ['id', 'attack_cat']],
            keep='first',
        )
    removed = initial_rows - len(combined)
    print(f"✓ (Removed {removed} duplicates)")
    
    # Clean data
    print("Cleaning data...", end=" ")
    # Remove rows with all NaN features
    feature_cols = [c for c in combined.columns if c not in ['id', 'label', 'attack_cat', 'srcip', 'proto', 'service', 'state']]
    combined = combined.dropna(subset=feature_cols, how='all')
    
    # Fix inf values
    combined = combined.replace([np.inf, -np.inf], np.nan)
    
    print(f"✓ ({len(combined)} clean rows)")
    
    # Save combined dataset
    output_file = output_path / 'multi_dataset_combined.csv'
    combined.to_csv(output_file, index=False)
    print(f"\n✅ Combined dataset saved: {output_file}")
    
    # Save summary stats
    print("\n📊 Dataset Summary:")
    print(f"  Total samples: {len(combined):,}")
    print(f"  Normal: {(combined['label'] == 0).sum():,} ({(combined['label'] == 0).sum() / len(combined) * 100:.1f}%)")
    print(f"  Attack: {(combined['label'] == 1).sum():,} ({(combined['label'] == 1).sum() / len(combined) * 100:.1f}%)")
    print(f"  Columns: {combined.shape[1]}")
    print(f"  Attack categories: {combined['attack_cat'].nunique()}")
    
    print(f"\n📋 Attack categories:")
    for cat, count in combined['attack_cat'].value_counts().head(10).items():
        print(f"    {cat}: {count:,}")
    
    return str(output_file)


if __name__ == "__main__":
    output = process_all_raw_datasets()
    if output:
        print(f"\n✨ Ready to use: {output}")
