import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Adicionar config
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

try:
    from config.settings import config
except ImportError:
    class FallbackConfig:
        LAT_MIN, LAT_MAX = 39.0, 41.0
        LON_MIN, LON_MAX = 115.0, 117.0
        MAX_JUMP_KM = 50.0
        MAX_SPEED_KMH = 200.0
    config = FallbackConfig()


def _ensure_list_like(x):
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, str):
        try:
            return eval(x)
        except:
            return []
    return []


def _parse_path(lat_str, lon_str):
    """Parse path de string ou lista"""
    lat = _ensure_list_like(lat_str)
    lon = _ensure_list_like(lon_str)
    return lat, lon


def clean_dataframe(df: pd.DataFrame, *, is_train: bool = True, min_points: int = 2, drop_missing_target: bool = True) -> pd.DataFrame:
    """Limpeza conservadora do DataFrame de trajetórias.

    Regras aplicadas:
    - Remove duplicatas por trajectory_id.
    - Converte colunas numéricas para tipos numéricos.
    - Remove coordenadas fora dos limites do config (Beijing).
    - Remove paths vazios ou muito curtos.
    - Drop linhas com target ausente (para treino).
    """
    if df is None:
        return df

    df = df.copy()
    original_len = len(df)

    # Remove trajectory específica se existir
    if is_train and 'trajectory_id' in df.columns:
        df = df[df['trajectory_id'] != '128_20090222093321']

    # Drop duplicates
    if 'trajectory_id' in df.columns:
        df = df.drop_duplicates(subset=['trajectory_id'])

    # Convert numeric columns
    for col in ['start_lat', 'start_lon', 'end_lat', 'end_lon', 'dest_lat', 'dest_lon']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remove coordinates outside valid range (from config)
    for col, min_val, max_val in [
        ('start_lat', config.LAT_MIN, config.LAT_MAX),
        ('end_lat', config.LAT_MIN, config.LAT_MAX),
        ('dest_lat', config.LAT_MIN, config.LAT_MAX),
    ]:
        if col in df.columns:
            df = df[df[col].between(min_val, max_val) | df[col].isna()]

    for col, min_val, max_val in [
        ('start_lon', config.LON_MIN, config.LON_MAX),
        ('end_lon', config.LON_MIN, config.LON_MAX),
        ('dest_lon', config.LON_MIN, config.LON_MAX),
    ]:
        if col in df.columns:
            df = df[df[col].between(min_val, max_val) | df[col].isna()]

    # Parse paths
    if 'path_lat_parsed' in df.columns and 'path_lon_parsed' in df.columns:
        df['path_lat_parsed'] = df['path_lat_parsed'].apply(_ensure_list_like)
        df['path_lon_parsed'] = df['path_lon_parsed'].apply(_ensure_list_like)
    else:
        if 'path_lat' in df.columns and 'path_lon' in df.columns:
            df['path_lat_parsed'], df['path_lon_parsed'] = zip(
                *df.apply(lambda r: _parse_path(r.get('path_lat'), r.get('path_lon')), axis=1)
            )

    # Ensure columns exist
    if 'path_lat_parsed' not in df.columns:
        df['path_lat_parsed'] = [[] for _ in range(len(df))]
    if 'path_lon_parsed' not in df.columns:
        df['path_lon_parsed'] = [[] for _ in range(len(df))]

    # Drop rows with too few points
    if is_train:
        mask = df['path_lat_parsed'].apply(lambda x: len(x) >= min_points if isinstance(x, list) else False)
        df = df[mask]

    # Drop missing targets
    if is_train and drop_missing_target:
        if 'dest_lat' in df.columns and 'dest_lon' in df.columns:
            df = df[df['dest_lat'].notna() & df['dest_lon'].notna()]

    df = df.reset_index(drop=True)
    
    return df


def clean_train_test(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
    """Aplica limpeza tanto no treino quanto no teste."""
    train_clean = clean_dataframe(train_df, is_train=True)
    test_clean = clean_dataframe(test_df, is_train=False)
    return train_clean, test_clean


def detect_outliers(df, max_jump_km=50.0, max_speed_kmh=200.0):
    """Detecta outliers nas trajetórias.
    
    Returns:
        dict com contagem de outliers por tipo
    """
    from math import radians, sin, cos, sqrt, atan2
    
    def haversine_km(lat1, lon1, lat2, lon2):
        """Calcula distância Haversine em km"""
        R = 6371  # Earth's radius in km
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return R * c
    
    stats = {
        'total': len(df),
        'empty_paths': 0,
        'large_jumps': 0,
        'impossible_speeds': 0,
        'out_of_region': 0
    }
    
    for idx, row in df.head(5000).iterrows():  # Sample for speed
        lat_pts = row.get('path_lat_parsed', [])
        lon_pts = row.get('path_lon_parsed', [])
        
        if not lat_pts or not lon_pts or len(lat_pts) < 2:
            stats['empty_paths'] += 1
            continue
        
        # Check for large jumps and speeds
        for i in range(1, len(lat_pts)):
            try:
                dist = haversine_km(lat_pts[i-1], lon_pts[i-1], lat_pts[i], lon_pts[i])
                
                if dist > max_jump_km:
                    stats['large_jumps'] += 1
                
                # Assume 1 point per second for speed calculation
                speed = dist * 3600  # km/h if 1 sec between points
                if speed > max_speed_kmh:
                    stats['impossible_speeds'] += 1
            except:
                pass
        
        # Check if in Beijing region
        if lat_pts and lon_pts:
            if not (config.LAT_MIN <= min(lat_pts) <= config.LAT_MAX and 
                   config.LON_MIN <= min(lon_pts) <= config.LON_MAX):
                stats['out_of_region'] += 1
    
    return stats


def clean_dataframe(df: pd.DataFrame, *, is_train: bool = True, min_points: int = 2, drop_missing_target: bool = True) -> pd.DataFrame:
    """Limpeza conservadora do DataFrame de trajetórias.

    Regras aplicadas:
    - Remove duplicatas por `trajectory_id` mantendo a primeira ocorrência.
    - Converte colunas numéricas básicas para tipos numéricos (coerce errors).
    - Remove coordenadas impossíveis (lat fora de [-90,90], lon fora de [-180,180]).
    - Garante que `path_lat_parsed` e `path_lon_parsed` sejam listas; drop se pontos < min_points para treino.
    - Para treino, opcionalmente remove linhas com target ausente (`dest_lat`/`dest_lon`) se `drop_missing_target`.

    Esta função é intencionalmente conservadora: evita transformações complexas que possam mascarar problemas.
    """
    if df is None:
        return df

    df = df.copy()

    # Remove specific trajectory from training set
    if is_train and 'trajectory_id' in df.columns:
        df = df[df['trajectory_id'] != '128_20090222093321']

    # Drop duplicates by trajectory_id if present
    if 'trajectory_id' in df.columns:
        df = df.drop_duplicates(subset=['trajectory_id'])

    # Convert numeric columns if exist
    for col in ['start_lat', 'start_lon', 'end_lat', 'end_lon', 'dest_lat', 'dest_lon']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remove impossible coordinates
    if 'start_lat' in df.columns:
        df = df[df['start_lat'].between(-90, 90) | df['start_lat'].isna()]
    if 'end_lat' in df.columns:
        df = df[df['end_lat'].between(-90, 90) | df['end_lat'].isna()]
    if 'start_lon' in df.columns:
        df = df[df['start_lon'].between(-180, 180) | df['start_lon'].isna()]
    if 'end_lon' in df.columns:
        df = df[df['end_lon'].between(-180, 180) | df['end_lon'].isna()]

    # Handle parsed path columns
    if 'path_lat_parsed' in df.columns and 'path_lon_parsed' in df.columns:
        df['path_lat_parsed'] = df['path_lat_parsed'].apply(_ensure_list_like)
        df['path_lon_parsed'] = df['path_lon_parsed'].apply(_ensure_list_like)
    else:
        # ensure columns exist to avoid KeyErrors downstream
        if 'path_lat_parsed' not in df.columns:
            df['path_lat_parsed'] = [[] for _ in range(len(df))]
        if 'path_lon_parsed' not in df.columns:
            df['path_lon_parsed'] = [[] for _ in range(len(df))]

    # Drop rows with too few points in train set (insufficient trajectory information)
    if is_train:
        mask_enough_points = df['path_lat_parsed'].apply(lambda x: len(x) >= min_points)
        if mask_enough_points.sum() < len(df):
            df = df[mask_enough_points]

    # Drop rows with missing targets for training
    if is_train and drop_missing_target:
        if 'dest_lat' in df.columns and 'dest_lon' in df.columns:
            df = df[df['dest_lat'].notna() & df['dest_lon'].notna()]

    # Reset index for safety
    df = df.reset_index(drop=True)

    return df


def clean_train_test(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
    """Aplica limpeza tanto no treino quanto no teste usando regras conservadoras."""
    train_clean = clean_dataframe(train_df, is_train=True)
    test_clean = clean_dataframe(test_df, is_train=False)
    return train_clean, test_clean
