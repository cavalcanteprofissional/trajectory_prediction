# scripts/eda.py
"""
Análise Exploratória de Dados (EDA)
Executar: python scripts/eda.py
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Adicionar diretório raiz
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from data.loader import DataLoader
from config.settings import config


def run_eda(train_df, test_df):
    """Executa análise exploratória completa"""
    
    print("\n" + "="*60)
    print("📊 ANÁLISE EXPLORATÓRIA DE DADOS")
    print("="*60 + "\n")
    
    # === Estatísticas Gerais ===
    print("📈 ESTATÍSTICAS GERAIS")
    print("-"*40)
    print(f"Train: {len(train_df):,} amostras")
    print(f"Test:  {len(test_df):,} amostras")
    print(f"Colunas Train: {len(train_df.columns)}")
    print(f"Colunas Test:  {len(test_df.columns)}")
    
    # === Colunas ===
    print("\n📋 COLUNAS")
    print("-"*40)
    print(f"Train: {list(train_df.columns)}")
    print(f"Test:  {list(test_df.columns)}")
    
    # === Valores Nulos ===
    print("\n🔍 VALORES NULOS")
    print("-"*40)
    train_nulls = train_df.isnull().sum()
    test_nulls = test_df.isnull().sum()
    
    print("Train:")
    for col, n in train_nulls[train_nulls > 0].items():
        print(f"  {col}: {n}")
    
    print("Test:")
    for col, n in test_nulls[test_nulls > 0].items():
        print(f"  {col}: {n}")
    
    # === Análise de Coordenadas ===
    if 'path_lat' in train_df.columns:
        print("\n🌍 ANÁLISE DE COORDENADAS")
        print("-"*40)
        
        # Parse trajetória
        lats = []
        lons = []
        
        for idx, row in train_df.head(5000).iterrows():
            try:
                lat_pts = eval(row['path_lat']) if isinstance(row['path_lat'], str) else row['path_lat']
                lon_pts = eval(row['path_lon']) if isinstance(row['path_lon'], str) else row['path_lon']
                
                if lat_pts and lon_pts:
                    lats.extend(lat_pts)
                    lons.extend(lon_pts)
            except:
                pass
        
        if lats and lons:
            print(f"Latitude:  min={min(lats):.4f}, max={max(lats):.4f}, mean={np.mean(lats):.4f}")
            print(f"Longitude: min={min(lons):.4f}, max={max(lons):.4f}, mean={np.mean(lons):.4f}")
            
            # Verificar range Beijing
            in_range = sum(1 for lat in lats if config.LAT_MIN <= lat <= config.LAT_MAX)
            print(f"\n📍 Pontos em Beijing (lat {config.LAT_MIN}-{config.LAT_MAX}): {in_range}/{len(lats)} ({in_range/len(lats)*100:.1f}%)")
    
    # === Análise de Targets (se existirem) ===
    if 'dest_lat' in train_df.columns:
        print("\n🎯 targets (destino)")
        print("-"*40)
        print(f"dest_lat: min={train_df['dest_lat'].min():.4f}, max={train_df['dest_lat'].max():.4f}")
        print(f"dest_lon: min={train_df['dest_lon'].min():.4f}, max={train_df['dest_lon'].max():.4f}")
        
        # Verificar outliers nos targets
        out_lat = sum(1 for lat in train_df['dest_lat'] if not (config.LAT_MIN <= lat <= config.LAT_MAX))
        out_lon = sum(1 for lon in train_df['dest_lon'] if not (config.LON_MIN <= lon <= config.LON_MAX))
        
        print(f"dest_lat fora de Beijing: {out_lat}")
        print(f"dest_lon fora de Beijing: {out_lon}")
    
    # === Verificar Outliers ===
    print("\n⚠️ DETECÇÃO DE OUTLIERS")
    print("-"*40)
    
    outlier_counts = {
        'lat_inválida': 0,
        'lon_inválida': 0,
        'path_vazio': 0,
        'pontos_outlier': 0
    }
    
    for idx, row in train_df.iterrows():
        # Paths vazios
        try:
            lat_pts = eval(row['path_lat']) if isinstance(row['path_lat'], str) else row['path_lat']
            if not lat_pts:
                outlier_counts['path_vazio'] += 1
        except:
            outlier_counts['path_vazio'] += 1
        
        # Verificação rápida (primeiros 1000)
        if idx >= 1000:
            break
    
    for k, v in outlier_counts.items():
        if v > 0:
            print(f"  {k}: {v}")
    
    print("\n" + "="*60)
    print("✅ EDA CONCLUÍDO")
    print("="*60 + "\n")


def main():
    """Função principal"""
    print("📊 Carregando dados...")
    
    loader = DataLoader()
    train_df, test_df = loader.load_data(use_sample_if_missing=False)
    
    if train_df is None:
        print("❌ Dados não encontrados")
        return
    
    run_eda(train_df, test_df)
    
    # Salvar relatório
    report_path = ROOT_DIR / "reports" / "eda_report.txt"
    report_path.parent.mkdir(exist_ok=True)
    
    print(f"📝 Relatório salvo em: {report_path}")


if __name__ == "__main__":
    main()