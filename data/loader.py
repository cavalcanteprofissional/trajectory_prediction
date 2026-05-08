# data/loader.py
"""
Data Loader - Versão com Suporte a Kaggle Dataset

Suporta:
- Arquivos locais train.csv e test.csv
- Download automático do Kaggle Dataset
- Upload de novos dados via interface
- Múltiplos formatos de entrada
"""
import pandas as pd
import numpy as np
import ast
from pathlib import Path
import sys
import os
import shutil

# Adicionar diretório raiz ao path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config.settings import config

KAGGLE_DATASET = "muitomalakoi/trajectory-prediction-beijing"


def download_kaggle_dataset():
    """
    Baixa o dataset do Kaggle usando kagglehub.
    
    Returns:
        Path: Caminho para o diretório dos dados
    """
    try:
        import kagglehub
        import os
        
        # Configurar credenciais do Kaggle a partir do .env.local
        username = os.getenv('KAGGLE_USERNAME')
        key = os.getenv('KAGGLE_KEY')
        
        if username and key:
            # Configurar variáveis de ambiente para Kaggle
            os.environ['KAGGLE_USERNAME'] = username
            os.environ['KAGGLE_KEY'] = key
            print(f"Configurado Kaggle para usuário: {username}")
        
        print(f"Baixando dataset do Kaggle: {KAGGLE_DATASET}")
        path = kagglehub.dataset_download(KAGGLE_DATASET)
        print(f"Dataset baixado para: {path}")
        
        # Mover arquivos para data/
        data_dir = ROOT_DIR / "data"
        data_dir.mkdir(exist_ok=True)
        
        # Copiar arquivos
        source_path = Path(path)
        for file in ["train.csv", "test.csv"]:
            src = source_path / file
            if src.exists():
                dst = data_dir / file
                print(f"Copiando {file} para {dst}")
                shutil.copy2(src, dst)
        
        return data_dir
        
    except ImportError:
        print("kagglehub não instalado. Execute: pip install kagglehub")
        raise
    except Exception as e:
        print(f"Erro ao baixar dataset: {e}")
        raise


class DataLoader:
    """Classe para carregamento e pré-processamento de dados"""
    
    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.sample_submission = None
        self.logger = self._get_logger()
    
    def _get_logger(self):
        """Obtém logger de forma segura"""
        try:
            from utils.logger import get_logger
            return get_logger(__name__)
        except ImportError:
            import logging
            logging.basicConfig(level=logging.INFO)
            return logging.getLogger(__name__)
    
    def ensure_data_exists(self, download_if_missing=True):
        """
        Verifica se os dados existem localmente e baixa do Kaggle se necessário.
        
        Args:
            download_if_missing: Se True, baixa do Kaggle se não encontrar localmente
        
        Returns:
            bool: True se os dados existem (ou foram baixados)
        """
        # procurar por diferentes nomes de arquivo
        train_path = config.get_train_path()
        test_path = config.get_test_path()
        
        all_exist = train_path.exists() and test_path.exists()
        
        if all_exist:
            self.logger.info("✅ Arquivos de dados encontrados localmente")
            return True
        
        # Dados não encontrados localmente
        if download_if_missing:
            self.logger.info("⬇ Baixando dataset do Kaggle...")
            try:
                download_kaggle_dataset()
                self.logger.info("✅ Dataset baixado com sucesso!")
                return True
            except Exception as e:
                self.logger.error(f"❌ Erro ao baixar: {e}")
                return False
        
        self.logger.warning("⚠ Arquivos de dados não encontrados")
        self.logger.info(f"   Procurando em: {config.DATA_DIR}")
        self.logger.info(f"   Train: {train_path}")
        self.logger.info(f"   Test: {test_path}")
        return False
    
    @staticmethod
    def parse_path_string(path_str):
        """Converte string de lista para lista de floats"""
        try:
            if isinstance(path_str, str):
                # Remover possíveis espaços extras
                path_str = path_str.strip()
                return ast.literal_eval(path_str)
            elif isinstance(path_str, list):
                return path_str
            elif pd.isna(path_str):
                return []
            else:
                return []
        except (ValueError, SyntaxError) as e:
            print(f"⚠ Erro ao parsear string: {path_str[:50]}... - Erro: {e}")
            return []
    
    def load_data(self, use_sample_if_missing=True, train_path=None, test_path=None):
        """
        Carrega os dados de treino e teste de arquivos locais.
        
        Args:
            use_sample_if_missing: Se True, cria dados de exemplo se reais não existirem
            train_path: Caminho personalizado para dados de treino (opcional)
            test_path: Caminho personalizado para dados de teste (opcional)
        
        Returns:
            tuple: (train_data, test_data)
        """
        # Usar caminhos fornecidos ou os padrões
        train_file = train_path or config.get_train_path()
        test_file = test_path or config.get_test_path()
        
        # Verificar se os dados existem
        if not train_file.exists() or not test_file.exists():
            if use_sample_if_missing:
                self.logger.warning("⚠ Arquivos não encontrados. Usando dados de exemplo.")
                return self.create_sample_data()
            else:
                raise FileNotFoundError(
                    f"Arquivos não encontrados.\n"
                    f"Train: {train_file}\n"
                    f"Test: {test_file}\n"
                    f"Disponha os arquivos CSV em: {config.DATA_DIR}"
                )
        
        # Carregar dados reais
        try:
            self.logger.info(f"📖 Carregando dados de {config.DATA_DIR}")
            self.logger.info(f"   Train: {train_file.name}")
            self.logger.info(f"   Test: {test_file.name}")
            
            # Carregar train.csv
            self.train_data = pd.read_csv(train_file)
            self.logger.info(f"✅ Train carregado: {len(self.train_data)} linhas, {len(self.train_data.columns)} colunas")
            
            # Carregar test.csv
            self.test_data = pd.read_csv(test_file)
            self.logger.info(f"✅ Test carregado: {len(self.test_data)} linhas, {len(self.test_data.columns)} colunas")
            
            # Parse das trajetórias
            self._parse_trajectories()
            
            # Validar dados
            self._validate_data()
            
            return self.train_data, self.test_data
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao carregar dados: {e}")
            
            if use_sample_if_missing:
                self.logger.info("📝 Criando dados de exemplo...")
                return self.create_sample_data()
            else:
                raise
    
    def load_csv(self, filepath, validate=True):
        """
        Carrega dados de um arquivo CSV qualquer.
        
        Args:
            filepath: Caminho do arquivo CSV
            validate: Se True, valida o formato
        
        Returns:
            DataFrame pandas
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")
        
        df = pd.read_csv(filepath)
        self.logger.info(f"✅ Carregado: {filepath.name} ({len(df)} linhas)")
        
        if validate:
            self._validate_csv_format(df)
        
        return df
    
    def _validate_csv_format(self, df):
        """Valida o formato básico do CSV"""
        required_cols = ['path_lat', 'path_lon']
        
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            self.logger.warning(f"⚠ Colunas faltando: {missing}")
            self.logger.info(f"   Colunas encontradas: {list(df.columns)}")
        
        return len(missing) == 0
    
    def validate_trajectory_data(self, df):
        """
        Valida dados de trajetória.
        
        Returns:
            dict: Estatísticas de validação
        """
        stats = {
            'total': len(df),
            'with_path': 0,
            'empty_paths': 0,
            'invalid_coords': 0,
            'out_of_range': 0
        }
        
        if 'path_lat_parsed' not in df.columns:
            df['path_lat_parsed'] = df['path_lat'].apply(self.parse_path_string)
            df['path_lon_parsed'] = df['path_lon'].apply(self.parse_path_string)
        
        for idx, row in df.iterrows():
            if 'path_lat_parsed' in row and len(row.get('path_lat_parsed', [])) > 0:
                stats['with_path'] += 1
                
                # Verificar range
                lats = row.get('path_lat_parsed', [])
                lons = row.get('path_lon_parsed', [])
                
                if lats and lons:
                    all_in_range = (
                        config.LAT_MIN <= min(lats) <= config.LAT_MAX and
                        config.LAT_MIN <= max(lats) <= config.LAT_MAX and
                        config.LON_MIN <= min(lons) <= config.LON_MAX and
                        config.LON_MIN <= max(lons) <= config.LON_MAX
                    )
                    if not all_in_range:
                        stats['out_of_range'] += 1
            else:
                stats['empty_paths'] += 1
        
        return stats
    
    @staticmethod
    def convert_discrete_to_list_format(df):
        """
        Converte formato discreto para formato lista.
        
        Formato discreto esperado:
        trajectory_id,lat,lon,sequence
        000_001,39.9,116.3,1
        000_001,40.0,116.4,2
        
        Retorna DataFrame no formato lista:
        trajectory_id,path_lat,path_lon
        000_001,"[39.9,40.0]","[116.3,116.4]"
        """
        if df is None:
            return None
        
        # Verificar colunas necessárias para formato discreto
        required = {'trajectory_id', 'lat', 'lon'}
        if not required.issubset(set(df.columns)):
            return df  # Retorna original se não for formato discreto
        
        # Agrupar por trajectory_id
        result = []
        
        for tid, group in df.groupby('trajectory_id'):
            group = group.sort_values('lat' if 'lat' in df.columns else 'sequence')
            
            lats = group['lat'].tolist()
            lons = group['lon'].tolist()
            
            result.append({
                'trajectory_id': tid,
                'path_lat': str(lats),
                'path_lon': str(lons)
            })
        
        return pd.DataFrame(result)
    
    @staticmethod
    def detect_format(df):
        """
        Detecta o formato dos dados.
        
        Returns:
            str: 'list' ou 'discrete'
        """
        if df is None:
            return 'unknown'
        
        # Formato lista: path_lat contém listas
        if 'path_lat' in df.columns:
            sample = str(df['path_lat'].iloc[0])
            if '[' in sample:
                return 'list'
        
        # Formato discreto: lat单个数值
        if 'lat' in df.columns and 'sequence' in df.columns:
            return 'discrete'
        
        return 'unknown'
    
    def _parse_trajectories(self):
        """Parse das colunas de trajetória"""
        # Verificar colunas existentes
        if 'path_lat' in self.train_data.columns:
            self.train_data['path_lat_parsed'] = self.train_data['path_lat'].apply(self.parse_path_string)
            self.train_data['path_lon_parsed'] = self.train_data['path_lon'].apply(self.parse_path_string)
        
        if 'path_lat' in self.test_data.columns:
            self.test_data['path_lat_parsed'] = self.test_data['path_lat'].apply(self.parse_path_string)
            self.test_data['path_lon_parsed'] = self.test_data['path_lon'].apply(self.parse_path_string)
    
    def _validate_data(self):
        """Valida os dados carregados"""
        self.logger.info("🔍 Validando dados...")
        
        if self.train_data is not None:
            train_nulls = self.train_data.isnull().sum().sum()
            self.logger.info(f"   Train - Valores nulos: {train_nulls}")
            
            if 'path_lat_parsed' in self.train_data.columns:
                train_empty = (self.train_data['path_lat_parsed'].str.len() == 0).sum()
                self.logger.info(f"   Train - Trajetórias vazias: {train_empty}")
        
        if self.test_data is not None:
            test_nulls = self.test_data.isnull().sum().sum()
            self.logger.info(f"   Test - Valores nulos: {test_nulls}")
            
            if 'path_lat_parsed' in self.test_data.columns:
                test_empty = (self.test_data['path_lat_parsed'].str.len() == 0).sum()
                self.logger.info(f"   Test - Trajetórias vazias: {test_empty}")
    
    def get_data_summary(self):
        """Retorna resumo dos dados"""
        if self.train_data is None or self.test_data is None:
            raise ValueError("Dados não carregados. Execute load_data() primeiro.")
        
        summary = {
            'train_samples': len(self.train_data),
            'test_samples': len(self.test_data),
            'train_columns': list(self.train_data.columns),
            'test_columns': list(self.test_data.columns),
            'train_memory_mb': self.train_data.memory_usage(deep=True).sum() / 1024**2,
            'test_memory_mb': self.test_data.memory_usage(deep=True).sum() / 1024**2,
            'has_target': 'dest_lat' in self.train_data.columns and 'dest_lon' in self.train_data.columns
        }
        
        return summary
    
    def create_sample_data(self, n_train=1000, n_test=200):
        """Cria dados de exemplo (fallback)"""
        self.logger.info(f"📝 Criando dados de exemplo: {n_train} train, {n_test} test")
        
        np.random.seed(config.SEED)
        
        # Função para criar trajetória realista
        def create_trajectory(n_points=None):
            if n_points is None:
                n_points = np.random.randint(5, 20)
            
            start_lat = 40.0 + np.random.uniform(-2, 2)
            start_lon = -73.0 + np.random.uniform(-2, 2)
            
            # Criar pontos com direção
            lat_points = [start_lat]
            lon_points = [start_lon]
            
            for _ in range(1, n_points):
                lat_points.append(lat_points[-1] + np.random.uniform(-0.01, 0.02))
                lon_points.append(lon_points[-1] + np.random.uniform(-0.01, 0.02))
            
            return lat_points, lon_points
        
        # Dados de treino
        train_records = []
        for i in range(n_train):
            n_points = np.random.randint(5, 20)
            lat_points, lon_points = create_trajectory(n_points)
            
            # Destino baseado na direção
            if len(lat_points) > 1:
                lat_trend = lat_points[-1] - lat_points[0]
                lon_trend = lon_points[-1] - lon_points[0]
                dest_lat = lat_points[-1] + lat_trend * np.random.uniform(1, 3)
                dest_lon = lon_points[-1] + lon_trend * np.random.uniform(1, 3)
            else:
                dest_lat = 40.0
                dest_lon = -73.0
            
            train_records.append({
                'trajectory_id': f'train_{i:04d}',
                'path_lat': str(lat_points),
                'path_lon': str(lon_points),
                'dest_lat': dest_lat,
                'dest_lon': dest_lon
            })
        
        self.train_data = pd.DataFrame(train_records)
        
        # Dados de teste
        test_records = []
        for i in range(n_test):
            n_points = np.random.randint(5, 20)
            lat_points, lon_points = create_trajectory(n_points)
            
            test_records.append({
                'trajectory_id': f'test_{i:04d}',
                'path_lat': str(lat_points),
                'path_lon': str(lon_points)
            })
        
        self.test_data = pd.DataFrame(test_records)
        
        # Parse das trajetórias
        self._parse_trajectories()
        
        self.logger.info(f"✅ Dados de exemplo criados")
        
        return self.train_data, self.test_data
    
    def save_sample_data(self):
        """Salva os dados de exemplo para referência"""
        if self.train_data is not None:
            sample_train_path = config.DATA_DIR / 'train_sample.csv'
            self.train_data.to_csv(sample_train_path, index=False)
            self.logger.info(f"💾 Train sample salvo: {sample_train_path}")
        
        if self.test_data is not None:
            sample_test_path = config.DATA_DIR / 'test_sample.csv'
            self.test_data.to_csv(sample_test_path, index=False)
            self.logger.info(f"💾 Test sample salvo: {sample_test_path}")

# Função de conveniência
def load_trajectory_data():
    """Carrega dados de forma simples"""
    loader = DataLoader()
    return loader.load_data()