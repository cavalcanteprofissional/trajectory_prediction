# config/settings.py
import os
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# Carregar variáveis de ambiente (usa .env.local)
load_dotenv('.env.local')

class Config:
    """Configurações do projeto - Modo Independente (sem Kaggle)"""
    
    # Seeds para reprodutibilidade
    SEED = int(os.getenv('SEED', 42))
    
    # Configurações do modelo
    DEFAULT_MODELS = [
        'RandomForest',
        'XGBoost',
        'LightGBM',
        'GradientBoosting',
        'CatBoost',
        'LinearRegression'
    ]
    
    # Configurações de treinamento
    KFOLD_SPLITS = 5
    TEST_SIZE = 0.2
    
    # Limites geográficos (Beijing, China)
    LAT_MIN = 39.0
    LAT_MAX = 41.0
    LON_MIN = 115.0
    LON_MAX = 117.0
    
    # Limites para outliers
    MAX_JUMP_KM = 50.0
    MAX_SPEED_KMH = 200.0
    
    @property
    def ROOT_DIR(self):
        # Tenta primeiro pelo arquivo de configuração
        root = Path(__file__).parent.parent
        
        # Se não encontrou pasta data, usa diretório atual
        if not (root / 'data').exists():
            root = Path.cwd()
            # Se ainda não encontrou, tenta subir na hierarquia
            if not (root / 'data').exists():
                root = root.parent
        
        return root.resolve()
    
    @property
    def DATA_DIR(self):
        data_dir = os.getenv('DATA_DIR', 'data')
        if not data_dir:
            data_dir = 'data'
        
        path = Path(data_dir)
        if not path.is_absolute():
            path = self.ROOT_DIR / data_dir
        
        # Fallback: usa diretório atual se não encontrar
        if not path.exists():
            path = Path.cwd() / 'data'
        
        return path.resolve()
    
    @property
    def MODELS_DIR(self):
        models_dir = self.ROOT_DIR / 'models'
        models_dir.mkdir(exist_ok=True, parents=True)
        return models_dir
    
    @property
    def MODELS_SAVED_DIR(self):
        saved_dir = self.MODELS_DIR / 'saved'
        saved_dir.mkdir(exist_ok=True, parents=True)
        return saved_dir
    
    @property
    def SUBMISSIONS_DIR(self):
        submissions_dir = self.ROOT_DIR / 'submissions'
        submissions_dir.mkdir(exist_ok=True, parents=True)
        return submissions_dir
    
    @property
    def LOGS_DIR(self):
        logs_dir = self.ROOT_DIR / 'logs'
        logs_dir.mkdir(exist_ok=True, parents=True)
        return logs_dir
    
    @property
    def TRAIN_DATA_PATH(self):
        return self.DATA_DIR / 'train.csv'
    
    @property
    def TEST_DATA_PATH(self):
        return self.DATA_DIR / 'test.csv'
    
    @property
    def TRAIN_DATA_PATH_LOCAL(self):
        """Caminho local para dados de treino (aceita diferentes nomes)"""
        possible_names = ['train.csv', 'train_data.csv', 'treino.csv']
        for name in possible_names:
            path = self.DATA_DIR / name
            if path.exists():
                return path
        return self.TRAIN_DATA_PATH
    
    @property
    def TEST_DATA_PATH_LOCAL(self):
        """Caminho local para dados de teste"""
        possible_names = ['test.csv', 'test_data.csv', 'teste.csv']
        for name in possible_names:
            path = self.DATA_DIR / name
            if path.exists():
                return path
        return self.TEST_DATA_PATH
    
    def load_csv(self, filename):
        """Carrega um CSV do diretório de dados"""
        path = self.DATA_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")
        return pd.read_csv(path)
    
    def get_train_path(self, custom_path=None):
        """Obtém caminho dos dados de treino"""
        if custom_path:
            return Path(custom_path)
        # Usar caminho direto do DATA_DIR
        return self.DATA_DIR / 'train.csv'
    
    def get_test_path(self, custom_path=None):
        """Obtém caminho dos dados de teste"""
        if custom_path:
            return Path(custom_path)
        # Usar caminho direto do DATA_DIR
        return self.DATA_DIR / 'test.csv'
    
    def __init__(self):
        """Inicializa criando diretórios"""
        self._create_directories()
    
    def _create_directories(self):
        """Cria diretórios necessários"""
        directories = [
            self.DATA_DIR,
            self.MODELS_DIR,
            self.MODELS_SAVED_DIR,
            self.SUBMISSIONS_DIR,
            self.LOGS_DIR
        ]
        
        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)

# Instância global
config = Config()