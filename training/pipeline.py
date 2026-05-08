# training/pipeline.py
"""
Pipeline de Treinamento e Predição Reutilizável

Fornece funções unificadas para:
- Extração de features
- Treinamento de modelo
- Predição
- Pipeline completo

Usado por: main.py (CLI) e app.py (Streamlit)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional, Any

ROOT_DIR = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(ROOT_DIR))

from config.settings import config
from config.beijing_bounds import clamp_to_beijing_array, BEIJING_BOUNDS
from features.engineering import FeatureEngineer
from features.clustering import DataClusterer
from features.cleaning import clean_train_test
from models import ModelFactory
from models.persistence import ModelPersistence
from training import ModelTrainer


class Pipeline:
    """Pipeline completo de ML para predição de trajetórias"""
    
    def __init__(self, use_clustering: bool = True, use_augmentation: bool = False):
        self.use_clustering = use_clustering
        self.use_augmentation = use_augmentation
        self.feature_engineer = FeatureEngineer()
        self.trainer = ModelTrainer()
        self.model_factory = None
        self.model = None
        self.metadata = {}
        self.scaler = None
        self.feature_cols = None
        self.use_local_coords = False
        self.refs_lat = None
        self.refs_lon = None
        self.start_lats = None
        self.start_lons = None
    
    def prepare_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """Prepara features para treino e predição.
        
        Args:
            train_df: DataFrame de treino (com dest_lat, dest_lon)
            test_df: DataFrame de teste
            
        Returns:
            tuple: (X_train, X_test, metadata)
        """
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info("Preparando features...")
        
        # Clustering opcional
        if self.use_clustering:
            logger.info("Aplicando clustering...")
            clusterer = DataClusterer()
            cluster_labels = clusterer.fit_predict(train_df)
            train_df = clusterer.get_largest_cluster_data(train_df, cluster_labels)
            logger.info(f"Dados após clustering: {len(train_df)}")
        
        # Augmentação opcional
        if self.use_augmentation:
            try:
                from features.augmentation import augment_dataframe
                logger.info("Aplicando augmentation...")
                train_df = augment_dataframe(train_df, methods=['jitter'], p=0.3, seed=42)
            except Exception as e:
                logger.warning(f"Augmentation falhou: {e}")
        
        # Extrair features
        logger.info("Extraindo features...")
        X_train = self.feature_engineer.extract_all_features(train_df)
        X_test = self.feature_engineer.extract_all_features(test_df)
        
        # Adicionar targets se existirem
        if 'dest_lat' in train_df.columns and 'dest_lon' in train_df.columns:
            X_train['dest_lat'] = train_df['dest_lat'].values
            X_train['dest_lon'] = train_df['dest_lon'].values
        
        # Preparar para treinamento
        logger.info("Preparando features para modelo...")
        prepared = self.feature_engineer.prepare_features_for_training(
            X_train, X_test,
            target_cols=['dest_lat', 'dest_lon'],
            use_local_target=False,
            scaler_type='robust'
        )
        
        X_train = prepared['X_train']
        X_test = prepared['X_test']
        self.scaler = prepared.get('scaler')
        self.feature_cols = prepared.get('feature_cols', [])
        
        # Armazenar referências para conversão
        if prepared.get('use_local_target'):
            self.use_local_coords = True
            # Verificar se é DataFrame ou array
            if hasattr(X_train, 'columns'):
                if 'start_lat' in X_train.columns:
                    self.refs_lat = X_train['start_lat'].values
                if 'start_lon' in X_train.columns:
                    self.refs_lon = X_train['start_lon'].values
                if 'start_lat' in X_test.columns:
                    self.start_lats = X_test['start_lat'].values
                if 'start_lon' in X_test.columns:
                    self.start_lons = X_test['start_lon'].values
            else:
                # É numpy array - usar feature_cols
                if self.feature_cols and 'start_lat' in self.feature_cols:
                    idx = list(self.feature_cols).index('start_lat')
                    self.refs_lat = X_train[:, idx]
                if self.feature_cols and 'start_lon' in self.feature_cols:
                    idx = list(self.feature_cols).index('start_lon')
                    self.refs_lon = X_train[:, idx]
        
        # Store test trajectory_ids for submission
        self.test_ids = test_df['trajectory_id'].values if 'trajectory_id' in test_df.columns else None
        if hasattr(X_test, 'columns'):
            self.test_starts = X_test[['start_lat', 'start_lon']].values if 'start_lat' in X_test.columns else None
        else:
            # É numpy array
            self.test_starts = None
        
        metadata = {
            'n_train': len(X_train),
            'n_test': len(X_test),
            'n_features': X_train.shape[1] if hasattr(X_train, 'shape') else X_train[0].shape[0],
            'use_clustering': self.use_clustering,
            'use_augmentation': self.use_augmentation,
            'use_local_coords': self.use_local_coords
        }
        
        logger.info(f"Features preparadas: {X_train.shape[1] if hasattr(X_train, 'shape') else 'N/A'} colunas")
        
        return X_train, X_test, metadata
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
         priority_only: bool = True,
         selected_models: list = None,
         seed: int = 42,
         use_ensemble: bool = False) -> Dict:
        """Treina o modelo.
        
        Args:
            X_train: Features de treino
            y_train: Targets de treino (dest_lat, dest_lon)
            priority_only: Usar apenas modelos prioritários
            selected_models: Lista de modelos específicos (None = usar priority_only)
            seed: Seed aleatória
            use_ensemble: Criar ensemble dos modelos
        
        Returns:
            dict: Informações do modelo treinado
        """
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info(f"Treinando modelo (priority_only={priority_only}, seed={seed})...")
        
        # Criar factory com seed
        self.model_factory = ModelFactory(n_samples=len(X_train), seed=seed)
        
        # Criar modelos
        if selected_models and len(selected_models) > 0:
            # Modelos específicos selecionados
            models = {}
            for model_name in selected_models:
                try:
                    models[model_name] = self.model_factory.create_model(
                        model_name, n_features=X_train.shape[1])
                    logger.info(f"Modelo criado: {model_name}")
                except Exception as e:
                    logger.warning(f"Erro ao criar {model_name}: {e}")
            
            # Criar ensemble se solicitado e se há múltiplos modelos
            if use_ensemble and len(models) >= 2:
                try:
                    from sklearn.ensemble import VotingRegressor
                    from sklearn.multioutput import MultiOutputRegressor
                    
                    base_models = [(name, model) for name, model in models.items()]
                    voting = VotingRegressor(estimators=base_models)
                    ensemble = MultiOutputRegressor(voting)
                    ensemble.model_name = 'EnsembleVoting'
                    models['EnsembleVoting'] = ensemble
                    logger.info("Ensemble criado: EnsembleVoting")
                except Exception as e:
                    logger.warning(f"Erro ao criar ensemble: {e}")
        else:
            # Usar lógica existente (priority or all)
            models = self.model_factory.create_all_models(
                priority_only=priority_only,
                include_ensemble=use_ensemble,
                n_features=X_train.shape[1]
            )
        
        if not models:
            raise ValueError("Nenhum modelo foi criado. Verifique os nomes dos modelos.")
        
        logger.info(f"Total de modelos: {len(models)}")
        
        # Validação cruzada
        results = self.trainer.train_all_models(
            X_train, y_train, models, cv_folds=5
        )
        
        # Treinar modelo final
        final_info = self.trainer.train_final_model(X_train, y_train)
        self.model = final_info['model']
        
        logger.info(f"Modelo final: {final_info['model_name']}")
        
        # Armazenar results para display
        self.cv_results = results
        
        self.metadata = {
            'model_name': final_info['model_name'],
            'cv_results': {name: {'mean_error': res.get('mean_error', 0)} for name, res in results.items() if res},
            'best_cv_error': self.trainer.best_model_info.get('mean_error') if self.trainer.best_model_info else None,
            'seed': seed
        }
        
        return self.metadata
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Faz predições.
        
        Args:
            X_test: Features de teste
            
        Returns:
            np.ndarray: Predições (dest_lat, dest_lon)
        """
        import logging
        logger = logging.getLogger(__name__)
        
        if self.model is None:
            raise ValueError("Modelo não treinado. Execute train() primeiro.")
        
        logger.info("Fazendo predições...")
        predictions = self.model.predict(X_test)
        
        # Converter de coordenadas locais se necessário
        if self.use_local_coords and self.start_lats is not None:
            logger.info("Convertendo de coordenadas locais para lat/lon...")
            dest_lats = []
            dest_lons = []
            for i in range(len(predictions)):
                lat, lon = FeatureEngineer.local_xy_to_latlon(
                    self.start_lats[i], self.start_lons[i],
                    predictions[i, 0], predictions[i, 1]
                )
                dest_lats.append(lat)
                dest_lons.append(lon)
            predictions = np.column_stack([dest_lats, dest_lons])
        
        # Aplicar clamp para manter coordenadas dentro de Beijing
        logger.info("Aplicando clamp para Beijing...")
        lat_clamped, lon_clamped = clamp_to_beijing_array(
            predictions[:, 0], predictions[:, 1]
        )
        predictions = np.column_stack([lat_clamped, lon_clamped])
        
        logger.info(f"{len(predictions)} predicoes geradas (clampadas para Beijing: {BEIJING_BOUNDS['lat_min']}-{BEIJING_BOUNDS['lat_max']}N, {BEIJING_BOUNDS['lon_min']}-{BEIJING_BOUNDS['lon_max']}E)")
        return predictions
    
    def save_model(self, model_name: str = "pipeline_model") -> Path:
        """Salva o modelo treinado.
        
        Args:
            model_name: Nome do modelo
            
        Returns:
            Path: Caminho do arquivo salvo
        """
        import logging
        logger = logging.getLogger(__name__)
        
        if self.model is None:
            raise ValueError("Modelo não treinado")
        
        # Limpar metadados - remover objetos não-serializáveis
        clean_metadata = {
            'model_name': self.metadata.get('model_name'),
            'best_cv_error': self.metadata.get('best_cv_error'),
            'n_train': self.metadata.get('n_train'),
            'n_test': self.metadata.get('n_test'),
            'n_features': self.metadata.get('n_features'),
        }
        
        filepath = ModelPersistence.save_model(
            self.model,
            model_name,
            metrics={'best_cv_error': self.metadata.get('best_cv_error')},
            metadata=clean_metadata
        )
        
        logger.info(f"Modelo salvo: {filepath}")
        return filepath
    
    @staticmethod
    def load_model(model_path: Path = None):
        """Carrega modelo do disco.
        
        Args:
            model_path: Caminho específico ou None para último modelo
            
        Returns:
            tuple: (model, metadata)
        """
        if model_path is None:
            return ModelPersistence.get_latest_model()
        return ModelPersistence.load_model(model_path)


# Funções de Convenience (API Simples)


def prepare_features_only(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepara features apenas (sem treino).
    
    Args:
        train_df: DataFrame de treino
        test_df: DataFrame de teste
        
    Returns:
        tuple: (X_train, X_test)
    """
    pipeline = Pipeline(use_clustering=False)
    X_train, X_test, _ = pipeline.prepare_features(train_df, test_df)
    return X_train, X_test


def train_only(train_df: pd.DataFrame, test_df: pd.DataFrame,
              save: bool = True, priority_only: bool = True) -> Dict:
    """Executa apenas treino.
    
    Args:
        train_df: DataFrame de treino
        test_df: DataFrame de teste (usado para FeatureEngineer.prepare_features_for_training)
        save: Salvar modelo
        priority_only: Usar apenas modelos prioritários
        
    Returns:
        dict: Metadados do modelo
    """
    pipeline = Pipeline(use_clustering=False)
    
    X_train, X_test, meta = pipeline.prepare_features(train_df, test_df)
    
    y_train = X_train[['dest_lat', 'dest_lon']].values
    X_train = X_train.drop(columns=['dest_lat', 'dest_lon'], errors='ignore')
    
    metadata = pipeline.train(X_train, y_train, priority_only=priority_only)
    
    if save:
        pipeline.save_model(metadata['model_name'])
    
    return metadata


def predict_only(model, X_test: np.ndarray, start_coords: np.ndarray = None) -> np.ndarray:
    """Executa apenas predição.
    
    Args:
        model: Modelo treinado
        X_test: Features de teste
        start_coords: Coordenadas iniciais (start_lat, start_lon) para conversão
        
    Returns:
        np.ndarray: Predições (dest_lat, dest_lon)
    """
    predictions = model.predict(X_test)
    
    # Aplicar clamp para manter coordenadas dentro de Beijing
    lat_clamped, lon_clamped = clamp_to_beijing_array(
        predictions[:, 0], predictions[:, 1]
    )
    predictions = np.column_stack([lat_clamped, lon_clamped])
    
    return predictions


def run_full_pipeline(train_df: pd.DataFrame, test_df: pd.DataFrame,
                     save_model: bool = True,
                     generate_submission: bool = True,
                     priority_only: bool = True) -> Dict:
    """Executa pipeline completo (treino + predição).
    
    Args:
        train_df: DataFrame de treino
        test_df: DataFrame de teste
        save_model: Salvar modelo treinado
        generate_submission: Gerar arquivo de submissão
        priority_only: Usar apenas modelos prioritários
        
    Returns:
        dict: {
            'predictions': np.ndarray,
            'submission': pd.DataFrame,
            'metadata': dict
        }
    """
    import logging
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("EXECUTANDO PIPELINE COMPLETO")
    logger.info("=" * 50)
    
    pipeline = Pipeline(use_clustering=False)
    
    logger.info("1. Preparando features...")
    X_train, X_test, meta = pipeline.prepare_features(train_df, test_df)
    
    y_train = X_train[['dest_lat', 'dest_lon']].values
    X_train = X_train.drop(columns=['dest_lat', 'dest_lon'], errors='ignore')
    
    logger.info("2. Treinando modelo...")
    metadata = pipeline.train(X_train, y_train, priority_only=priority_only)
    
    if save_model:
        logger.info("3. Salvando modelo...")
        model_path = pipeline.save_model(metadata['model_name'])
    
    logger.info("4. Fazendo predições...")
    predictions = pipeline.predict(X_test)
    
    result = {
        'predictions': predictions,
        'metadata': metadata
    }
    
    if generate_submission:
        logger.info("5. Gerando submissão...")
        from submission.generator import SubmissionGenerator
        
        test_ids = test_df['trajectory_id'].values if 'trajectory_id' in test_df.columns else None
        
        submission = SubmissionGenerator.generate_submission(
            test_ids=test_ids,
            predictions=predictions,
            model_name=metadata['model_name']
        )
        
        result['submission'] = submission
    
    logger.info("Pipeline completo finalizado!")
    return result


# Alias para compatibilidade
create_pipeline = Pipeline
run_training_pipeline = train_only
run_prediction_pipeline = predict_only