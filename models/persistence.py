# models/persistence.py
"""
Módulo para persistência de modelos ML.
Salva e carrega modelos treinados em disco.
"""
import joblib
import json
from pathlib import Path
import sys
from datetime import datetime

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config.settings import config


class ModelPersistence:
    """Gerencia persistência de modelos"""
    
    @staticmethod
    def save_model(model, model_name, metrics=None, metadata=None):
        """
        Salva um modelo treinado.
        
        Args:
            model: Modelo sklearn/XGBoost/LightGBM
            model_name: Nome do modelo
            metrics: Métricas do modelo (dict)
            metadata: Metadados extras (dict)
        
        Returns:
            Path: Caminho do modelo salvo
        """
        models_dir = config.MODELS_SAVED_DIR
        models_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{timestamp}.joblib"
        filepath = models_dir / filename
        
        # Salvar modelo
        joblib.dump(model, filepath)
        
        # Salvar metadados
        meta = {
            "model_name": model_name,
            "filename": filename,
            "timestamp": timestamp,
            "metrics": metrics or {},
            "metadata": metadata or {}
        }
        
        meta_path = filepath.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        
        return filepath
    
    @staticmethod
    def load_model(filepath):
        """
        Carrega um modelo salvo.
        
        Args:
            filepath: Caminho do arquivo .joblib
        
        Returns:
            tuple: (model, metadata)
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Modelo não encontrado: {filepath}")
        
        model = joblib.load(filepath)
        
        meta_path = filepath.with_suffix(".json")
        if meta_path.exists():
            with open(meta_path, "r") as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        return model, metadata
    
    @staticmethod
    def list_models():
        """
        Lista todos os modelos salvos.
        
        Returns:
            list: Lista de modelos com metadados
        """
        models_dir = config.MODELS_SAVED_DIR
        
        if not models_dir.exists():
            return []
        
        models = []
        
        for filepath in models_dir.glob("*.joblib"):
            try:
                meta_path = filepath.with_suffix(".json")
                if meta_path.exists():
                    with open(meta_path, "r") as f:
                        metadata = json.load(f)
                else:
                    metadata = {}
                
                models.append({
                    "filepath": str(filepath),
                    "name": filepath.stem,
                    "size_kb": filepath.stat().st_size / 1024,
                    "modified": datetime.fromtimestamp(filepath.stat().st_mtime),
                    "metadata": metadata
                })
            except Exception as e:
                print(f"Erro ao carregar {filepath}: {e}")
        
        return sorted(models, key=lambda x: x["modified"], reverse=True)
    
    @staticmethod
    def get_latest_model():
        """
        Pega o modelo mais recente.
        
        Returns:
            tuple: (model, metadata) ou (None, None)
        """
        models = ModelPersistence.list_models()
        
        if not models:
            return None, None
        
        latest = models[0]
        return ModelPersistence.load_model(latest["filepath"])
    
    @staticmethod
    def delete_model(filepath):
        """Deleta um modelo"""
        filepath = Path(filepath)
        
        if filepath.exists():
            filepath.unlink()
            
            meta_path = filepath.with_suffix(".json")
            if meta_path.exists():
                meta_path.unlink()
            
            return True
        
        return False


def save_trained_model(model, model_name, cv_score=None):
    """
    Função de conveniencia para salvar modelo.
    
    Args:
        model: Modelo treinado
        model_name: Nome do modelo
        cv_score: Score de cross-validation (opcional)
    
    Returns:
        Path: Caminho do modelo salvo
    """
    metrics = {}
    if cv_score is not None:
        metrics["cv_score"] = cv_score
    
    return ModelPersistence.save_model(model, model_name, metrics=metrics)


def load_trained_model(model_name=None):
    """
    Carrega um modelo treinado.
    
    Args:
        model_name: Nome específico ou None para mais recente
    
    Returns:
        tuple: (model, metadata) ou (None, None)
    """
    if model_name:
        # Buscar por nome
        models_dir = config.MODELS_SAVED_DIR
        filepath = models_dir / f"{model_name}.joblib"
        
        if filepath.exists():
            return ModelPersistence.load_model(filepath)
    
    # Carregar mais recente
    return ModelPersistence.get_latest_model()