# -*- coding: utf-8 -*-
"""
Trajectory Prediction - Streamlit App

Aplicação web para visualização e predição de trajetórias GPS.

用法:
    streamlit run app.py

Para desenvolvimento:
    streamlit run app.py --dev
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time

# Configuração de página
st.set_page_config(
    page_title="Trajectory Prediction",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Adicionar diretório raiz ao path
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))


# ============== DataLoader Singleton ==============

@st.cache_data
def load_all_data_cached():
    """Carrega todos os dados uma vez"""
    try:
        from data.loader import DataLoader
        loader = DataLoader()
        return loader.load_data(use_sample_if_missing=True)
    except Exception:
        return None, None


# ============== Funções Auxiliares ==============

def load_train_data():
    """Carrega dados de treino em cache"""
    train, _ = load_all_data_cached()
    return train

def load_test_data():
    """Carrega dados de teste em cache"""
    _, test = load_all_data_cached()
    return test

def load_all_data():
    """Carrega todos os dados com fallback"""
    return load_all_data_cached()


def load_config():
    """Carrega configurações"""
    try:
        from config.settings import config
        return config
    except ImportError:
        class FallbackConfig:
            ROOT_DIR = Path(__file__).parent
            DATA_DIR = ROOT_DIR / "data"
            MODELS_DIR = ROOT_DIR / "models"
            SUBMISSIONS_DIR = ROOT_DIR / "submissions"
            SEED = 42
            def get_train_path(self):
                return self.DATA_DIR / "train.csv"
            def get_test_path(self):
                return self.DATA_DIR / "test.csv"
        return FallbackConfig()


# ============== Página Inicial ==============

def show_home(config):
    """Página inicial com estatísticas"""
    st.title("🏠 Dashboard - Trajectory Prediction")
    st.markdown("---")
    
    train_data, test_data = load_all_data()
    
    if train_data is None:
        st.warning("⚠️ Dados não encontrados.")
        st.markdown("""
        ### Como adicionar dados:
        1. Coloque `train.csv` e `test.csv` na pasta `data/`
        2. Ou faça upload abaixo
        """)
        
        # Upload
        st.subheader("📤 Upload de Dados")
        col1, col2 = st.columns(2)
        
        with col1:
            train_file = st.file_uploader(
                "Train CSV",
                type=["csv"],
                key="train_upload"
            )
            if train_file is not None:
                df = pd.read_csv(train_file)
                st.success(f"✅ {len(df)} linhas")
                dest = config.DATA_DIR / "train.csv"
                df.to_csv(dest, index=False)
                st.success(f"Salvo em {dest.name}")
                st.rerun()
        
        with col2:
            test_file = st.file_uploader(
                "Test CSV",
                type=["csv"],
                key="test_upload"
            )
            if test_file is not None:
                df = pd.read_csv(test_file)
                st.success(f"✅ {len(df)} linhas")
                dest = config.DATA_DIR / "test.csv"
                df.to_csv(dest, index=False)
                st.success(f"Salvo em {dest.name}")
                st.rerun()
        
        return
    
    # Métricas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Train", f"{len(train_data):,}")
    with col2:
        st.metric("Test", f"{len(test_data):,}")
    with col3:
        st.metric("Features", len(train_data.columns))
    with col4:
        has_target = "dest_lat" in train_data.columns
        st.metric("Com Alvo", "✅ Sim" if has_target else "❌ Não")
    
    st.markdown("---")
    
    # Estrutura
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Estrutura - Train")
        st.dataframe(
            pd.DataFrame({
                "Coluna": train_data.columns,
                "Tipo": [str(t)[:8] for t in train_data.dtypes]
            }),
            use_container_width=True,
            hide_index=True
        )
    
    with col2:
        st.subheader("📊 Estrutura - Test")
        st.dataframe(
            pd.DataFrame({
                "Coluna": test_data.columns,
                "Tipo": [str(t)[:8] for t in test_data.dtypes]
            }),
            use_container_width=True,
            hide_index=True
        )
    
    st.markdown("---")
    
    # Upload adicional
    st.subheader("📤 Upload Adicional")
    new_file = st.file_uploader("CSV alternativo", type=["csv"])
    if new_file:
        df = pd.read_csv(new_file)
        st.dataframe(df.head())


def show_mapa(config):
    """Página do mapa"""
    st.title("🗺️ Mapa de Trajetórias - Beijing, China")
    
    train_data, test_data = load_all_data()
    
    if train_data is None:
        st.warning("Carregue os dados primeiro na página inicial.")
        return
    
    # Descrição contextual
    st.markdown("""
    ### Sobre os Dados
    - **Localização:** Beijing, China
    - **Tipo:** Trajetórias de transporte (ônibus/táxi)
    - **Coordenadas:** ~39.9°N, ~116.3°E (Centro de Beijing)
    - **Dataset:** Trajetórias de ônibus/táxi em Beijing para predição de destino
    """)
    
    # Controles
    max_train = min(100, len(train_data))
    max_test = min(50, len(test_data))
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_train = st.slider("Qtd Treino", 10, max_train, 50)
    with col2:
        n_test = st.slider("Qtd Teste", 10, max_test, 20)
    with col3:
        show_predictions = st.toggle("Mostrar Previsões")
    
    # Usar avaliação existente
    try:
        from evaluation.visualization import TrajectoryVisualizer
        from data.loader import DataLoader
        
        # Preparar visualização
        with st.spinner("Gerando mapa..."):
            viz = TrajectoryVisualizer()
            html_path = ROOT_DIR / "reports" / "streamlit_map.html"
            html_path.parent.mkdir(exist_ok=True)
            
            # Gerar mapa
            viz.create_trajectory_map(
                max_trajectories=n_train + n_test,
                show_predictions=show_predictions,
                save_path=html_path
            )
        
        # Mostrar
        if html_path.exists():
            with open(html_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=600)
        
    except Exception as e:
        st.error(f"Erro ao gerar mapa: {e}")
        st.info("Execute: python evaluation/visualization.py --full")
        # Legenda continua abaixo


def show_treinamento(config):
    """Página de treinamento"""
    st.title("🤖 Treinamento do Modelo")
    
    train_data, test_data = load_all_data()
    
    if train_data is None or test_data is None:
        st.warning("⚠️ Carregue os dados primeiro na aba Início")
        return
    
    st.markdown(f"**Train:** {len(train_data)} linhas | **Test:** {len(test_data)} linhas")
    
    # Opções de treinamento
    st.markdown("### Opções de Treinamento")
    
    col1, col2, col3 = st.columns(3)
    
    available_models = ['XGBoost', 'LightGBM', 'CatBoost', 'RandomForest', 
                      'GradientBoosting', 'BaggedGB', 'HistGradientBoosting']
    
    with col1:
        st.markdown("**Modelos**")
        selected_models = st.multiselect(
            "Selecione modelos",
            available_models,
            default=['BaggedGB']
        )
        seed = st.number_input("Seed", value=config.SEED, min_value=0, max_value=9999, step=1)
    
    with col2:
        st.markdown("**Configurações**")
        cv_folds = st.selectbox("CV Folds", [3, 5, 10], index=1)
        use_clustering = st.checkbox("Usar Clustering", value=False)
        use_augmentation = st.checkbox("Data Augmentation", value=False)
    
    with col3:
        st.markdown("**Ensemble**")
        use_ensemble = st.checkbox("Usar Ensemble", value=False)
        priority_only = st.checkbox("Apenas Priority", value=False)
        show_cv = st.checkbox("Mostrar CV Results", value=True)
    
    train_button = st.button("🚀 Iniciar Treinamento", type="primary")
    
    if train_button:
        with st.spinner("Treinando modelo..."):
            try:
                from training.pipeline import Pipeline
                
                pipeline = Pipeline(use_clustering=use_clustering, use_augmentation=use_augmentation)
                
                st.info(f"Treinando {len(selected_models)} modelo(s) with {cv_folds}-fold CV...")
                
                # Preparar features
                X_train, X_test, meta = pipeline.prepare_features(train_data, test_data)
                
                # Pegar targets do dados original
                y_train = train_data[['dest_lat', 'dest_lon']].values
                
                # Treinar
                metadata = pipeline.train(
                    X_train, y_train,
                    selected_models=selected_models,
                    seed=seed,
                    use_ensemble=use_ensemble,
                    priority_only=priority_only
                )
                
                st.success(f"Modelo treinado: {metadata['model_name']}")
                
                if metadata.get('best_cv_error'):
                    st.metric("Melhor Erro CV", f"{metadata['best_cv_error']:.4f} km")
                
                # Mostrar ranking se disponível
                if show_cv and metadata.get('cv_results'):
                    st.markdown("### 📊 Resultados CV")
                    cv_df = []
                    for name, res in metadata['cv_results'].items():
                        cv_df.append({
                            'Modelo': name,
                            'Erro (km)': res.get('mean_error', 0)
                        })
                    if cv_df:
                        import pandas as pd
                        cv_df = sorted(cv_df, key=lambda x: x['Erro (km)'])
                        st.dataframe(pd.DataFrame(cv_df), use_container_width=True)

# Salvar modelo
                model_path = pipeline.save_model(metadata['model_name'])
                st.success(f"Modelo salvo: {model_path.name}")
                
                st.info("Agora pode fazer previsoes na aba Previsao!")
                
            except Exception as e:
                st.error(f"Erro: {e}")
    

def show_previsao(config):
    """Página de previsão"""
    st.title("🔮 Previsão de Trajetórias")
    
    st.markdown("### 📤 Upload de dados para previsão")
    
    # Verificar modelos disponíveis
    try:
        from models.persistence import ModelPersistence
        saved_models = ModelPersistence.list_models()
    except ImportError:
        saved_models = []
    
    if saved_models:
        st.success(f"📂 {len(saved_models)} modelo(s) salvo(s)")
        
        # Listar modelos
        for m in saved_models[:5]:
            st.markdown(f"- **{m['name']}** ({m['size_kb']:.1f} KB)")
    else:
        st.warning("⚠️ Nenhum modelo salvo.")
    
    # Upload com exemplo de formato
    with st.expander("📋 Formato esperado"):
        st.markdown("""
        **Formato Lista:**
        ```csv
        trajectory_id,path_lat,path_lon
        001_20230601,"[39.9,40.0,40.1]","[116.3,116.4,116.5]
        ```
        
        **Formato Discreto:**
        ```csv
        trajectory_id,lat,lon,sequence
        001_20230601,39.9,116.3,1
        001_20230601,40.0,116.4,2
        ```
        """)
    
    file = st.file_uploader("CSV com trajetórias", type=["csv"], key="previsao")
    
    if file:
        try:
            df = pd.read_csv(file)
            st.success(f"✅ Carregado: {len(df)} trajetórias")
            
            # Detectar formato
            if 'lat' in df.columns and 'sequence' in df.columns:
                st.info("📝 Formato discreto detectado - convertendo...")
                from data.loader import DataLoader
                df = DataLoader.convert_discrete_to_list_format(df)
                st.success(f"✅ Convertido: {len(df)} trajetórias")
            
            st.dataframe(df.head(10), use_container_width=True)
            
            if saved_models and st.button("🔮 Executar Previsão", type="primary"):
                with st.spinner("Prevendo..."):
                    try:
                        # Carregar modelo
                        model, metadata = ModelPersistence.get_latest_model()
                        st.success(f"✅ Modelo carregado: {metadata.get('model_name', 'desconhecido')}")
                        
                        # Preparar features usando pipeline
                        from training.pipeline import prepare_features_only
                        
                        # Usar dados de teste carregados
                        test_loader = load_test_data()
                        if test_loader is None:
                            st.error("❌ Dados de teste não disponíveis")
                            return
                        
                        # Obter train_data para preparação de features (necessário para scaler)
                        train_loader = load_train_data()
                        
                        st.info("📊 Preparando features...")
                        X_train, X_test = prepare_features_only(train_loader, test_loader)
                        
                        # Remover targets do X_train se existirem
                        try:
                            X_train = X_train.drop(columns=['dest_lat', 'dest_lon'])
                        except AttributeError:
                            pass  # X_train é numpy array, não precisa drop
                        
                        # Fazer predição
                        from training.pipeline import predict_only
                        X_test_arr = X_test.values if hasattr(X_test, 'values') else X_test
                        predictions = predict_only(model, X_test_arr)
                        
                        # Criar DataFrame de resultados
                        result_df = pd.DataFrame({
                            'trajectory_id': test_loader['trajectory_id'].values[:len(predictions)],
                            'predicted_dest_lat': predictions[:, 0],
                            'predicted_dest_lon': predictions[:, 1]
                        })
                        
                        st.success(f"✅ {len(predictions)} predições geradas!")
                        
                        # Mostrar tabela
                        st.dataframe(result_df.head(20), use_container_width=True)
                        
                        # Download
                        csv = result_df.to_csv(index=False)
                        st.download_button(
                            "📥 Baixar Predições",
                            csv,
                            "predictions.csv",
                            "text/csv",
                            key="download_pred"
                        )
                        
                        # Salvar em submissions/ para análise
                        from submission.generator import SubmissionGenerator
                        generator = SubmissionGenerator()
                        sub_path = generator.generate_submission(
                            test_ids=test_loader['trajectory_id'].values[:len(predictions)],
                            predictions=predictions,
                            model_name=metadata.get('model_name', 'streamlit'),
                            test_df=test_loader
                        )
                        st.success(f"Salvo: {sub_path}")
                        
                    except Exception as e:
                        st.error(f"Erro: {e}")
            
            # Treino integrado disponível na aba Treinamento
                
            st.markdown("---")
            
        except Exception as e:
            st.error(f"Erro ao carregar: {e}")


def show_analise(config):
    """Página de análise"""
    st.title("📊 Análise de Erros")
    
    train_data, test_data = load_all_data()
    
    if train_data is None:
        st.warning("⚠️ Carregue os dados primeiro.")
        return
    
    # Check if has targets
    has_targets = 'dest_lat' in train_data.columns and 'dest_lon' in train_data.columns
    
    st.markdown("### 🎯 Estatísticas do Dataset")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Train", len(train_data))
    with col2:
        st.metric("Total Test", len(test_data))
    with col3:
        st.metric("Tem Targets", "✅ Sim" if has_targets else "❌ Não")
    with col4:
        # Unique users
        n_users = train_data['user_id'].nunique() if 'user_id' in train_data.columns else 0
        st.metric("Usuários", n_users)
    
    st.markdown("---")
    
    # Show destination distribution
    if has_targets:
        st.markdown("### 🌍 Distribuição dos Destinos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Latitude**")
            lat_data = train_data['dest_lat'].dropna()
            st.markdown(f"Min: {lat_data.min():.4f}")
            st.markdown(f"Max: {lat_data.max():.4f}")
            st.markdown(f"Mean: {lat_data.mean():.4f}")
        
        with col2:
            st.markdown("**Longitude**")
            lon_data = train_data['dest_lon'].dropna()
            st.markdown(f"Min: {lon_data.min():.4f}")
            st.markdown(f"Max: {lon_data.max():.4f}")
            st.markdown(f"Mean: {lon_data.mean():.4f}")
    
    st.markdown("---")
    
    # Load submissions for error analysis
    st.markdown("### 📈 Análise de Predições")
    
    submissions_dir = config.SUBMISSIONS_DIR
    sub_files = list(submissions_dir.glob("submission_*.csv"))
    
    if sub_files:
        st.success(f"📂 {len(sub_files)} arquivo(s) de predição encontrado(s)")
        
        # Load latest submission
        latest = max(sub_files, key=lambda x: x.stat().st_mtime)
        df_sub = pd.read_csv(latest)
        
        st.markdown(f"**Última predição:** {latest.name}")
        st.markdown(f"- Previsões: {len(df_sub)}")
        st.dataframe(df_sub.head(), use_container_width=True)
        
        # Merge with test for comparison
        if has_targets:
            try:
                merged = test_data.merge(df_sub, on='trajectory_id')
                
                # Calculate errors (if real targets exist in test - unlikely but possible)
                # This would need real targets in test, which normally doesn't exist
                st.info("ℹ️ Para calcular erros, o arquivo de teste precisa ter destinations.")
            except Exception as e:
                st.info(f"ℹ️ {e}")
    else:
        st.info("Execute previsões na aba Previsão para gerar resultados.")
    
    st.markdown("---")
    st.markdown("""
    ### 📋 Relatórios Disponíveis
    - Logs: `logs/`
    - Modelos: `models/`
    - Predições: `submissions/`
    - Relatórios: `reports/`
    """)


def show_sidebar(config):
    """Menu lateral"""
    st.sidebar.title("🗺️ Navegação")
    
    pages = [
        ("home", "🏠 Início"),
        ("mapa", "🗺️ Mapa"),
        ("treinamento", "🤖 Treinamento"),
        ("previsao", "🔮 Previsão"),
        ("analise", "📊 Análise")
    ]
    
    selected = st.sidebar.radio("Navegar para:", [p[1] for p in pages])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Configurações")
    st.sidebar.markdown(f"**Dados:** `{config.DATA_DIR}`")
    st.sidebar.markdown(f"**Seed:** (config settings)")
    
    return selected


def main():
    """Função principal"""
    config = load_config()
    
    page = show_sidebar(config)
    
    if "Início" in page:
        show_home(config)
    elif "Mapa" in page:
        show_mapa(config)
    elif "Treinamento" in page:
        show_treinamento(config)
    elif "Previsão" in page:
        show_previsao(config)
    elif "Análise" in page:
        show_analise(config)


if __name__ == "__main__":
    main()