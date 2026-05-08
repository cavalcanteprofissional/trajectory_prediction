# Projeto de Predição de Trajetórias

Pipeline de Machine Learning para predição de coordenadas de destino (latitude e longitude) com base em dados de trajetórias GPS.

## 📋 Índice

- [Sobre o Projeto](#sobre-o-projeto)
- [Características Principais](#características-principais)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Requisitos](#requisitos)
- [Instalação](#instalação)
- [Configuração](#configuração)
- [Uso](#uso)
- [Pipeline](#pipeline)
- [Modelos](#modelos)
- [Features](#features)
- [Validação e Métricas](#validação-e-métricas)
- [Estrutura de Diretórios](#estrutura-de-diretórios)
- [Troubleshooting](#troubleshooting)

## 🎯 Sobre o Projeto

Este projeto implementa um pipeline completo de Machine Learning para predição de trajetórias GPS, utilizando múltiplos algoritmos de aprendizado supervisionado para prever coordenadas geográficas finais (destino) com base em dados históricos de trajetórias.

### Objetivo

Prever as coordenadas de destino (`dest_lat`, `dest_lon`) de trajetórias com base em:
- Dados de caminho percorrido (`path_lat`, `path_lon`) - apenas o prefixo inicial da trajetória
- Features extraídas da trajetória (espaciais, temporais e geométricas)
- Múltiplos modelos de regressão com validação cruzada robusta

### Métrica de Avaliação

O projeto utiliza a **Distância Haversine** (em quilômetros) como métrica principal, calculando a distância geodésica entre as coordenadas preditas e reais na superfície da Terra.

## 📍 Sobre os Dados

Este projeto utiliza dados de trajetórias GPS da região de **Beijing, China**, consistindo em trajetórias de transporte (ônibus/táxi) coletadas ao longo do tempo.

### Características dos Dados

- **Localização:** Beijing, China
- **Tipo:** Trajetórias de transporte (ônibus/táxi)
- **Centro de Beijing:** ~39.9°N, ~116.3°E
- **Limites geográficos:** lat [39.4, 40.5]N, lon [116.0, 117.0]E

### Arquivos de Dados

| Arquivo | Descrição | Colunas |
|---------|-----------|---------|
| `data/train.csv` | Dados de treino | trajectory_id, path_lat, path_lon, dest_lat, dest_lon |
| `data/test.csv` | Dados de teste | trajectory_id, path_lat, path_lon |

Cada arquivo contém listas de coordenadas GPS representando o caminho percorrido pela trajetória.

### Limites Geográficos

O projeto inclui validação e clamp de predições para garantir que todas as coordenadas estejam dentro dos limites de Beijing, evitando predições fora da região de interesse.

## ✨ Características Principais

- ✅ **Pipeline Completo**: Do carregamento de dados à geração de predições
- ✅ **Múltiplos Modelos**: Suporte a 16+ algoritmos de ML
- ✅ **Validação Cruzada Robusta**: 5-fold cross-validation com métrica Haversine
- ✅ **Detecção de Outliers**: Sistema inteligente de detecção e remoção de outliers
- ✅ **Clusterização de Dados**: Agrupamento dos dados após limpeza para focar no maior cluster
- ✅ **Engenharia de Features Avançada**: 30+ features extraídas das trajetórias
- ✅ **Ensemble de Modelos**: Suporte a Voting Regressor e Bagging
- ✅ **Separação de Dados**: Garantia de que train.csv e test.csv são usados corretamente
- ✅ **Interface Streamlit**: Visualização interactive com mapas Folium
- ✅ **Logging Completo**: Sistema de logs detalhado

## 📁 Estrutura do Projeto

```
trajectory-prediction/
├── app.py                    # Streamlit app principal
├── main.py                   # Pipeline CLI
├── config/                  # Configurações
│   ├── __init__.py
│   ├── settings.py
│   └── beijing_bounds.py
├── data/                    # Dados e processamento
│   ├── __init__.py
│   ├── loader.py
│   ├── train.csv
│   └── test.csv
├── features/                # Engenharia de features
│   ├── __init__.py
│   ├── engineering.py
│   ├── outlier_detection.py
│   ├── cleaning.py
│   └── clustering.py
├── models/                  # Modelos de ML
│   ├── __init__.py
│   ├── factory.py
│   ├── trainer.py
│   └── saved/             # Modelos salvos
├── training/               # Pipeline de treinamento
│   ├── __init__.py
│   └── pipeline.py
├── evaluation/            # Avaliação
│   ├── __init__.py
│   ├── metrics.py
│   └── visualization.py
├── submission/            # Geração de submissões
│   ├── __init__.py
│   └── generator.py
├── pages/                 # Páginas Streamlit
│   ├── home.py
│   ├── mapa.py
│   ├── treinamento.py
│   ├── previsao.py
│   └── analise.py
├── scripts/               # Scripts auxiliares
│   └── eda.py
├── utils/                 # Utilitários
├── logs/                  # Arquivos de log
├── submissions/           # Arquivos de submissão
├── .env.local               # Configurações locais
├── requirements.txt
├── pyproject.toml
└── README.md
```

## 🔧 Requisitos

- **Python**: >= 3.10
- **Git**: Para controle de versão

### Dependências Principais

- `scikit-learn` >= 1.3.0
- `pandas` >= 2.0.0
- `numpy` >= 1.24.0
- `xgboost` >= 1.7.0
- `lightgbm` >= 3.3.0
- `catboost` >= 1.0.0
- `streamlit` >= 1.28.0
- `folium` >= 0.20.0
- `joblib` >= 1.3.0

## 📦 Instalação

### 1. Clone o repositório

```bash
git clone <url-do-repositorio>
cd trajectory-prediction
```

### 2. Crie um ambiente virtual (recomendado)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. Instale as dependências

**Opção A: Usando pip**
```bash
pip install -r requirements.txt
```

**Opção B: Usando Poetry** (recomendado)
```bash
poetry install
poetry shell
```

### 4. Pré-requisitos

- Python >= 3.10
- API Token do Kaggle (para download automático de dados)

### 5. Dados do Projeto

O projeto utiliza dados de trajetórias GPS de Beijing, China. O download é automático na primeira execução.

#### Configuração do Kaggle

**Opção A: Via .env.local (Recomendado)**
```bash
cp .env.example .env.local
```
Edite o `.env.local` com suas credenciais:
```env
KAGGLE_USERNAME=seu_usuario
KAGGLE_KEY=seu_token
```
Obtenha em: https://www.kaggle.com/account

**Opção B: Via Streamlit Cloud**
Adicione nos **Secrets** do app:
- `KAGGLE_USERNAME`
- `KAGGLE_KEY`

**Opção C: Download Manual**
```bash
kaggle datasets download -d muitomalakoi/trajectory-prediction-beijing -p data/ --unzip
```

## ⚙️ Configuração

### Variáveis de Ambiente

O projeto usa um arquivo `.env.local` para armazenar configurações. Copie o arquivo de exemplo e configure se necessário:

```bash
cp .env.example .env.local
```

```env
# Seed para reprodutibilidade
SEED=42

# Diretório de dados (opcional)
DATA_DIR=data

# Kaggle API (obrigatório para download automático)
KAGGLE_USERNAME=seu_usuario
KAGGLE_KEY=seu_token
```

### Formatos de Dados

O projeto aceita dois formatos de arquivo CSV:

**Formato Lista** (recomendado):
```csv
trajectory_id,path_lat,path_lon
1,"[[39.9,116.4],[39.91,116.41],[39.92,116.42]]","[[116.4,116.41],[116.42,116.43]]"
```

**Formato Disco**:
```csv
trajectory_id,path_lat,path_lon
1,39.9_39.91_39.92,116.4_116.41_116.42
```

**Nota**: Para dados de treino, adicione colunas `dest_lat` e `dest_lon`.

## 🚀 Uso

### Execução com Streamlit (Recomendado)

```bash
streamlit run app.py
```

Acesse em: http://localhost:8501

### Execução via CLI

```bash
python main.py
```

### Opções do CLI

```bash
python main.py [OPTIONS]

Options:
  --model MODEL      Modelo específico (default: best)
  --seed SEED       Seed para reprodutibilidade (default: 42)
  --ensemble       Usa ensemble de modelos
  -h, --help       Mostra ajuda
```

## 🎨 Interface Streamlit

O projeto inclui uma interface web Streamlit completa para visualização e interação com os dados. Acesse via `streamlit run app.py`.

### Abas do Menu Lateral

| Aba | Ícone | Descrição |
|-----|------|----------|
| 🏠 Início | Início | Carregamento de dados e overview do dataset |
| 🗺️ Mapa | Mapa | Visualização interativa Folium das trajetórias no mapa |
| 🤖 Treinamento | Treinamento | Treino de modelos com opções configuráveis |
| 🔮 Previsão | Previsão | Geração de predições com modelo treinado |
| 📊 Análise | Análise | Métricas de avaliação e análise de erros |

### Detalhes das Abas

- **Início**: Carrega os dados do diretório `data/` e exibe estatísticas básicas
- **Mapa**: Visualiza as trajetórias em um mapa interativo de Beijing com legendas
- **Treinamento**: Permite configurar modelos, seed, ensemble e executar treinamento
- **Previsão**: Gera predições usando o modelo treinado e salva em CSV
- **Análise**: Exibe métricas de erro e análise visual dos resultados

---

## 🔄 Pipeline

O pipeline executa as seguintes etapas em ordem:

### 1. Carregamento de Dados
- Verifica se os dados existem localmente em `data/`
- Carrega `train.csv` e `test.csv`
- Valida integridade dos dados
- Suporta formatos lista e disco

### 2. Detecção de Outliers
- **Outliers Geográficos**: Coordenadas inválidas
- **Outliers de Trajetória**: Saltos grandes e velocidades impossíveis
- **Outliers de Target**: Destinos com coordenadas inválidas
- **Proteções**: Limite máximo de remoção para evitar perda excessiva de dados

### 2b. Clusterização de Dados (opcional)
- Agrupamento dos dados limpos usando K-means
- Extração de features de clusterização (posição, distância, geometria)
- Seleção automática do número ótimo de clusters via silhouette score
- Filtragem para usar apenas o maior cluster no treinamento

### 3. Engenharia de Features
- Extração de 30+ features das trajetórias
- Features básicas, de distância, geométricas e direcionais
- Normalização e tratamento de valores faltantes

### 4. Preparação dos Dados
- Separação de features e target
- Normalização com StandardScaler
- **IMPORTANTE**: `train.csv` usado para treino/validação, `test.csv` apenas para predições

### 5. Treinamento com Validação Cruzada
- **5-fold cross-validation** no conjunto de treino
- Métrica: Distância Haversine média (km)
- Testa múltiplos modelos em paralelo
- Seleciona o melhor modelo baseado na métrica

### 6. Treinamento do Modelo Final
- Treina o melhor modelo em todos os dados de treino

### 7. Predição
- Gera predições para `test.csv`
- Valida formato e ranges das predições
- Clamp para região geográfica (Beijing)

### 8. Geração de Arquivo
- Cria arquivo CSV no formato de submissão
- Salva em `submissions/` com timestamp

## 🤖 Modelos

O projeto suporta 16+ algoritmos de Machine Learning:

### Modelos Prioritários

- **RandomForest**: Ensemble de árvores de decisão
- **XGBoost**: Gradient boosting otimizado
- **LightGBM**: Gradient boosting rápido
- **GradientBoosting**: Boosting tradicional (com otimização Optuna)
- **HistGradientBoosting**: Versão otimizada do scikit-learn

### Outros Modelos Disponíveis

- CatBoost
- Extra Trees
- Ridge Regression
- Lasso Regression
- Elastic Net
- Bayesian Ridge
- K-Nearest Neighbors (KNN)
- Support Vector Regression (SVR)
- Multi-Layer Perceptron (MLP)
- AdaBoost
- Bagged Gradient Boosting

### Ensemble

- **Ensemble Avançado**: Combinação de GradientBoosting otimizado + RandomForest
- **BaggedGB**: Bagging com GradientBoosting base

## 📊 Features

O projeto extrai **30+ features** das trajetórias:

### Features Básicas
- `start_lat`, `start_lon`: Posição inicial
- `end_lat`, `end_lon`: Posição final do prefixo
- `mean_lat`, `mean_lon`: Médias de latitude e longitude
- `std_lat`, `std_lon`: Desvios padrão

### Features de Distância
- `total_distance`: Distância total percorrida (metros)
- `mean_distance`: Distância média entre pontos
- `straight_distance`: Distância em linha reta
- `straightness`: Razão entre distância reta e total

### Features Geométricas
- `lat_range`, `lon_range`: Amplitude das coordenadas
- `area_bbox`: Área do bounding box
- `aspect_ratio`: Razão aspecto
- `centroid_lat`, `centroid_lon`: Centroide da trajetória

### Features Direcionais
- `bearing`: Direção do início ao fim (graus)
- `bearing_sin`, `bearing_cos`: Versões trigonométricas
- `direction_variance`: Variabilidade de direção

## 📈 Validação e Métricas

### Validação Cruzada

- **Método**: K-Fold Cross-Validation
- **Folds**: 5
- **Métrica**: Distância Haversine média (km)
- **Dados**: Apenas `train.csv`

### Métrica Principal: Distância Haversine

Calcula a distância geodésica entre dois pontos na Terra usando a fórmula:

```
d = 2R · arcsin(√(sin²(Δφ/2) + cos(φ₁)cos(φ₂)sin²(Δλ/2)))
```

Onde R = 6371 km (raio médio da Terra).

### Separação de Dados

**CRÍTICO**: Garantia de separação correta:
- ✅ `train.csv`: Treino e validação cruzada
- ✅ `test.csv`: Apenas predições finais
- ❌ `test.csv` NUNCA usado em treino/validação

## 📂 Estrutura de Diretórios

- **`data/`**: Dados brutos (`train.csv`, `test.csv`)
- **`logs/`**: Arquivos de log (`pipeline.log`)
- **`submissions/`**: Arquivos de submissão gerados
- **`reports/`**: Relatórios (`pipeline_report.txt`, resultados Optuna)
- **`scripts/`**: Scripts de otimização (Optuna)
- **`models/`**: Implementações de modelos
- **`features/`**: Engenharia de features
- **`training/`**: Lógica de treinamento
- **`evaluation/`**: Métricas e visualizações

## 🐛 Troubleshooting

### Dados não encontrados
- Verifique se `data/train.csv` e `data/test.csv` existem
- Use a interface Streamlit para fazer upload de novos dados

### Dependências não encontradas
```bash
pip install --upgrade -r requirements.txt
```

### Erro de memória
- Reduza número de modelos testados
- Processe dados em lotes menores

### Erro no Ensemble
- Verifique se modelos base suportam multi-output

---

## 📚 Referências

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Haversine Formula](https://en.wikipedia.org/wiki/Haversine_formula)

*Última atualização: Maio 2025*