# TODO.md - Projeto Trajectory Prediction

## Visão Geral do Projeto

Transformar o projeto em uma **aplicação independente** com:
- ✅ Interface Streamlit completa
- ✅ Upload de novos dados (ambos formatos)
- ✅ Re-treino via interface
- ✅ Visualização Folium integrada
- ✅ Análise de erros
- ✅ Sem dependência do Kaggle

---

## ✅ Fases Concluídas

### Fase 1: Kaggle Independente ⚡
- [x] Removido código de download Kaggle
- [x] Usa apenas arquivos locais (train.csv, test.csv)
- [x] Configurações simplificadas

### Fase 2: Engenharia de Dados ⚡
- [x] Script EDA (`scripts/eda.py`)
- [x] Tratamento de outliers
- [x] Suporte a upload (formatos lista e discreto)

### Fase 3: Streamlit Completa 🗺️
- [x] app.py com 5 páginas completas
- [x] Home, Mapa, Treinamento, Previsão, Análise

### Fase 4: Pipeline ML 🤖
- [x] Persistence module (`models/persistence.py`)
- [x] serializers joblib

### Fase 5: Limpeza e Refatoração 🧹
- [x] Removidos scripts não utilizados
- [x] Removidos dados antigos (.zip)
- [x] Limpos relatórios grandes
- [x] Limpos __pycache__
- [x] Atualizado .env.example
- [x] Atualizado .gitignore
- [x] Limpo dependências pyproject.toml
- [x] **Corrigido**: config/settings.py para detectar data/ corretamente
- [x] **Etapa 5.1**: README.md atualizado para modo local
  - Removidas dependências Kaggle CLI
  - Adicionada seção de formatos de dados
  - Atualizados comandos de uso (Streamlit + CLI)
  - Estrutura de diretórios simplificada
- [x] **Etapa 5.2**: CHANGELOG.md criado
  - Versões semânticas documentadas
  - Mudanças organizadas por tipo (Added, Changed, Fixed, Removed)

### Fase 6: Kaggle Dataset ☁️
- [x] Dataset criado no Kaggle: muitomalakoi/trajectory-prediction-beijing
- [x] Adicionado download automático via kagglehub
- [x] Adicionado kagglehub às dependências
- [x] data/loader.py atualizado com download automático
  - Baixa do Kaggle se dados não existirem localmente
  - Funciona com Streamlit Cloud
- [x] **Etapa 6.1**: Streamlit Cloud compatível
  - Download automático funcionando
  - catboost removido (incompatível com Python 3.14)
  - Mapa simplificado (apenas início/fim da trajetória)
  - Predições limitadas a 10 no toggle
  - st.components.v1.html → st.iframe
  - Valores default reduzir: Train 5, Test 3
  - Correção de sintaxe try/except

---

## 📋 RESUMO - Problemas do Mapa no Streamlit Cloud

### Problemas Identificados e Corrigidos:

| # | Problema | Causa | Solução Aplicada |
|---|---------|-------|-----------------|
| 1 | Dados não encontrados | Falta download | ensure_data_exists() antes load_data() |
| 2 | catboost não instala | Sem wheel Python 3.14 | Removido catboost |
| 3 | Mapa travava | many pontos por PolyLine | Apenas início e fim |
| 4 | Toggle predições crasha | 3000+ marcadores | Limitado a 10 |
| 5 | st.components deprecated | API antiga | Trocado para st.iframe |
| 6 | Timeout no startup | many processamentos | Botão para geração manual (revertido) |
| 7 | Syntax error (182) | try/except mal indentado | Corrigido |

### Arquivos Alterados:
- `evaluation/visualization.py` - Visualização do mapa
- `data/loader.py` - Download Kaggle
- `config/settings.py` - Configurações
- `app.py` - Interface Streamlit

---

## ⚠️ REINICIALIZAÇÃO DO COMPUTADOR

**Status:** Usuário irá reiniciar o PC

**Ao retornar:**
1. Rebuild app no Streamlit Cloud
2. Testar todas as abas:
   - 🏠 Início (dados carregam)
   - 🗺️ Mapa (com e sem toggle)
   - 🤖 Treinamento
   - 🔮 Previsão
   - 📊 Análise

**Próximos passos se algo falhar:**
- Verificar logs do Streamlit Cloud
- Ajustar valores default se necessário

---

## Entrada do Projeto

| Comando | Descrição |
|---------|------------|
| `streamlit run app.py` | Interface Web Streamlit |
| `python main.py` | Pipeline CLI (treinamento) |
| `python scripts/eda.py` | Análise exploratória |

### 5.2 Changelog
- [ ] Documentar todas as mudanças
- [ ] Versões semânticas

---

## Dependências (pyproject.toml)

### Manter
```toml
folium = "^0.20.0"
geopy = "^2.4.1"
xgboost = ">=1.7.0"
lightgbm = ">=3.3.0"
catboost = ">=1.0.0"
scikit-learn = ">=1.3.0"
pandas = ">=2.0.0"
numpy = ">=1.24.0"
```

### Adicionar
```toml
streamlit = "^1.28.0"
joblib = ">=1.3.0"  # para salvar modelos
```

### Opcional (manter compatibilidade)
```toml
# kaggle = ">=1.5.0"  # não obrigatório, apenas para quem quiser usar
```

---

## Estrutura de Arquivos Final

```
trajectory-prediction/
├── app.py                    # Streamlit app principal
├── main.py                   # Pipeline CLI (manter)
├── config/
│   └── settings.py           # Configurações locais
├── data/
│   ├── loader.py            # Carregador local
│   ├── train.csv           # Dados treino
│   └── test.csv            # Dados teste
├── features/
│   ├── cleaning.py         # Limpeza de dados
│   ├── engineering.py       # Feature engineering
│   └── outlier_detection.py
├── models/
│   ├── factory.py         # Factory de modelos
│   ├── trainer.py          # Treinamento
│   └── saved/            # Modelos salvos
├── pages/                  # Páginas Streamlit
│   ├── home.py
│   ├── mapa.py
│   ├── treinamento.py
│   ├── previsao.py
│   └── analise.py
├── scripts/
│   └── eda.py            # Análise exploratória
├── requirements.txt        # ou pyproject.toml
├── README.md
└── .env                  # Configurações locais
```

---

## Histórico de Alterações

### v0.2.0 (2025-05-07)
- Transformado em aplicação independente do Kaggle
- Adicionada interface Streamlit completa
- Suporte a upload de novos dados (formatos lista e discreto)
- Visualização Folium integrada
- Análise de erros implementada
- **Corrigido**: Indentação em config/settings.py
- **Corrigido**: data/__init__.py (removida dependência de downloader.py)
- **Corrigido**: get_train_path/get_test_path usam DATA_DIR diretamente
- **Corrigido**: pyproject.toml adicionado package-mode=false
- **Corrigido**: Instaladas dependências faltantes (catboost, folium, optuna, geopy)
- **Removido**: models/predictors.py (arquivo vazio, não usado)
- **NOVO**: training/pipeline.py - funções reutilizáveis para treino e predição
- **NOVO**: app.py agora faz treino e predição integrado (sem main.py necessário)
- **REMOVIDO**: tools/ directory completo (não usado)
- **REMOVIDO**: seaborn e geopy (dependências não usadas)
- **CORRIGIDO**: optuna versão sem restrição (<5.0.0)
- **CORRIGIDO**: app.py - treinamento usa train_data para targets (numpy array fix)
- **CORRIGIDO**: training/pipeline.py -.columns em numpy array (hasattr check)
- **CORRIGIDO**: app.py previsao try/except para drop columns
- **CORRIGIDO**: pipeline.py save_model - metadata limpo (no RandomForestRegressor JSON)
- **NOVO**: app.py - seletor de modelos, ensemble, seed (via Streamlit)
- **NOVO**: training/pipeline.py - suporta selected_models, seed, use_ensemble
- **NOVO**: model_factory.py - seed customizável no __init__
- **NOVO**: main.py CLI --model 'best'/específico --seed
- **NOVO**: app.py - seed input usa config.SEED como default mas permite override

- **Deploy Streamlit Cloud
- Pronto para deploy no Streamlit Cloud gratuito
- Modelos salvos em models/saved/ (temporários até reinicialização)
- Predições agora salvas em submissions/ para análise
- **CORRIGIDO**: app.py - X_test .values em numpy array (hasattr check)
- **CORRIGIDO**: app.py - SubmissionGenerator chamadacomo instância
- **CORRIGIDO**: app.py - sub_path.name (string não tem .name)
- **NOVO**: app.py - Mapa com descrição (Beijing, China) + legenda
- **CORRIGIDO**: app.py - Legenda incluía ponto verde (predição)
- **NOVO**: evaluation/visualization.py - Legenda integrada no mapa Folium
- **CORRIGIDO**: visualization.py - Legenda usa pontos (círculos) para marcadores
- **CORRIGIDO**: visualization.py - Legenda completa com todas as cores do mapa
- **CORRIGIDO**: visualization.py/app.py - toggle "Mostrar Previsões" agora funciona
- **NOVO**: config/beijing_bounds.py - limites geográficos de Beijing
- **NOVO**: pipeline.py - clamp de predições para Beijing (evita coordenadas fora)
- **NOVO**: outlier_detection.py - limites de Beijing (em vez de globais)
- **MELHORADO**: outlier detection com limites menores para Beijing

### v0.1.0 (2024-12-30)
- Versão original para competição Kaggle
- Pipeline completo com múltiples modelos
- Validação cruzada
- Geração de submissão