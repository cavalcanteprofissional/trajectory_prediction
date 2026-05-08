# Changelog

Todos as notáveis mudanças deste projeto serão documentadas neste arquivo.

O formato é baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/),
e este projeto adere ao Semantic Versioning (https://semver.org/lang/pt-BR/).

## [0.2.0] - 2025-05-07

### Added
- Transformado em aplicação independente do Kaggle
- Interface Streamlit completa com 5 páginas
- Suporte a upload de novos dados (formatos lista e discreto)
- Visualização Folium integrada com legenda
- Análise de erros implementada
- Limites geográficos de Beijing (config/beijing_bounds.py)
- Clamp de predições para Beijing

### Changed
- Removida dependência do Kaggle CLI
- Pipeline simplificado (main.py sem --submit)
- Estrutura de diretórios limpa

### Fixed
- Indentação em config/settings.py
- data/__init__.py (removida dependência de downloader.py)
- get_train_path/get_test_path usam DATA_DIR diretamente
- pyproject.toml adicionado package-mode=false
- X_test .values em numpy array (hasattr check)
- SubmissionGenerator chamada como instância
- Legenda do mapa incluída

### Removed
- Downloader Kaggle
- tools/ directory completo
- seaborn e geopy (dependências não usadas)
- models/predictors.py (arquivo vazio)

---

## [0.1.0] - 2024-12-30

### Added
- Versão original para competição Kaggle
- Pipeline completo com múltiplos modelos
- Validação cruzada 5-fold
- Geração de submissão automática