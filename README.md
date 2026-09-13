tech_challenge_03
==============================

project 3 fiap

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


# Predição e Inteligência Analítica para Alfabetização no Brasil

## 1. Contexto do problema
A alfabetização infantil é um importante indicador do desenvolvimento educacional e social. Entretanto, a análise de resultados passados, isoladamente, não permite antecipar situações de risco.

O projeto busca utilizar Ciência de Dados e Machine Learning para transformar dados educacionais, territoriais e socioeconômicos em informações que possam apoiar a identificação antecipada de municípios com maior risco de não atingir suas metas de alfabetização.

O Tech Challenge propõe justamente o desenvolvimento de um modelo supervisionado capaz de prever se um aluno será considerado alfabetizado ou não alfabetizado.

## 2. Objetivo
Desenvolver um modelo de classificação capaz de prever a condição de alfabetização dos alunos e utilizar essas previsões para:

identificar padrões associados à alfabetização;
identificar municípios com maior risco educacional;
comparar previsões com as metas municipais;
apoiar a priorização de ações educacionais.

O objetivo não é substituir a decisão dos gestores, mas fornecer uma ferramenta analítica para apoiar a tomada de decisão.

## 3. Dados utilizados
oram utilizadas informações provenientes da camada Gold, construída na Fase 2 do projeto.

A base analítica reúne informações relacionadas a:

resultados de alfabetização;
municípios;
unidades federativas;
características territoriais;
indicadores socioeconômicos;
indicadores educacionais;
metas de alfabetização;
informações históricas de desempenho.

O desafio prevê a utilização desses diferentes grupos de informações na construção da base analítica.

A variável-alvo utilizada no modelo foi:

IN_ALFABETIZADO = (NOTA_LP > 749).astype(int)

Assim:

1 → aluno alfabetizado;
0 → aluno não alfabetizado.

## 4. Preparação dos dados
### 4.1 Tratamento
### 4.2 Engenharia de atributos
Foram criadas variáveis históricas para representar o desempenho anterior dos territórios.

Entre elas:

TAXA_ALF_MUNICIPIO_HIST
TAXA_ALF_UF_HIST
MEDIA_LP_MUNICIPIO_HIST
MEDIA_LP_UF_HIST
FREQ_UF_HIST

Essas variáveis utilizam informações de períodos anteriores, evitando que o resultado do próprio ano seja utilizado como informação de entrada do modelo.

### 4.3 Prevenção de Data Leakage
Foram excluídas variáveis que representavam informações do próprio período de previsão e poderiam revelar o resultado que o modelo deveria prever.

Também foi adotada uma separação temporal:

2024 → treinamento

2025 → teste

Essa estratégia busca reproduzir um cenário mais próximo da utilização real do modelo.

O tratamento de data leakage e a separação adequada entre treino, validação e teste são requisitos explícitos do desafio.

### 4.4 Separação temporal

## 5. Modelagem
### 5.1 Random Forest
### 5.2 XGBoost
### 5.3 LightGBM

## 6. Comparação dos modelos

## 7. Modelo selecionado

## 8. Interpretabilidade
### 8.1 Feature Importance
### 8.2 SHAP

## 9. Análise de risco e metas

## 10. Principais insights

## 11. Limitações

## 12. Aplicação em políticas públicas

## 13. Evoluções futuras

## 14. Conclusão