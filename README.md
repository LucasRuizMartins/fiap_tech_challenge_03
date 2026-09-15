tech_challenge_03
==============================

[Video de apresentação](https://docs.google.com/videos/u/0/d/1iJ1wNroL6RKNjG-Z4AqEarHvlPy2PBTSFOYR_Ic26-Y/play?usp=chrome_extension_sharing) <br>
[PDF da apresentação](https://docs.google.com/presentation/d/1lVsukVksM9MWq8dIc1N_nfXhg2AwJ0dvEajGFXvfRxw/edit?usp=sharing) <br> [Link 2](https://docs.google.com/presentation/d/1lVsukVksM9MWq8dIc1N_nfXhg2AwJ0dvEajGFXvfRxw/edit?slide=id.g3fb576cc7a4_36_0#slide=id.g3fb576cc7a4_36_0)


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

# Tech Challenge – Fase 3
## Predição e Inteligência Analítica para Alfabetização no Brasil

## 1. Contexto

A alfabetização infantil é um importante indicador do desenvolvimento educacional e social. Entretanto, analisar apenas os resultados já observados não é suficiente para apoiar decisões estratégicas.

Neste projeto, desenvolvido no contexto do Tech Challenge – Fase 3, utilizamos Ciência de Dados e Machine Learning para transformar dados educacionais, territoriais e socioeconômicos em informações capazes de apoiar a identificação de riscos e a análise de cenários relacionados à alfabetização.

O desafio propõe o desenvolvimento de um modelo supervisionado capaz de prever se um aluno será considerado alfabetizado ou não alfabetizado, utilizando informações provenientes da camada Gold desenvolvida na Fase 2.

---

## 2. Objetivo

Desenvolver uma solução de Machine Learning capaz de:

- prever a probabilidade de alfabetização dos alunos;
- identificar fatores associados às previsões do modelo;
- identificar municípios com maior risco educacional;
- analisar o desempenho previsto em relação às metas municipais;
- explorar cenários que possam apoiar a tomada de decisão em políticas públicas educacionais.

Além da performance preditiva, buscamos produzir uma análise interpretável e aplicável ao contexto educacional brasileiro.

---

## 3. Dados utilizados

A base analítica foi construída a partir da camada Gold desenvolvida na Fase 2, integrando informações educacionais, territoriais e socioeconômicas.

Foram utilizados dados referentes aos anos de 2023, 2024 e 2025.

Entre as principais informações utilizadas estão:

### Dados educacionais
- Indicador de alfabetização;
- frequência escolar;
- absenteísmo;
- dependência administrativa;
- características das escolas;
- histórico de desempenho educacional.

### Dados territoriais
- Unidade Federativa (UF);
- município;
- região;
- características de porte municipal e escolar.

### Dados socioeconômicos
- rendimento médio mensal;
- renda per capita de beneficiários de programas sociais;
- indicadores de frequência escolar.

### Metas educacionais
Foram utilizadas as metas municipais de alfabetização para comparação entre o desempenho previsto e o objetivo estabelecido para 2025.

---

## 4. Variável alvo

A variável alvo foi construída a partir da nota de Língua Portuguesa:


IN_ALFABETIZADO = (NOTA_LP > 749).astype(int)

|      Nota LP | Classificação    |
| -----------: | ---------------- |
|  750 ou mais | Alfabetizado     |
| 749 ou menos | Não alfabetizado |

O problema foi tratado como uma classificação binária:

1 → alfabetizado
0 → não alfabetizado

## 5. Análise exploratória

A análise exploratória foi utilizada para compreender:

- distribuição das variáveis;
- proporção de alunos alfabetizados;
- diferenças entre anos;
- relações entre desempenho histórico e desempenho futuro;
- comportamento das variáveis socioeconômicas;
- padrões relacionados a frequência e absenteísmo;
- diferenças entre municípios e regiões.

A análise exploratória também orientou a seleção e construção das variáveis utilizadas na modelagem.


## 6. Engenharia de atributos

Foram construídas variáveis relacionadas a:

- porte das escolas;
- porte dos municípios;
- frequência escolar;
- absenteísmo histórico;
- características socioeconômicas;
- região;
- rede de ensino;
- histórico municipal de alfabetização;
- histórico estadual de alfabetização.

**Um dos principais cuidados foi evitar data leakage.**

As variáveis históricas utilizadas para prever 2025 foram calculadas utilizando informações disponíveis anteriormente, principalmente os dados de 2024.
Por exemplo, a variável:

TAXA_ALF_MUNICIPIO_HIST

representa o desempenho histórico do município e **não utiliza o resultado de alfabetização do próprio ano que está sendo previsto.**

## 7. Estratégia de treinamento e validação

Para aproximar o experimento de uma situação real de previsão temporal, foi utilizada a seguinte divisão:

| Ano  | Utilização                            |
| ---- | ------------------------------------- |
| 2023 | Construção das informações históricas |
| 2024 | Treinamento                           |
| 2025 | Teste final                           |


Essa estratégia evita que informações do futuro sejam utilizadas durante o treinamento.
O conjunto de 2024 também foi dividido internamente em treino e validação para apoiar a seleção e ajuste dos modelos.
O conjunto de 2025 permaneceu reservado para a avaliação final.

## 8. Modelos avaliados

Foram avaliados três algoritmos de classificação:

- Random Forest;
- XGBoost;
- LightGBM.

O objetivo foi comparar diferentes abordagens de aprendizado supervisionado e selecionar o modelo com melhor desempenho no cenário temporal definido.

## 9. Resultados dos modelos

Os resultados obtidos no conjunto de teste de 2025 foram:

| Modelo        |   Acurácia |   Precisão |     Recall |         F1 |    ROC-AUC |
| ------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| Random Forest | **60,14%** | **64,90%** | **69,71%** | **67,22%** | **63,01%** |
| XGBoost       |     59,17% |     64,33% |     68,14% |     66,18% |     61,53% |
| LightGBM      |     59,27% |     64,51% |     67,84% |     66,14% |     61,36% |


O Random Forest apresentou o melhor desempenho geral no experimento temporal, apresentando o maior ROC-AUC, acurácia, recall e F1-score entre os modelos avaliados.
Por esse motivo, o Random Forest foi utilizado como modelo de referência para as análises seguintes.

## 10. Interpretabilidade

A análise de importância das variáveis mostrou que o histórico de alfabetização possui papel relevante nas previsões.
As duas principais variáveis foram:

TAXA_ALF_MUNICIPIO_HIST
TAXA_ALF_UF_HIST

Juntas, essas duas variáveis representaram aproximadamente: 51,1% da importância das variáveis do modelo.
Esse resultado indica uma forte persistência temporal do desempenho educacional. Ou seja, municípios e estados que apresentam determinado padrão histórico tendem a manter parte desse comportamento nas previsões futuras.
É importante destacar que essa análise representa associação e capacidade preditiva do modelo, não causalidade.

## 11. Desempenho histórico e previsão

Foi observada uma forte associação entre a taxa histórica de alfabetização dos municípios e a taxa prevista para 2025.
A correlação observada foi aproximadamente: 0,89
Esse resultado indica que o modelo captura a persistência do desempenho educacional ao longo do tempo.
Ao mesmo tempo, as previsões não reproduzem simplesmente o resultado histórico, pois o modelo também considera outras características educacionais, territoriais e socioeconômicas.

## 12. Identificação de municípios em risco

As probabilidades individuais geradas pelo modelo foram agregadas por município.
Para cada município foram calculados indicadores como:

- número de alunos avaliados;
- probabilidade média de alfabetização;
- probabilidade média de não alfabetização;
- taxa prevista de alfabetização;
- distância em relação à meta de 2025.

A partir desses indicadores, foram criadas duas perspectivas de priorização.

### 12.1. Municípios com maior déficit em relação à meta

Essa análise prioriza municípios cuja taxa prevista está mais distante da meta estabelecida para 2025.
Exemplos identificados:

| Município           | UF | Previsto | Meta 2025 |     Déficit |
| ------------------- | -- | -------: | --------: | ----------: |
| Araricá             | RS |   33,77% |    76,98% | -43,21 p.p. |
| Canela              | RS |   36,45% |    79,06% | -42,62 p.p. |
| São Lourenço do Sul | RS |   37,65% |    79,35% | -41,70 p.p. |
| Constantina         | RS |   38,46% |    80,00% | -41,54 p.p. |
| Serafina Corrêa     | RS |   39,09% |    80,00% | -40,91 p.p. |


Essa perspectiva permite identificar municípios que apresentam maior distância proporcional em relação ao objetivo estabelecido.

### 12.2. Municípios com maior impacto potencial

Também foi analisado o número estimado de alunos que poderiam permanecer não alfabetizados, considerando as probabilidades previstas pelo modelo.
Essa abordagem prioriza municípios com maior número de alunos e, consequentemente, maior impacto potencial.
Entre os municípios identificados estão:

- Rio de Janeiro;
- Manaus;
- Salvador;
- Guarulhos;
- Curitiba;
- Porto Alegre;
- Belo Horizonte;
- Belém.

As duas análises possuem objetivos diferentes:
**déficit em relação à meta** → intensidade do problema;
**impacto potencial** → escala do problema.

Por isso, não devem ser tratadas como um único ranking.

## 13. Simulação de cenário: redução do absenteísmo

Foi realizada uma análise de cenário utilizando as variáveis históricas de absenteísmo.
Mantendo as demais características constantes, simulamos diferentes níveis de redução do absenteísmo:

Redução do absenteísmo	Alfabetização prevista	Ganho estimado
| Redução do absenteísmo | Alfabetização prevista | Ganho estimado |
| ---------------------: | ---------------------: | -------------: |
|                     0% |                 53,68% |              — |
|                    10% |                 54,12% |     +0,44 p.p. |
|                    20% |                 54,65% |     +0,98 p.p. |
|                    30% |                 55,10% |     +1,42 p.p. |
|                    40% |                 55,59% |     +1,92 p.p. |


O cenário de redução de 40% apresentou uma taxa prevista de alfabetização de 55,59%, contra 53,68% no cenário de referência.
Isso representa um ganho estimado de: +1,92 ponto percentual.
**Essa análise deve ser interpretada como uma simulação contrafactual do modelo, e não como uma estimativa de efeito causal.**

## 14. Principais insights
### 1. O histórico educacional é um forte sinal preditivo
As taxas históricas de alfabetização municipal e estadual representam aproximadamente 51,1% da importância das variáveis utilizadas pelo modelo.
Isso evidencia a persistência do desempenho educacional ao longo do tempo.

### 2. O risco educacional é heterogêneo
Os municípios apresentam diferentes níveis de desempenho previsto e diferentes distâncias em relação às metas.
Isso reforça a necessidade de políticas públicas direcionadas às realidades locais.

### 3. Escala e gravidade são dimensões diferentes
Um município pequeno pode apresentar um déficit percentual muito elevado, enquanto um município grande pode concentrar um número muito maior de alunos em situação de risco.
Por isso, a priorização pode considerar tanto a intensidade quanto o impacto potencial.

### 4. O absenteísmo aparece como variável relevante
A análise de cenários mostrou que reduções simuladas no absenteísmo estão associadas a aumentos na taxa de alfabetização prevista pelo modelo.
No cenário de redução de 40%, o ganho estimado foi de 1,92 p.p.
Esse resultado deve ser entendido como cenário preditivo, não como relação causal.

## 15. Limitações
O projeto apresenta algumas limitações importantes.

### Performance preditiva
O modelo apresentou ROC-AUC de aproximadamente 0,63, indicando capacidade preditiva moderada.
Portanto, suas previsões devem ser utilizadas como instrumento de apoio à decisão e não como diagnóstico definitivo.

### Dados predominantemente contextuais
Grande parte das variáveis utilizadas possui características municipais ou estaduais.
Consequentemente, alunos de um mesmo município podem receber probabilidades semelhantes.
Isso limita a interpretação da previsão como um risco individual preciso.

### Ausência de causalidade
As relações identificadas pelo modelo representam associações preditivas.
Não é possível afirmar, apenas a partir deste modelo, que uma determinada variável seja a causa direta da alfabetização ou não alfabetização.

### Simulação de cenários
Os cenários de redução do absenteísmo são simulações contrafactuais.
Eles indicam como o modelo reage a alterações nas variáveis, mas não representam estimativas causais de impacto de uma política pública.

### Dados disponíveis
A qualidade e abrangência das previsões estão condicionadas às variáveis disponíveis na camada Gold e aos períodos históricos utilizados.

## 16. Aplicação em políticas públicas
A solução pode ser utilizada como uma ferramenta de apoio à gestão educacional.
Entre as possíveis aplicações estão:

- identificação antecipada de municípios em situação de maior risco;
- priorização de recursos e programas educacionais;
- acompanhamento de municípios em relação às metas;
- identificação de regiões que necessitam de maior atenção;
- monitoramento de indicadores históricos;
- simulação de cenários para apoiar decisões estratégicas.

A proposta não é substituir a avaliação dos gestores, mas fornecer evidências quantitativas para apoiar a definição de prioridades.

## 17. Possíveis evoluções
Como próximos passos, a solução pode evoluir por meio de:

- inclusão de novas variáveis educacionais e socioeconômicas;
- integração com dados externos, como IBGE, Censo Escolar, FUNDEB e PNAD;
- utilização de SHAP Values para explicações individuais e municipais;
- modelos específicos para diferentes regiões ou perfis de municípios;
- inclusão de mais anos históricos;
- desenvolvimento de modelos hierárquicos ou temporais;
- melhoria da calibração das probabilidades;
- criação de dashboards para acompanhamento dos municípios;
- avaliação de cenários de políticas públicas com métodos causais;
- monitoramento contínuo das previsões após a disponibilização de novos dados.


## 18. Reprodutibilidade
O projeto foi desenvolvido utilizando Python e bibliotecas de Ciência de Dados e Machine Learning.
As principais bibliotecas utilizadas incluem:

- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Random Forest
- XGBoost
- LightGBM

As etapas de preparação, modelagem, avaliação e análise foram organizadas em notebooks e scripts para permitir a reprodução dos experimentos.

-----------------------

## 20. Conclusão
O projeto demonstra como dados públicos podem ser transformados em inteligência analítica para apoiar decisões relacionadas à alfabetização infantil.
O modelo desenvolvido apresentou capacidade preditiva moderada e permitiu identificar padrões históricos, municípios potencialmente vulneráveis e diferentes níveis de distância em relação às metas educacionais.
Mais do que buscar apenas uma alta métrica de classificação, a solução procura transformar as previsões em informações úteis para a tomada de decisão.
A combinação entre previsão, interpretabilidade, priorização territorial e simulação de cenários permite uma visão mais ampla do problema e cria possibilidades para o uso de Ciência de Dados no planejamento de políticas públicas educacionais.

