import boto3
import dotenv
import os 
import pandas as pd 
from io import BytesIO

from pathlib import Path

# Garante que o .env seja encontrado independente de onde o notebook está rodando
caminho_env = Path(__file__).parent.parent.parent / '.env'
dotenv.load_dotenv(caminho_env)

# Criando a cessão 

def iniciar_cessao_aws():
    ID_CONTA = os.getenv("ID_CONTA")
    AWS_REGION = os.getenv("AWS_REGION")
    AWS_ACCESS_KEY_ID=os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY=os.getenv("AWS_SECRET_ACCESS_KEY")


    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )
    return session



def carregar_parquet_s3(
    s3_client,
    bucket: str,
    nome_tabela: str,
    camada: str = "gold",
    ano: str | int = None,
    subpasta: str = None,
    ler_dicionario: bool = False,
) -> pd.DataFrame:
    """
    Carrega qualquer arquivo Parquet do S3 de forma flexível.
    
    Exemplos de chaves geradas:
    - Com ano:        'gold/ano=historico/dados/TS_ALUNO.parquet'
    - Dados externos: 'gold/dados_externos/fact_escola.parquet'
    - Fato 2026:      'gold/fato/ano=2026/dados/FATO_ALFABETIZACAO.parquet'
    """
    nome_arquivo = f"dicionario_{nome_tabela}.parquet" if ler_dicionario else f"{nome_tabela}.parquet"
    
    # Montagem dinâmica da chave no S3
    if subpasta:
        # Se você passar uma subpasta direta (ex: 'dados_externos' ou 'fato/ano=2026/dados')
        chave = f"{camada}/{subpasta}/{nome_arquivo}"
    elif ano is not None:
        # Padrão com partição de ano
        pasta_interna = "dicionario" if ler_dicionario else "dados"
        chave = f"{camada}/ano={ano}/{pasta_interna}/{nome_arquivo}"
    else:
        # Padrão direto na camada
        chave = f"{camada}/{nome_arquivo}"

    print(f"Lendo: s3://{bucket}/{chave}")

    try:
        obj = s3_client.get_object(Bucket=bucket, Key=chave)
        return pd.read_parquet(BytesIO(obj["Body"].read()))
    except s3_client.exceptions.NoSuchKey:
        raise FileNotFoundError(f"Arquivo não encontrado:\ns3://{bucket}/{chave}")
    except Exception as e:
        raise RuntimeError(f"Erro ao ler {chave} do bucket {bucket}.\n{e}")



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.base import clone

def projetar_alfabetizacao(
    df_metas: pd.DataFrame,
    modelo=None,
    nome_uf: str = "Brasil",
    anos_treino: list = [2019, 2023],
    meta_2030: float = 80.0,
    plotar: bool = True,
    figsize: tuple = (12, 6)
):
    """
    Projeta taxas de alfabetização com base em um modelo de regressão e compara com as metas oficiais do MEC.

    Parâmetros:
    -----------
    df_metas : pd.DataFrame
        DataFrame com colunas de histórico e metas (ex: resultado_metas).
    modelo : estimador scikit-learn (opcional)
        Instância do modelo (ex: LinearRegression(), Ridge(), SVR(), Pipeline(...)).
        Se None, usa LinearRegression(). Se não estiver treinado, a função treina automaticamente.
    nome_uf : str, default 'Brasil'
        Localidade a ser filtrada na coluna 'NOME_UF'.
    anos_treino : list, default [2019, 2023]
        Anos históricos usados para ajustar o modelo de tendência.
    meta_2030 : float, default 80.0
        Meta fixada para o ano de 2030.
    plotar : bool, default True
        Se True, gera e exibe o gráfico executivo.
    figsize : tuple, default (12, 6)
        Dimensões da figura do gráfico.

    Retorna:
    --------
    dict contendo:
        - 'df_projecao': DataFrame com os anos, projeção, metas e gaps calculados.
        - 'gap_2030': Gap em pontos percentuais para 2030.
        - 'modelo': Instância do modelo ajustado.
        - 'historico': Dicionário com anos e taxas observadas.
    """
    # 1. Filtro da localidade
    dados_uf = df_metas[df_metas["NOME_UF"].str.upper() == nome_uf.upper()]
    if dados_uf.empty:
        raise ValueError(f"Localidade '{nome_uf}' não encontrada na coluna 'NOME_UF'.")
    linha = dados_uf.iloc[0]

    # 2. Extração do Histórico Real
    mapa_historico = {
        2019: float(linha["SAEB_2019"]),
        2021: float(linha["SAEB_2021"]),
        2023: float(linha["PC_ALUNO_ALFABETIZADO"])
    }
    anos_hist = np.array(list(mapa_historico.keys()))
    taxas_hist = np.array(list(mapa_historico.values()))

    # 3. Metas Oficiais MEC (2024 a 2030)
    anos_futuros = np.array([2024, 2025, 2026, 2027, 2028, 2029, 2030]).reshape(-1, 1)
    metas_mec = np.array([
        float(linha["META_FINAL_2024"]),
        float(linha["META_FINAL_2025"]),
        float(linha["META_FINAL_2026"]),
        float(linha["META_FINAL_2027"]),
        float(linha["META_FINAL_2028"]),
        float(linha["META_FINAL_2029"]),
        float(meta_2030)
    ])

    # 4. Configuração e Ajuste do Modelo
    if modelo is None:
        modelo = LinearRegression()
    else:
        # Clona para não alterar o objeto original externamente
        try:
            modelo = clone(modelo)
        except Exception:
            pass

    # Treina o modelo se ele ainda não foi ajustado
    X_treino = np.array(anos_treino).reshape(-1, 1)
    y_treino = np.array([mapa_historico[ano] for ano in anos_treino])
    
    # Se já tiver sido treinado previamente, mantemos; caso contrário, damos fit
    if not hasattr(modelo, "predict") or hasattr(modelo, "fit"):
        modelo.fit(X_treino, y_treino)

    # 5. Previsão
    projecao_tendencia = modelo.predict(anos_futuros).flatten()
    gap_2030 = metas_mec[-1] - projecao_tendencia[-1]

    # DataFrame com os resultados tabulados
    df_projecao = pd.DataFrame({
        "Ano": anos_futuros.flatten(),
        "Meta_MEC": metas_mec,
        "Projecao": projecao_tendencia,
        "Gap_pp": metas_mec - projecao_tendencia
    })

    # 6. Plotagem (Opcional)
    if plotar:
        nome_modelo = modelo.__class__.__name__
        plt.figure(figsize=figsize)

        # Histórico Real
        plt.plot(anos_hist, taxas_hist, marker='o', color='#1f77b4', lw=2.8, markersize=8, label="Histórico Real (SAEB / INEP)")
        for x, y in zip(anos_hist, taxas_hist):
            plt.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontweight='bold', color='#1f77b4')

        # Meta MEC
        plt.plot(anos_futuros.flatten(), metas_mec, marker='s', color='#2ca02c', linestyle='--', lw=2.2, markersize=7, label="Meta Oficial MEC")
        plt.annotate(f"Meta 2030: {metas_mec[-1]:.0f}%", (2030, metas_mec[-1]), textcoords="offset points", xytext=(0, 10), ha='center', fontweight='bold', color='#2ca02c')

        # Projeção do Modelo
        plt.plot(anos_futuros.flatten(), projecao_tendencia, marker='^', color='#d62728', linestyle=':', lw=2.2, markersize=7, label=f"Projeção ({nome_modelo})")
        plt.annotate(f"Previsto: {projecao_tendencia[-1]:.1f}%", (2030, projecao_tendencia[-1]), textcoords="offset points", xytext=(0, -18), ha='center', fontweight='bold', color='#d62728')

        # Linha e área de Gap
        plt.axhline(meta_2030, color='darkgreen', linestyle='-', alpha=0.3, lw=1.5)
        plt.fill_between(anos_futuros.flatten(), projecao_tendencia, metas_mec, color='#ff9999', alpha=0.2, label=f"Déficit em 2030 (~{gap_2030:.1f} p.p.)")

        plt.title(f"Projeção da Taxa de Alfabetização: {nome_uf.title()} | Modelo: {nome_modelo}", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Ano", fontsize=12)
        plt.ylabel("% Alunos Alfabetizados (2º ano)", fontsize=12)
        plt.xticks(np.arange(2019, 2031, 1))
        plt.ylim(min(min(taxas_hist), min(projecao_tendencia)) - 5, max(max(metas_mec), max(projecao_tendencia)) + 8)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)
        plt.tight_layout()
        plt.show()

        print("=" * 60)
        print(f"📊 Relatório de Projeção - {nome_uf.upper()} ({nome_modelo})")
        print("=" * 60)
        print(f"Taxa observada em 2023:        {taxas_hist[-1]:.2f}%")
        print(f"Projeção para 2030:            {projecao_tendencia[-1]:.2f}%")
        print(f"Meta oficial do MEC para 2030: {meta_2030:.2f}%")
        print(f"Gap a ser superado:            {gap_2030:.2f} p.p.")
        print("=" * 60)

    return {
        "df_projecao": df_projecao,
        "gap_2030": gap_2030,
        "modelo": modelo,
        "historico": mapa_historico
    }


def projetar_alfabetizacao_estados(
    df_metas: pd.DataFrame,
    ano_avaliacao: int = 2023,
    ano_inicio: int = 2019,
    ano_alvo: int = 2030,
    meta_corte: float = 80.0,
    ufs_excluir: list = None,
    plotar: bool = True,
    figsize: tuple = (12, 10)
):
    """
    Calcula a projeção da taxa de alfabetização por estado (UF) para um ano-alvo
    com base no ritmo histórico e compara com a meta de corte (ex: 80% do MEC).

    Parâmetros:
    -----------
    df_metas : pd.DataFrame
        DataFrame contendo os dados de avaliação e metas dos estados.
    ano_avaliacao : int, default 2023
        Ano base mais recente da avaliação.
    ano_inicio : int, default 2019
        Ano de início para o cálculo do ritmo anual histórico.
    ano_alvo : int, default 2030
        Ano final da projeção.
    meta_corte : float, default 80.0
        Taxa percentual de corte da meta oficial do MEC.
    ufs_excluir : list, optional
        Lista de siglas de UFs a serem desconsideradas (default: ["SC*"]).
    plotar : bool, default True
        Se True, gera e exibe o gráfico de barras horizontais ordenado.
    figsize : tuple, default (12, 10)
        Dimensões da figura do gráfico.

    Retorna:
    --------
    dict contendo:
        - 'df_ranking': DataFrame com as UFs ordenadas pela projeção, ritmo e gap.
        - 'atingem': Quantidade de estados que atingirão a meta.
        - 'total': Quantidade total de estados válidos analisados.
        - 'pct_atingem': Percentual de estados que atingirão a meta.
    """
    if ufs_excluir is None:
        ufs_excluir = ["SC*"]

    # 1. Filtra registros válidos por estado no ano de avaliação
    df_ufs = df_metas[
        (df_metas["NU_ANO_AVALIACAO"] == ano_avaliacao) &
        (df_metas["SIGLA_UF"].notnull()) &
        (~df_metas["SIGLA_UF"].isin(ufs_excluir))
    ].copy()

    # 2. Cálculo do ritmo anual e projeção
    delta_hist = float(ano_avaliacao - ano_inicio)
    delta_proj = float(ano_alvo - ano_avaliacao)

    col_proj = f"PROJ_{ano_alvo}"
    df_ufs["RITMO_ANUAL"] = (df_ufs["PC_ALUNO_ALFABETIZADO"] - df_ufs["SAEB_2019"]) / delta_hist
    df_ufs[col_proj] = (df_ufs["PC_ALUNO_ALFABETIZADO"] + df_ufs["RITMO_ANUAL"] * delta_proj).clip(lower=0.0, upper=100.0)
    df_ufs["GAP_META"] = meta_corte - df_ufs[col_proj]
    df_ufs["ATINGE_META"] = df_ufs[col_proj] >= meta_corte

    # 3. Ordenação para análise e plotagem
    df_plot = df_ufs.dropna(subset=[col_proj]).sort_values(col_proj, ascending=True).copy()

    atingem = int((df_plot[col_proj] >= meta_corte).sum())
    total = len(df_plot)
    pct_atingem = (atingem / total * 100.0) if total > 0 else 0.0

    # 4. Gráfico de Barras Horizontais
    if plotar and total > 0:
        cores = ["#2ca02c" if val >= meta_corte else "#d62728" for val in df_plot[col_proj]]
        labels_y = df_plot["SIGLA_UF"] + " - " + df_plot["NOME_UF"]

        plt.figure(figsize=figsize)
        barras = plt.barh(labels_y, df_plot[col_proj], color=cores, alpha=0.85)

        # Linha de corte da Meta
        plt.axvline(meta_corte, color="darkgreen", linestyle="--", lw=2, label=f"Meta {ano_alvo} MEC ({meta_corte:.0f}%)")

        # Rótulos com os valores em cada barra
        for barra in barras:
            w = barra.get_width()
            cor_texto = "#2ca02c" if w >= meta_corte else "#b22222"
            plt.text(
                w + 1,
                barra.get_y() + barra.get_height() / 2,
                f"{w:.1f}%",
                va='center',
                ha='left',
                fontsize=9,
                fontweight='bold',
                color=cor_texto
            )

        plt.title(f"Projeção da Taxa de Alfabetização por Estado em {ano_alvo} (Ritmo Histórico {ano_inicio}-{ano_avaliacao})", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel(f"% Previsto de Alunos Alfabetizados em {ano_alvo}", fontsize=11)
        plt.xlim(0, 110)
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        plt.legend(loc='lower right', frameon=True)
        plt.tight_layout()
        plt.show()

        print("=" * 70)
        print(f"📊 Panorama Nacional por Estados - Meta {ano_alvo} ({meta_corte:.0f}%)")
        print("=" * 70)
        print(f"Apenas {atingem} de {total} estados ({pct_atingem:.1f}%) atingirão a meta de {meta_corte:.0f}% no ritmo atual.")
        print("=" * 70)

    colunas_retorno = [
        "SIGLA_UF", "NOME_UF", "SAEB_2019", "PC_ALUNO_ALFABETIZADO",
        "RITMO_ANUAL", col_proj, "GAP_META", "ATINGE_META"
    ]
    colunas_existentes = [c for c in colunas_retorno if c in df_plot.columns]

    return {
        "df_ranking": df_plot[colunas_existentes],
        "atingem": atingem,
        "total": total,
        "pct_atingem": pct_atingem
    }


def plotar_importancia_features(
    modelo,
    X: pd.DataFrame = None,
    feature_names: list = None,
    top_n: int = 15,
    titulo: str = None,
    xlabel: str = "Importância",
    cor: str = "#1f77b4",
    plotar: bool = True,
    figsize: tuple = (10, 7)
):
    """
    Calcula, exibe e plota as features mais importantes de um modelo treinado
    (compatível com LightGBM, XGBoost, CatBoost, RandomForest e modelos lineares).

    Parâmetros:
    -----------
    modelo : Estimador treinado (ex: LGBMRegressor, RandomForest, Pipeline, etc.)
    X : pd.DataFrame, optional
        DataFrame com as features de treino/teste para extrair nomes de colunas caso o modelo não tenha.
    feature_names : list, optional
        Lista explícita com os nomes das features.
    top_n : int, default 15
        Quantidade de features mais relevantes a serem plotadas.
    titulo : str, optional
        Título customizado do gráfico.
    xlabel : str, default 'Importância'
        Legenda do eixo X.
    cor : str, default '#1f77b4'
        Cor das barras do gráfico.
    plotar : bool, default True
        Se True, gera e exibe o gráfico de barras horizontais.
    figsize : tuple, default (10, 7)
        Dimensões da figura do gráfico.

    Retorna:
    --------
    pd.Series ordenada de forma decrescente com as importâncias de todas as features.
    """
    # 1. Se for Pipeline do scikit-learn, extrai o estimador final
    estimador = modelo
    if hasattr(modelo, "steps"):
        estimador = modelo.steps[-1][1]

    # 2. Extração dos pesos / importâncias
    if hasattr(estimador, "feature_importances_"):
        importancias_raw = estimador.feature_importances_
    elif hasattr(estimador, "coef_"):
        importancias_raw = np.abs(estimador.coef_).flatten()
    else:
        raise AttributeError("O modelo fornecido não possui 'feature_importances_' nem 'coef_'.")

    # 3. Resolução dos nomes das features
    if feature_names is not None:
        nomes = list(feature_names)
    elif hasattr(estimador, "feature_name_"):  # Padrão LightGBM
        nomes = list(estimador.feature_name_)
    elif hasattr(estimador, "feature_names_in_"):  # Padrão Scikit-Learn 1.0+
        nomes = list(estimador.feature_names_in_)
    elif X is not None and hasattr(X, "columns"):
        nomes = list(X.columns)
    else:
        nomes = [f"feature_{i}" for i in range(len(importancias_raw))]

    # 4. Criação da Série ordenada
    importancias = pd.Series(importancias_raw, index=nomes).sort_values(ascending=False)

    # 5. Gráfico
    if plotar:
        top_features = importancias.head(top_n)
        nome_modelo = estimador.__class__.__name__

        plt.figure(figsize=figsize)
        ax = top_features.plot(kind="barh", color=cor, alpha=0.85)
        plt.gca().invert_yaxis()

        # Rótulos de texto com os valores nas barras
        max_val = top_features.max()
        offset = max_val * 0.01 if max_val > 0 else 0.1

        for i, val in enumerate(top_features):
            texto = f" {val:.0f}" if val >= 100 else f" {val:.2f}"
            ax.text(val + offset, i, texto, va='center', ha='left', fontsize=9, fontweight='bold', color='#333333')

        titulo_final = titulo or f"Top {min(top_n, len(top_features))} Features Mais Importantes - {nome_modelo}"
        plt.title(titulo_final, fontsize=13, fontweight='bold', pad=12)
        plt.xlabel(xlabel, fontsize=11)
        plt.xlim(0, max_val * 1.12 if max_val > 0 else 1)
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

        print(f"📌 Top {min(top_n, len(top_features))} Features:")
        print(top_features)

    return importancias


import os
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def carregar_modelo(nome_ou_caminho: str):
    """
    Carrega um modelo salvo em formato .joblib de forma resiliente.
    Busca automaticamente em múltiplos diretórios (raiz, models/, notebooks/models/).

    Parâmetros:
    -----------
    nome_ou_caminho : str
        Nome do arquivo (ex: 'modelo_lgbm_profundo.joblib') ou caminho relativo/absoluto.

    Retorna:
    --------
    Objeto do modelo carregado via joblib.
    """
    import joblib

    p = Path(nome_ou_caminho)
    # Se não tiver extensão, adiciona .joblib
    if not p.suffix:
        p = p.with_suffix(".joblib")

    # Lista de caminhos potenciais
    candidatos = [
        p,
        Path("models") / p.name,
        Path("notebooks/models") / p.name,
        Path("../models") / p.name,
        Path.cwd() / p,
        Path.cwd().parent / "models" / p.name,
    ]

    for cand in candidatos:
        if cand.exists() and cand.is_file():
            print(f"📦 Modelo carregado com sucesso de: {cand.resolve()}")
            return joblib.load(cand)

    raise FileNotFoundError(
        f"Não foi possível encontrar o arquivo '{nome_ou_caminho}'.\n"
        f"Tentativas verificadas:\n" + "\n".join([f" - {c.resolve()}" for c in candidatos])
    )


def salvar_modelo(modelo, nome_arquivo: str, pasta: str = "models"):
    """
    Salva um modelo treinado usando joblib, criando o diretório se necessário.

    Parâmetros:
    -----------
    modelo : estimador scikit-learn / lightgbm / etc.
    nome_arquivo : str
        Nome do arquivo (ex: 'modelo_lgbm_profundo.joblib').
    pasta : str, default 'models'
        Diretório onde o modelo será salvo.

    Retorna:
    --------
    Path do arquivo salvo.
    """
    import joblib

    p_pasta = Path(pasta)
    p_pasta.mkdir(parents=True, exist_ok=True)

    if not nome_arquivo.endswith(".joblib"):
        nome_arquivo += ".joblib"

    destino = p_pasta / nome_arquivo
    joblib.dump(modelo, destino)
    print(f"💾 Modelo salvo com sucesso em: {destino.resolve()}")
    return destino


def fazer_previsoes(
    modelo_ou_caminho,
    X,
    y_true=None,
    plotar: bool = True,
    figsize: tuple = (8, 6)
):
    """
    Gera previsões a partir de um modelo (ou caminho para o .joblib) e,
    caso os valores reais (y_true) sejam fornecidos, calcula métricas de erro
    (RMSE, MAE, R², MAPE) e plota o gráfico de Real vs Previsto.

    Parâmetros:
    -----------
    modelo_ou_caminho : Estimador ou str
        Instância do modelo carregado ou string com o caminho/nome do arquivo .joblib.
    X : pd.DataFrame ou np.ndarray
        Features de entrada para previsão (ex: X_test).
    y_true : pd.Series ou np.ndarray, optional
        Valores reais observados (ex: y_test) para validação.
    plotar : bool, default True
        Se True e y_true for fornecido, plota o gráfico de dispersão Real vs Previsto.
    figsize : tuple, default (8, 6)
        Tamanho da figura do gráfico.

    Retorna:
    --------
    dict contendo:
        - 'y_pred': Array com as previsões geradas.
        - 'df_resultados': DataFrame comparando Real, Previsto e Erro (se y_true informado).
        - 'metricas': Dicionário com RMSE, MAE, R2 e MAPE (se y_true informado).
        - 'modelo': Instância do modelo utilizado.
    """
    # 1. Carrega o modelo se for passado como caminho de texto
    if isinstance(modelo_ou_caminho, (str, Path)):
        modelo = carregar_modelo(str(modelo_ou_caminho))
    else:
        modelo = modelo_ou_caminho

    # 2. Faz as previsões
    y_pred = modelo.predict(X)
    if hasattr(y_pred, "flatten"):
        y_pred = y_pred.flatten()

    resultados = {
        "y_pred": y_pred,
        "modelo": modelo,
        "metricas": None,
        "df_resultados": None
    }

    # 3. Avaliação de Desempenho (caso y_true seja fornecido)
    if y_true is not None:
        y_val = np.array(y_true).flatten()

        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        mape = np.mean(np.abs((y_val - y_pred) / np.where(y_val == 0, 1e-6, y_val))) * 100

        metricas = {
            "RMSE": float(rmse),
            "MAE": float(mae),
            "R2": float(r2),
            "MAPE(%)": float(mape)
        }

        # DataFrame tabulado
        df_comp = pd.DataFrame({
            "Real": y_val,
            "Previsto": y_pred,
            "Erro_Absoluto": np.abs(y_val - y_pred),
            "Erro_Residual": y_val - y_pred
        })

        resultados["metricas"] = metricas
        resultados["df_resultados"] = df_comp

        print("=" * 55)
        print("🎯 Relatório de Desempenho do Modelo")
        print("=" * 55)
        print(f"R² Score: {r2:.4f}")
        print(f"RMSE:     {rmse:.4f}")
        print(f"MAE:      {mae:.4f}")
        print(f"MAPE:     {mape:.2f}%")
        print("=" * 55)

        # 4. Gráfico Real vs Previsto
        if plotar:
            plt.figure(figsize=figsize)
            plt.scatter(y_val, y_pred, alpha=0.6, color="#1f77b4", edgecolor='k', s=35, label="Observações")

            # Linha diagonal ideal (y = x)
            min_val = min(y_val.min(), y_pred.min())
            max_val = max(y_val.max(), y_pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], color='crimson', linestyle='--', lw=2, label="Previsão Ideal (y = x)")

            nome_mod = modelo.__class__.__name__
            plt.title(f"Validação: Real vs Previsto ({nome_mod})", fontsize=13, fontweight='bold', pad=12)
            plt.xlabel("Valor Real (y_true)", fontsize=11)
            plt.ylabel("Valor Previsto (y_pred)", fontsize=11)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend(loc="upper left")
            plt.tight_layout()
            plt.show()

    else:
        print(f"✅ {len(y_pred)} previsões geradas com sucesso!")

    return resultados



