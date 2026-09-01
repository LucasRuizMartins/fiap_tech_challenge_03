from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ==============================================================================
# 1. Configurações Globais de Estilo e Pastas
# ==============================================================================
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.autolayout'] = True

# Pasta de saída dos gráficos (reports/figures)
PASTA_FIGURES = Path(__file__).parent.parent.parent / 'reports' / 'figures'
PASTA_FIGURES.mkdir(parents=True, exist_ok=True)


def salvar_fig(fig, nome_arquivo: str):
    """Salva a figura na pasta reports/figures em alta resolução."""
    caminho = PASTA_FIGURES / nome_arquivo
    fig.savefig(caminho, dpi=300, bbox_inches='tight')
    print(f"📊 Gráfico salvo em: {caminho.resolve()}")


# ==============================================================================
# 2. Dicionários de Tradução e Mapeamento do INEP / Educação
# ==============================================================================
# De-para de Dependência Administrativa (Rede)
dependencia_map = {
    1: 'Federal',
    2: 'Estadual',
    3: 'Municipal',
    4: 'Privada'
}

# Paleta de cores harmoniosa para redes de ensino
cores_dependencia = {
    'Federal': '#2B5B84',    # Azul escuro
    'Estadual': '#E67E22',   # Laranja
    'Municipal': '#27AE60',  # Verde
    'Privada': '#8E44AD'     # Roxo
}

# Dicionário de nomes amigáveis para rótulos e títulos
dicionario_colunas = {
    'VL_MEAN_PROFICIENCIA_LP': 'Proficiência Média em Língua Portuguesa',
    'VL_MEDIA_LP': 'Média de Língua Portuguesa',
    'VL_MEDIA_LP_TOTAL_MUNICIPIO': 'Média de LP do Município',
    'VL_MEDIA_LP_TOTAL_UF': 'Média de LP do Estado',
    'QT_ALUNOS': 'Quantidade de Alunos',
    'QT_PRESENTES_LP': 'Alunos Presentes na Avaliação',
    'QT_ALFABETIZADOS': 'Quantidade de Alunos Alfabetizados',
    'PC_ALUNO_ALFABETIZADO': '% de Alunos Alfabetizados',
    'PC_ALUNO_ALFABETIZADO_TOTAL_MUNICIPIO': '% Alfabetizados no Município',
    'PC_ALUNO_ALFABETIZADO_TOTAL_UF': '% Alfabetizados no Estado',
    'ABSTENCAO_LP': 'Taxa de Abstenção (%)',
    'TP_DEPENDENCIA': 'Rede de Ensino',
    'NO_REGIAO': 'Região do Brasil',
    'SG_UF': 'Unidade da Federação (UF)',
    'DESC_SERIE': 'Série / Ano Escolar',
    'META_MUN_2024': 'Meta Municipal 2024',
    'META_MUN_2025': 'Meta Municipal 2025',
    'META_UF_2024': 'Meta Estadual 2024',
}


# ==============================================================================
# 3. Funções de Visualização para EDA
# ==============================================================================

def gerar_kdeplot(df: pd.DataFrame, x: str, hue: str = 'TP_DEPENDENCIA', bw_adjust: float = 1.2, salvar: bool = True):
    """
    Gera gráfico de densidade (KDE) para variáveis contínuas (ex: Proficiência ou % Alfabetizados),
    comparando por rede de ensino ou região.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    df_plot = df.copy()

    # Mapeia rótulos da rede se for TP_DEPENDENCIA
    palette = None
    if hue == 'TP_DEPENDENCIA' and 'TP_DEPENDENCIA' in df_plot.columns:
        df_plot[hue] = df_plot[hue].map(dependencia_map).fillna(df_plot[hue])
        palette = cores_dependencia

    sns.kdeplot(
        data=df_plot,
        x=x,
        hue=hue,
        palette=palette,
        fill=True,
        common_norm=False,
        bw_adjust=bw_adjust,
        alpha=0.35,
        ax=ax
    )

    nome_x = dicionario_colunas.get(x, x)
    nome_hue = dicionario_colunas.get(hue, hue)
    
    ax.set_title(f'Distribuição de {nome_x} por {nome_hue}', fontsize=13, weight='bold', pad=12)
    ax.set_xlabel(nome_x, fontsize=11)
    ax.set_ylabel('Densidade', fontsize=11)

    if salvar:
        salvar_fig(fig, f"kde_{x}_por_{hue}.png")
    
    plt.show()


def gerar_boxplot(df: pd.DataFrame, x: str, y: str, salvar: bool = True):
    """
    Gera Boxplot para comparar dispersão (ex: Proficiência por Região ou Rede).
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    df_plot = df.copy()

    if x == 'TP_DEPENDENCIA':
        df_plot[x] = df_plot[x].map(dependencia_map).fillna(df_plot[x])

    sns.boxplot(
        data=df_plot,
        x=x,
        y=y,
        hue=x,
        legend=False,
        palette="Set2",
        ax=ax,
        showmeans=True,
        meanprops={"marker":"o", "markerfacecolor":"red", "markeredgecolor":"red"}
    )

    nome_x = dicionario_colunas.get(x, x)
    nome_y = dicionario_colunas.get(y, y)

    ax.set_title(f'Dispersão de {nome_y} por {nome_x}', fontsize=13, weight='bold', pad=12)
    ax.set_xlabel(nome_x, fontsize=11)
    ax.set_ylabel(nome_y, fontsize=11)

    if salvar:
        salvar_fig(fig, f"box_{y}_por_{x}.png")
    
    plt.show()


def gerar_barplot_medias(df: pd.DataFrame, x: str, y: str, top_n: int = None, salvar: bool = True):
    """
    Gera gráfico de barras com a média de Y agrupada por X com rótulos de valores nas barras.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    
    df_media = df.groupby(x)[y].mean().reset_index().sort_values(by=y, ascending=False)
    if top_n:
        df_media = df_media.head(top_n)

    if x == 'TP_DEPENDENCIA':
        df_media[x] = df_media[x].map(dependencia_map).fillna(df_media[x])

    sns.barplot(data=df_media, x=x, y=y, hue=x, legend=False, palette="Blues_r", ax=ax)

    nome_x = dicionario_colunas.get(x, x)
    nome_y = dicionario_colunas.get(y, y)

    ax.set_title(f'Média de {nome_y} por {nome_x}', fontsize=13, weight='bold', pad=12)
    ax.set_xlabel(nome_x, fontsize=11)
    ax.set_ylabel(f'Média de {nome_y}', fontsize=11)
    plt.xticks(rotation=45 if len(df_media) > 6 else 0)

    # Rótulo de valores em cima das barras
    for p in ax.patches:
        altura = p.get_height()
        if not pd.isna(altura) and altura > 0:
            ax.annotate(f'{altura:.1f}',
                        (p.get_x() + p.get_width() / 2., altura),
                        ha='center', va='bottom', fontsize=10, xytext=(0, 3),
                        textcoords='offset points')

    if salvar:
        salvar_fig(fig, f"bar_media_{y}_por_{x}.png")
        
    plt.show()


def gerar_matriz_correlacao(df: pd.DataFrame, colunas: list = None, salvar: bool = True):
    """
    Gera um Heatmap de correlação para as colunas numéricas especificadas.
    """
    # Se não passar colunas, pega todas as numéricas (removendo colunas que contenham ID ou CO)
    if colunas is None:
        colunas = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if not any(id_term in c for id_term in ['ID_', 'CO_', 'CADERNO', 'BLOCO'])
        ]
    else:
        colunas = [c for c in colunas if c in df.columns]

    corr = df[colunas].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Nomes amigáveis nos eixos
    nomes_formatados = [dicionario_colunas.get(c, c) for c in colunas]

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        corr,
        mask=mask,
        cmap="coolwarm",
        vmax=1.0,
        vmin=-1.0,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        xticklabels=nomes_formatados,
        yticklabels=nomes_formatados,
        cbar_kws={"shrink": 0.75},
        ax=ax
    )

    ax.set_title("Matriz de Correlação das Variáveis", fontsize=14, weight="bold", pad=15)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    if salvar:
        salvar_fig(fig, "matriz_correlacao.png")

    plt.show()
