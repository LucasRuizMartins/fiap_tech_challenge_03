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




#-- carregar parquet por camada do s3
def carregar_parquet_s3(
    s3_client,
    bucket: str,
    ano: int,
    nome_tabela: str,
    camada: str = "bronze",
    ler_dicionario: bool = False,
) -> pd.DataFrame:
    """
    Carrega um arquivo Parquet do S3.

    Estrutura esperada:

    bronze/
        ano=2025/
            dados/
                TS_ALUNO.parquet

    ou

    bronze/
        ano=2025/
            dicionario/
                dicionario_TS_ALUNO.parquet
    """

    pasta = "dicionario" if ler_dicionario else "dados"

    arquivo = (
        f"dicionario_{nome_tabela}.parquet"
        if ler_dicionario
        else f"{nome_tabela}.parquet"
    )

    chave = f"{camada}/ano={ano}/{pasta}/{arquivo}"

    print(f"Lendo: s3://{bucket}/{chave}")

    try:

        obj = s3_client.get_object(
            Bucket=bucket,
            Key=chave
        )

        return pd.read_parquet(
            BytesIO(obj["Body"].read())
        )

    except s3_client.exceptions.NoSuchKey:
        raise FileNotFoundError(
            f"Arquivo não encontrado:\n"
            f"s3://{bucket}/{chave}"
        )

    except Exception as e:
        raise RuntimeError(
            f"Erro ao ler {chave} do bucket {bucket}.\n{e}"
        )



def ler_fato_s3(
    s3_client,
    bucket: str,
    ano: int,
    nome_tabela: str,
    camada: str = "bronze",
    ler_dicionario: bool = False,
) -> pd.DataFrame:
    """
    Carrega um arquivo Parquet do S3.

    Estrutura esperada:

    bronze/
        ano=2025/
            dados/
                TS_ALUNO.parquet

    ou

    bronze/
        ano=2025/
            dicionario/
                dicionario_TS_ALUNO.parquet
    """

    pasta = "dicionario" if ler_dicionario else "dados"

    arquivo = (
        f"dicionario_{nome_tabela}.parquet"
        if ler_dicionario
        else f"{nome_tabela}.parquet"
    )

    chave = f"{camada}/fato/ano=2026/dados/{arquivo}"

    print(f"Lendo: s3://{bucket}/{chave}")

    try:

        obj = s3_client.get_object(
            Bucket=bucket,
            Key=chave
        )

        return pd.read_parquet(
            BytesIO(obj["Body"].read())
        )

    except s3_client.exceptions.NoSuchKey:
        raise FileNotFoundError(
            f"Arquivo não encontrado:\n"
            f"s3://{bucket}/{chave}"
        )

    except Exception as e:
        raise RuntimeError(
            f"Erro ao ler {chave} do bucket {bucket}.\n{e}"
        )