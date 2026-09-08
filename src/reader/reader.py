import os
import requests
from tqdm import tqdm
from pyspark.sql import SparkSession, DataFrame

def download_and_read_parquet(
    target_path: str = "data/fashion_dataset.parquet",
    app_name: str = "FashionDatasetReader"
) -> tuple[SparkSession, DataFrame]:
    """
    Descarga el dataset de Hugging Face si no existe localmente
    y lo carga en una sesión de PySpark.
    """
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    if not os.path.exists(target_path):
        api_url = "https://huggingface.co/api/datasets/ashraq/fashion-product-images-small/parquet/default/train"
        print("--> Consultando endpoint de Hugging Face...")
        res = requests.get(api_url).json()
        parquet_direct_url = res[0]

        print(f"--> Descargando Parquet desde: {parquet_direct_url}")
        response = requests.get(parquet_direct_url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(target_path, "wb") as f, tqdm(
            desc="Progreso de descarga",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    size = f.write(chunk)
                    bar.update(size)

        print(f"\n✅ Archivo guardado localmente en: {target_path}")
    else:
        print(f"--> Archivo encontrado en caché local: {target_path}")

    # Inicializar Spark
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    print("--> Cargando Parquet en PySpark...")
    df_spark = spark.read.parquet(target_path)
    
    return spark, df_spark