import os
import requests
import sys
from tqdm import tqdm
from pyspark.sql import SparkSession, DataFrame

def download_and_read_parquet(
    target_path: str = "data/fashion_dataset.parquet",
    dataset_url: str = "https://huggingface.co/api/datasets/perinim/deepfashion2/parquet/default/train",
    app_name: str = "DeepFashionSparkReader"
) -> tuple[SparkSession, DataFrame]:

    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    if not os.path.exists(target_path):
        print("--> Consultando endpoint de Hugging Face...")
        res = requests.get(dataset_url).json()
        parquet_direct_url = res[0] if isinstance(res, list) else res

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

    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "256") \
        .config("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true") \
        .getOrCreate()

    print("--> Cargando Parquet en PySpark...")
    df_spark = spark.read.parquet(target_path)
    
    return spark, df_spark