from pyspark.sql import SparkSession
from src import config
from src.reader.batch_reader import read_parquet_batch
from src.extraction.feature_extractor import add_image_embeddings
from src.loader.qdrant_writer import write_to_qdrant

def main():
    print("🚀 Iniciando Pipeline ETL Moda Multimodal...")

    # 1. Crear sesión de PySpark
    spark = SparkSession.builder \
        .appName("VintedMultimodalIndexing") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    # 2. Leer Parquet limpio
    df = read_parquet_batch(spark, config.PARQUET_PATH)

    # Si quieres hacer una prueba rápida antes de procesar los 22k registros,
    # descomenta la siguiente línea para probar solo con 100 elementos:
    # df = df.limit(100)

    # 3. Generar Embeddings con CLIP
    print("--> Generando embeddings multimodales...")
    df_vectorized = add_image_embeddings(df)

    # 4. Insertar en Qdrant
    write_to_qdrant(df_vectorized)

    print("🎉 Pipeline finalizado con éxito.")

if __name__ == "__main__":
    main()