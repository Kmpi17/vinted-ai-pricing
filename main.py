import sys
from src.config import config
from src.reader.reader import download_and_read_parquet
from src.extraction.extractor import prepare_qdrant_points
from src.loader.loader import QdrantLoader

def main():
    print("🚀 Iniciando Pipeline ETL de Spark a Qdrant...")

    # 1. Reader: Descargar Parquet y cargar DataFrame en PySpark
    spark, df_spark = download_and_read_parquet(
        target_path=config.PARQUET_PATH,
        app_name=config.SPARK_APP_NAME
    )

    try:
        # 2. Extraction: Transformación de datos y generación de embeddings
        points = prepare_qdrant_points(df_spark)

        # 3. Loader: Carga masiva en Qdrant por lotes
        loader = QdrantLoader(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
        loader.init_collection(
            collection_name=config.COLLECTION_NAME, 
            vector_size=config.VECTOR_SIZE
        )
        loader.load_in_batches(
            collection_name=config.COLLECTION_NAME, 
            points=points, 
            batch_size=config.BATCH_SIZE
        )

    except Exception as e:
        print(f"❌ Error durante la ejecución del pipeline: {e}", file=sys.stderr)
        raise e
    finally:
        # Cerrar siempre la sesión de Spark limpia al finalizar
        spark.stop()
        print("🛑 Sesión de PySpark finalizada.")

if __name__ == "__main__":
    main()