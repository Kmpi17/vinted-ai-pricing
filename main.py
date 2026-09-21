import sys
import multiprocessing
from src.config import config
from src.reader.reader import download_and_read_parquet
from src.extraction.extractor import prepare_and_index_qdrant_points
from src.loader.loader import QdrantLoader

import sys
import multiprocessing

from src.config import config
from src.reader.reader import download_and_read_parquet
from src.extraction.extractor import prepare_and_index_qdrant_points
from src.loader.loader import QdrantLoader  

def main():
    print("🚀 Iniciando Pipeline ETL Distribuido (PySpark + Pandas UDF + Qdrant)...")

   
    num_cores = multiprocessing.cpu_count()
    print(f"--> Cores de CPU detectados para el cluster local: {num_cores}")

    # 1. Reader: Carga el Parquet en PySpark
    spark, df_spark = download_and_read_parquet(
        target_path=config.PARQUET_PATH,
        app_name=config.SPARK_APP_NAME
    )

    try:
        # 2. Recrear/Inicializar la colección en Qdrant antes de enviar datos
        print(f"--> Creando/Recreando colección '{config.COLLECTION_NAME}' en Qdrant...")
        loader = QdrantLoader(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
        loader.init_collection(
            collection_name=config.COLLECTION_NAME, 
            vector_size=config.VECTOR_SIZE,
            force_recreate=True
        )

        # 3. Extracción e Ingestión Distribuida (Workers -> Qdrant sin pasar por Driver)
        total_indexed = prepare_and_index_qdrant_points(
            df_spark=df_spark, 
            num_partitions=num_cores
        )

        print(f"🎉 Pipeline completado con éxito. Total de registros indexados: {total_indexed}")

    except Exception as e:
        print(f"❌ Error en la ejecución distribuida: {e}", file=sys.stderr)
        raise e
    finally:
        spark.stop()
        print("🛑 Sesión de PySpark finalizada correctamente.")

if __name__ == "__main__":
    main()