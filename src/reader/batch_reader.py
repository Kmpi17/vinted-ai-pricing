from pyspark.sql import SparkSession, DataFrame
from src import config

def read_parquet_batch(spark: SparkSession, path: str = config.PARQUET_PATH) -> DataFrame:
    return spark.read.parquet(path)