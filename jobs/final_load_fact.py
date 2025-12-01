from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, sum as _sum, count, avg,
    year, month, dayofmonth, to_date
)
import sys

PROJECT_ID = "vocal-sight-476612-r8"
GCS_BUCKET = "adventureworks-data-bc"
GCS_RAW_PATH = f"gs://{GCS_BUCKET}/raw-data"
BIGQUERY_DATASET = "adventureworks_dwh"
BIGQUERY_TEMP_BUCKET = GCS_BUCKET


def create_spark_session():
    return (
        SparkSession.builder
        .appName("AdventureWorks - Load Facts with Surrogate Keys")
        .config("spark.jars.packages",
                "com.google.cloud.spark:spark-bigquery-with-dependencies_2.12:0.32.2")
        .config("spark.sql.shuffle.partitions", "50")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.default.parallelism", "50")
        .getOrCreate()
    )


def load_parquet_table(spark, table_name):
    safe_name = table_name.replace(".", "_")
    return spark.read.parquet(f"{GCS_RAW_PATH}/{safe_name}")


def load_dimension(spark, table):
    return (
        spark.read.format("bigquery")
        .option("table", f"{PROJECT_ID}.{BIGQUERY_DATASET}.{table}")
        .load()
    )


# FACT SALES DETAILED
def create_fact_sales_detailed(spark):

    dim_product = load_dimension(spark, "dim_product")
    dim_customer = load_dimension(spark, "dim_customer")
    dim_territory = load_dimension(spark, "dim_territory")
    dim_date = load_dimension(spark, "dim_date")

    df_header = load_parquet_table(spark, "sales.salesorderheader")
    df_detail = load_parquet_table(spark, "sales.salesorderdetail")

    df_header = df_header.dropDuplicates(["salesorderid"])
    df_detail = df_detail.dropDuplicates(["salesorderid", "salesorderdetailid"])

    df_detail = df_detail.withColumn(
        "linetotal",
        col("unitprice") * col("orderqty") * (1 - col("unitpricediscount"))
    )

    df_header = (
        df_header
        .withColumn("order_date_key", to_date("orderdate"))
        .withColumn("due_date_key", to_date("duedate"))
        .withColumn("ship_date_key", to_date("shipdate"))
        .join(dim_date, col("order_date_key") == dim_date.date_key, "left")
        .withColumn("order_date_key", dim_date.date_key)
        .drop(dim_date.date_key)
    )

    fact_df = (
        df_detail
        .join(df_header, "salesorderid", "inner")
        .join(dim_product, df_detail.productid == dim_product.product_src_id, "left")
        .join(dim_customer, df_header.customerid == dim_customer.customer_src_id, "left")
        .join(dim_territory, df_header.territoryid == dim_territory.territory_src_id, "left")
        .join(dim_date.withColumnRenamed("date_key", "order_date_key_dim"),
              col("orderdate").cast("date") == col("order_date_key_dim"),
              "left")
        .select(
            "salesorderid",
            "salesorderdetailid",
            col("product_id"),
            col("customer_id"),
            col("territory_id"),
            col("order_date_key_dim").alias("order_date_key"),
            "orderqty",
            "unitprice",
            "unitpricediscount",
            "linetotal",
            "subtotal",
            "taxamt",
            "freight",
            "totaldue",
            "orderdate",
            "duedate",
            "shipdate",
            "status"
        )
    )

    return fact_df


# FACT AGGREGATED 
def create_fact_sales_aggregated(spark, fact_df):

    fact_agg = (
        fact_df
        .withColumn("order_date", col("orderdate").cast("date"))
        .groupBy("order_date", "product_id")
        .agg(
            count("salesorderdetailid").alias("order_count"),
            _sum("orderqty").alias("total_quantity"),
            _sum("linetotal").alias("total_sales"),
            avg("unitprice").alias("avg_unit_price")
        )
        .coalesce(10)
    )

    return fact_agg


def write_to_bigquery(df, table_name):
    df.coalesce(5).write.format("bigquery") \
        .option("table", f"{PROJECT_ID}.{BIGQUERY_DATASET}.{table_name}") \
        .option("temporaryGcsBucket", BIGQUERY_TEMP_BUCKET) \
        .mode("overwrite") \
        .save()


def main():
    spark = create_spark_session()

    fact_sales_detailed = create_fact_sales_detailed(spark)
    write_to_bigquery(fact_sales_detailed, "fact_sales_detailed")

    fact_sales_detailed.persist()

    fact_sales_aggregated = create_fact_sales_aggregated(spark, fact_sales_detailed)
    write_to_bigquery(fact_sales_aggregated, "fact_sales_aggregated")

    fact_sales_detailed.unpersist()

    spark.stop()


if __name__ == "__main__":
    main()
