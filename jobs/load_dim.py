from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import (
    col, row_number, current_timestamp,
    explode, sequence, lit,
    year, month, dayofmonth, quarter, weekofyear, dayofweek, to_date, expr
)

PROJECT_ID = "vocal-sight-476612-r8"
GCS_BUCKET = "adventureworks-data-bc"
GCS_RAW_PATH = f"gs://{GCS_BUCKET}/raw-data"
BIGQUERY_DATASET = "adventureworks_dwh"
BIGQUERY_TEMP_BUCKET = GCS_BUCKET


def create_spark_session():
    return (
        SparkSession.builder
        .appName("AdventureWorks - Transform and Load Dimensions")
        .config("spark.jars.packages",
                "com.google.cloud.spark:spark-bigquery-with-dependencies_2.12:0.32.2")
        .config("spark.sql.shuffle.partitions", "50")
        .getOrCreate()
    )


def load_parquet_table(spark, table_name):
    safe_name = table_name.replace(".", "_")
    return spark.read.parquet(f"{GCS_RAW_PATH}/{safe_name}")


# DIM PRODUCT
def create_dim_product(spark):
    product = load_parquet_table(spark, "production.product")
    sub = load_parquet_table(spark, "production.productsubcategory")
    cat = load_parquet_table(spark, "production.productcategory")

    dim = (
        product.join(sub, "productsubcategoryid", "left")
               .join(cat, "productcategoryid", "left")
               .select(
                    product.productid.alias("product_src_id"),
                    product.name.alias("product_name"),
                    product.productnumber,
                    sub.name.alias("subcategory_name"),
                    cat.name.alias("category_name"),
                    product.color,
                    product.size,
                    product.listprice,
                    product.standardcost,
                )
    )

    w = Window.orderBy("product_src_id")
    dim = dim.withColumn("product_id", row_number().over(w)) \
             .withColumn("insert_dt", current_timestamp())

    dim = dim.select(
        "product_id",
        "product_src_id",
        "product_name",
        "productnumber",
        "subcategory_name",
        "category_name",
        "color",
        "size",
        "listprice",
        "standardcost",
        "insert_dt"
    )

    return dim


# DIM CUSTOMER
def create_dim_customer(spark):
    customer = load_parquet_table(spark, "sales.customer")
    person = load_parquet_table(spark, "person.person")

    dim = (
        customer.join(person, customer.personid == person.businessentityid, "left")
                .select(
                    customer.customerid.alias("customer_src_id"),
                    person.firstname,
                    person.lastname,
                    person.middlename,
                    customer.territoryid.alias("territory_src_id")
                )
    )

    w = Window.orderBy("customer_src_id")
    dim = dim.withColumn("customer_id", row_number().over(w)) \
             .withColumn("insert_dt", current_timestamp())

    dim = dim.select(
        "customer_id",
        "customer_src_id",
        "firstname",
        "lastname",
        "middlename",
        "territory_src_id",
        "insert_dt"
    )

    return dim


# DIM TERRITORY
def create_dim_territory(spark):
    territory = load_parquet_table(spark, "sales.salesterritory")

    dim = (
        territory
        .select(
            col("territoryid").alias("territory_src_id"),
            col("name").alias("territory_name"),
            col("countryregioncode"),
            col("group").alias("territory_group")
        )
    )

    w = Window.orderBy("territory_src_id")
    dim = dim.withColumn("territory_id", row_number().over(w)) \
             .withColumn("insert_dt", current_timestamp())

    dim = dim.select(
        "territory_id",
        "territory_src_id",
        "territory_name",
        "countryregioncode",
        "territory_group",
        "insert_dt"
    )

    return dim


# DIM DATE (2000–2030)
def create_dim_date(spark):
    print("\n=== Building Date Dimension (2000–2030) ===")

    start_date = "2000-01-01"
    end_date   = "2030-12-31"

    df = (
        spark.createDataFrame([(1,)], ["dummy"])
        .select(
            explode(
                sequence(
                    to_date(lit(start_date)),
                    to_date(lit(end_date)),
                    expr("INTERVAL 1 DAY")
                )
            ).alias("date_key")
        )
    )

    dim_date = (
        df.select(
            col("date_key"),
            year("date_key").alias("year"),
            month("date_key").alias("month"),
            dayofmonth("date_key").alias("day"),
            quarter("date_key").alias("quarter"),
            weekofyear("date_key").alias("weekofyear"),
            dayofweek("date_key").alias("dayofweek")
        )
    )

    print("dim_date rows:", dim_date.count())
    return dim_date


def write_to_bigquery(df, table_name):
    df.coalesce(5).write.format("bigquery") \
        .option("table", f"{PROJECT_ID}.{BIGQUERY_DATASET}.{table_name}") \
        .option("temporaryGcsBucket", BIGQUERY_TEMP_BUCKET) \
        .mode("overwrite") \
        .save()



def main():
    spark = create_spark_session()
    print("=== Loading Dimension Tables ===")

    dim_product = create_dim_product(spark)
    write_to_bigquery(dim_product, "dim_product")

    dim_customer = create_dim_customer(spark)
    write_to_bigquery(dim_customer, "dim_customer")

    dim_territory = create_dim_territory(spark)
    write_to_bigquery(dim_territory, "dim_territory")

    dim_date = create_dim_date(spark)
    write_to_bigquery(dim_date, "dim_date")

    spark.stop()


if __name__ == "__main__":
    main()
