"""
Pipeline 1: Extract raw data from Cloud SQL PostgreSQL to GCS as Parquet
This job reads tables from AdventureWorks database and saves them as-is to GCS
"""

from pyspark.sql import SparkSession
from google.cloud import secretmanager
import sys

# Configuration
PROJECT_ID = "vocal-sight-476612-r8"  # Replace with your project ID
GCS_BUCKET = "adventureworks-data-bc"  # Replace with your bucket name
GCS_RAW_PATH = f"gs://{GCS_BUCKET}/raw-data"
JDBC_DRIVER_PATH = f"gs://{GCS_BUCKET}/drivers/postgresql-42.7.7.jar"

# Tables to extract
SALES_TABLES = [
    "sales.salesorderheader",
    "sales.salesorderdetail",
    "sales.customer",
    "sales.salesperson",
    "sales.salesterritory"
]

PRODUCTION_TABLES = [
    "production.product",
    "production.productcategory",
    "production.productsubcategory"
]

PERSON_TABLES = [
    "person.person",
    "person.address"
]

ALL_TABLES = SALES_TABLES + PRODUCTION_TABLES + PERSON_TABLES


def get_secret(secret_id):
    """Retrieve secret from Secret Manager"""
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{PROJECT_ID}/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode('UTF-8')


def create_spark_session():
    """Create Spark session with JDBC driver"""
    spark = SparkSession.builder \
        .appName("AdventureWorks - Extract Raw Data") \
        .config("spark.jars", JDBC_DRIVER_PATH) \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("INFO")
    return spark


def extract_table_to_parquet(spark, jdbc_url, table_name, db_user, db_password):
    """
    Extract a single table from PostgreSQL and save as Parquet to GCS
    """
    print(f"Extracting table: {table_name}")
    
    try:
        # Read table from PostgreSQL
        df = spark.read \
            .format("jdbc") \
            .option("url", jdbc_url) \
            .option("dbtable", table_name) \
            .option("user", db_user) \
            .option("password", db_password) \
            .option("driver", "org.postgresql.Driver") \
            .load()
        
        # Show row count
        row_count = df.count()
        print(f"Table {table_name}: {row_count} rows")
        
        # Replace dots with underscores for file naming
        safe_table_name = table_name.replace(".", "_")
        
        # Save as Parquet to GCS
        output_path = f"{GCS_RAW_PATH}/{safe_table_name}"
        df.write \
            .mode("overwrite") \
            .parquet(output_path)
        
        print(f"Successfully saved {table_name} to {output_path}")
        return True
        
    except Exception as e:
        print(f"Error extracting {table_name}: {str(e)}")
        return False


def main():
    """Main extraction pipeline"""
    print("=" * 80)
    print("Starting AdventureWorks Raw Data Extraction Pipeline")
    print("=" * 80)
    
    # Get database credentials from Secret Manager
    print("\nRetrieving credentials from Secret Manager...")
    db_host = get_secret("db-host")
    db_name = get_secret("db-name")
    db_user = get_secret("db-username")
    db_password = get_secret("db-password")
    db_port = get_secret("db-port")
    
    # Construct JDBC URL
    jdbc_url = f"jdbc:postgresql://{db_host}:{db_port}/{db_name}"
    print(f"Connecting to: jdbc:postgresql://{db_host}:{db_port}/{db_name}")
    
    # Create Spark session
    print("\nInitializing Spark session...")
    spark = create_spark_session()
    
    # Extract all tables
    print(f"\nExtracting {len(ALL_TABLES)} tables to GCS...")
    print("-" * 80)
    
    success_count = 0
    failed_tables = []
    
    for table in ALL_TABLES:
        success = extract_table_to_parquet(spark, jdbc_url, table, db_user, db_password)
        if success:
            success_count += 1
        else:
            failed_tables.append(table)
        print("-" * 80)
    
    # Summary
    print("\n" + "=" * 80)
    print("EXTRACTION SUMMARY")
    print("=" * 80)
    print(f"Total tables: {len(ALL_TABLES)}")
    print(f"Successfully extracted: {success_count}")
    print(f"Failed: {len(failed_tables)}")
    
    if failed_tables:
        print("\nFailed tables:")
        for table in failed_tables:
            print(f"  - {table}")
    
    print("\nRaw data saved to:", GCS_RAW_PATH)
    print("=" * 80)
    
    # Stop Spark session
    spark.stop()
    
    # Exit with appropriate code
    if failed_tables:
        sys.exit(1)
    else:
        print("\n✅ Pipeline completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()