#!/usr/bin/env python3
"""
Database setup script for e-commerce data analysis.
Creates SQLite database from CSV files in the csv-data folder.
"""

import sqlite3
import pandas as pd
import os
import logging
from pathlib import Path
from typing import Dict

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def clean_column_name(col_name: str) -> str:
    """Clean column names for SQL compatibility."""
    # Remove special characters and spaces, replace with underscores
    cleaned = col_name.strip()
    cleaned = cleaned.replace(" ", "_")
    cleaned = cleaned.replace("-", "_")
    cleaned = cleaned.replace(".", "_")
    cleaned = cleaned.replace("(", "")
    cleaned = cleaned.replace(")", "")
    cleaned = cleaned.replace("/", "_")
    cleaned = cleaned.replace("&", "and")
    cleaned = cleaned.replace("%", "percent")
    # Remove any remaining special characters
    cleaned = "".join(c for c in cleaned if c.isalnum() or c == "_")
    # Ensure it doesn't start with a number
    if cleaned and cleaned[0].isdigit():
        cleaned = "col_" + cleaned
    return cleaned.lower()


def infer_sql_type(series: pd.Series) -> str:
    """Infer SQL data type from pandas series."""
    if pd.api.types.is_integer_dtype(series):
        return "INTEGER"
    elif pd.api.types.is_float_dtype(series):
        return "REAL"
    elif pd.api.types.is_datetime64_any_dtype(series):
        return "DATE"
    elif pd.api.types.is_bool_dtype(series):
        return "BOOLEAN"
    else:
        return "TEXT"


def create_table_schema(df: pd.DataFrame, table_name: str) -> str:
    """Generate CREATE TABLE SQL statement from DataFrame."""
    columns = []

    for col in df.columns:
        clean_col = clean_column_name(col)
        sql_type = infer_sql_type(df[col])
        columns.append(f"    {clean_col} {sql_type}")

    schema = f"CREATE TABLE IF NOT EXISTS {table_name} (\n"
    schema += ",\n".join(columns)
    schema += "\n);"

    return schema


def setup_database(
    csv_folder: str = "csv-data", db_path: str = "ecommerce_data.db"
) -> Dict[str, str]:
    """
    Create SQLite database from CSV files.

    Args:
        csv_folder: Path to folder containing CSV files
        db_path: Path for the SQLite database file

    Returns:
        Dict mapping table names to their schemas
    """
    csv_path = Path(csv_folder)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV folder not found: {csv_folder}")

    # Remove existing database
    if os.path.exists(db_path):
        os.remove(db_path)
        logger.info(f"Removed existing database: {db_path}")

    # Connect to SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    table_schemas = {}
    csv_files = list(csv_path.glob("*.csv"))

    logger.info(f"Found {len(csv_files)} CSV files to process")

    for csv_file in csv_files:
        try:
            logger.info(f"Processing: {csv_file.name}")

            # Create table name from filename
            table_name = csv_file.stem.lower()
            table_name = clean_column_name(table_name)

            # Read CSV file
            df = pd.read_csv(csv_file, low_memory=False)
            logger.info(f"  - Loaded {len(df)} rows, {len(df.columns)} columns")

            # Drop the index column if it exists (pandas artifact)
            if "index" in df.columns:
                df = df.drop("index", axis=1)
                logger.info("  - Dropped index column")

            # Clean column names
            df.columns = [clean_column_name(col) for col in df.columns]

            # Handle missing values
            df = df.fillna("")  # Replace NaN with empty strings for now

            # Create table schema
            schema = create_table_schema(df, table_name)
            table_schemas[table_name] = schema

            # Create table
            cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
            cursor.execute(schema)
            logger.info(f"  - Created table: {table_name}")

            # Insert data
            df.to_sql(table_name, conn, if_exists="append", index=False)
            logger.info(f"  - Inserted {len(df)} rows")

            # Create some basic indexes for common columns
            common_index_columns = ["date", "sku", "order_id", "category", "status"]
            for col in df.columns:
                if any(keyword in col.lower() for keyword in common_index_columns):
                    try:
                        index_name = f"idx_{table_name}_{col}"
                        cursor.execute(
                            f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name}({col})"
                        )
                        logger.info(f"  - Created index: {index_name}")
                    except sqlite3.Error as e:
                        logger.warning(f"  - Could not create index on {col}: {e}")

        except Exception as e:
            logger.error(f"Error processing {csv_file.name}: {e}")
            continue

    # Create a summary table with metadata
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS table_metadata (
            table_name TEXT PRIMARY KEY,
            row_count INTEGER,
            column_count INTEGER,
            created_date TEXT
        )
    """)

    # Get table statistics
    for table_name in table_schemas.keys():
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cursor.fetchone()[0]

        cursor.execute(f"PRAGMA table_info({table_name})")
        column_count = len(cursor.fetchall())

        cursor.execute(
            """
            INSERT OR REPLACE INTO table_metadata
            (table_name, row_count, column_count, created_date)
            VALUES (?, ?, ?, datetime('now'))
        """,
            (table_name, row_count, column_count),
        )

    conn.commit()
    conn.close()

    logger.info(f"Database creation completed: {db_path}")
    logger.info(f"Created {len(table_schemas)} tables")

    return table_schemas


def print_database_info(db_path: str = "ecommerce_data.db"):
    """Print database information and table schemas."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("\n" + "=" * 80)
    print("E-COMMERCE DATABASE SUMMARY")
    print("=" * 80)

    # Get table metadata
    cursor.execute("SELECT * FROM table_metadata ORDER BY table_name")
    metadata = cursor.fetchall()

    print(f"\nTotal Tables: {len(metadata)}")
    print("-" * 80)

    for table_name, row_count, column_count, created_date in metadata:
        print(f"Table: {table_name}")
        print(f"  Rows: {row_count:,}")
        print(f"  Columns: {column_count}")
        print(f"  Created: {created_date}")

        # Show sample of column names
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [row[1] for row in cursor.fetchall()]
        print(
            f"  Sample columns: {', '.join(columns[:5])}{'...' if len(columns) > 5 else ''}"
        )
        print()

    conn.close()


if __name__ == "__main__":
    try:
        # Setup database
        schemas = setup_database()

        # Print summary
        print_database_info()

        # Save schemas to file for reference
        with open("database_schemas.sql", "w") as f:
            f.write("-- E-commerce Database Schemas\n")
            f.write("-- Generated automatically from CSV files\n\n")
            for table_name, schema in schemas.items():
                f.write(f"-- Table: {table_name}\n")
                f.write(schema)
                f.write("\n\n")

        print("Database schemas saved to: database_schemas.sql")

    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        raise
