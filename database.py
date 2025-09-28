#!/usr/bin/env python3
"""
Database Utilities
Dynamic schema extraction and database connection utilities
"""

import sqlite3

# Database connection
DB_PATH = "ecommerce_data.db"


def get_database_schema() -> str:
    """Dynamically retrieve database schema information for SQL expert agent."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Get all table names
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence';"
        )
        tables = cursor.fetchall()

        schema_info = "**Database Schema:**\n\n"

        for (table_name,) in tables:
            # Get table info
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()

            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]

            # Format table information
            schema_info += f"**{table_name}** ({row_count:,} rows):\n"
            column_list = []
            for col_info in columns:
                col_name = col_info[1]
                col_type = col_info[2]
                column_list.append(f"{col_name} ({col_type})")

            schema_info += f"- Columns: {', '.join(column_list)}\n"

            # Add sample data for better understanding
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
            sample_rows = cursor.fetchall()
            if sample_rows:
                schema_info += f"- Sample data preview: {len(sample_rows)} rows shown\n"
            schema_info += "\n"

        # Add important join relationships
        schema_info += "**Important Join Relationships:**\n"
        schema_info += (
            "- amazon_sale_report.sku = sale_report.sku_code (product inventory)\n"
        )
        schema_info += "- amazon_sale_report.sku = may_2022.sku (marketplace pricing)\n"
        schema_info += (
            "- amazon_sale_report.sku = international_sale_report.sku (B2B data)\n"
        )
        schema_info += (
            "- All tables can be joined on SKU variations (sku, sku_code)\n\n"
        )

        schema_info += "**Query Best Practices:**\n"
        schema_info += "- Always use LIMIT clauses for large tables (amazon_sale_report has 128k+ rows)\n"
        schema_info += "- Convert TEXT amounts to REAL: CAST(amount AS REAL)\n"
        schema_info += (
            "- Filter by status='Shipped - Delivered to Buyer' for confirmed sales\n"
        )
        schema_info += "- Use proper column names as shown in schema above\n"

        conn.close()
        return schema_info

    except Exception as e:
        return f"Error retrieving database schema: {str(e)}"
