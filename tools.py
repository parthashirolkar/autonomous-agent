#!/usr/bin/env python3
"""
Data Analysis Tools
Tools for web search, SQL queries, and Python code execution with transparency
"""

import sqlite3
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
import time
import datetime
import json
import math
import re
import statistics
import itertools
import functools
import collections
import warnings
from io import StringIO

_ = load_dotenv()
console = Console()

# Database connection
DB_PATH = "ecommerce_data.db"

# Create Tavily search tool
tavily_search = TavilySearch(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=False,
    include_images=False,
)


@tool
async def web_search(query: str) -> str:
    """Search the internet for current information using Tavily search API."""
    try:
        console.print(f"[yellow]🔍 Searching web for: {query}[/yellow]")
        result = await tavily_search.ainvoke({"query": query})
        return f"Web search results for '{query}':\n{result}"
    except Exception as e:
        return f"Web search failed: {str(e)}"


@tool
async def execute_code(code: str) -> str:
    """Execute Python code with access to pandas, numpy, matplotlib for data analysis.

    Safe for data analysis with controlled access to standard library modules.
    Supports data manipulation, visualization, statistical analysis, and database operations.
    """
    try:
        # Display the code being executed
        console.print(
            Panel.fit(
                Syntax(code, "python", theme="monokai", line_numbers=True),
                title="[bold green]🐍 Python Code Being Executed[/bold green]",
                border_style="green",
            )
        )

        start_time = time.time()

        # Create a controlled __builtins__ environment
        safe_builtins = {
            # Essential built-ins for data analysis
            "__import__": __import__,  # Needed for dynamic imports
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "set": set,
            "frozenset": frozenset,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs,
            "round": round,
            "sorted": sorted,
            "reversed": reversed,
            "any": any,
            "all": all,
            "map": map,
            "filter": filter,
            "iter": iter,
            "next": next,
            "type": type,
            "isinstance": isinstance,
            "issubclass": issubclass,
            "hasattr": hasattr,
            "getattr": getattr,
            "setattr": setattr,
            "dir": dir,
            "vars": vars,
            "id": id,
            "hash": hash,
            "print": print,
            "repr": repr,
            "format": format,
            "ord": ord,
            "chr": chr,
            "hex": hex,
            "oct": oct,
            "bin": bin,
            "pow": pow,
            "divmod": divmod,
            # Exception handling
            "Exception": Exception,
            "ValueError": ValueError,
            "TypeError": TypeError,
            "KeyError": KeyError,
            "IndexError": IndexError,
            "AttributeError": AttributeError,
            # Safe I/O (limited to StringIO for capturing output)
            "open": None,  # Block file operations for security
            # Python language features
            "slice": slice,
            "property": property,
            "staticmethod": staticmethod,
            "classmethod": classmethod,
            "super": super,
        }

        # Create namespace with data analysis libraries and safe environment
        namespace = {
            "__builtins__": safe_builtins,
            # Core data analysis libraries
            "pd": pd,
            "np": np,
            "plt": plt,
            "sns": sns,
            "sqlite3": sqlite3,
            "DB_PATH": DB_PATH,
            # Standard library modules (safe for data analysis)
            "datetime": datetime,
            "json": json,
            "math": math,
            "re": re,
            "statistics": statistics,
            "itertools": itertools,
            "functools": functools,
            "collections": collections,
            "warnings": warnings,
            "StringIO": StringIO,
        }

        # Execute code in controlled environment
        exec(code, namespace)

        execution_time = time.time() - start_time

        # Display execution results
        console.print(
            f"[green]⏱️ Execution completed in {execution_time:.3f} seconds[/green]"
        )

        # Show created variables (excluding built-ins and modules)
        user_vars = {
            k: v
            for k, v in namespace.items()
            if not k.startswith("__")
            and k
            not in [
                "pd",
                "np",
                "plt",
                "sns",
                "sqlite3",
                "DB_PATH",
                "datetime",
                "json",
                "math",
                "re",
                "statistics",
                "itertools",
                "functools",
                "collections",
                "warnings",
                "StringIO",
            ]
        }

        if user_vars:
            var_table = Table(
                title="Variables Created", show_header=True, header_style="bold blue"
            )
            var_table.add_column("Variable", style="cyan")
            var_table.add_column("Type", style="magenta")
            var_table.add_column("Value/Shape", style="green")

            for var_name, var_value in user_vars.items():
                var_type = type(var_value).__name__
                if hasattr(var_value, "shape"):  # pandas/numpy objects
                    var_desc = f"Shape: {var_value.shape}"
                elif hasattr(var_value, "__len__") and var_type in [
                    "list",
                    "dict",
                    "str",
                ]:
                    var_desc = f"Length: {len(var_value)}"
                else:
                    var_desc = (
                        str(var_value)[:50] + "..."
                        if len(str(var_value)) > 50
                        else str(var_value)
                    )

                var_table.add_row(var_name, var_type, var_desc)

            console.print(var_table)

        # Return any result if there's a 'result' variable, otherwise success message
        if "result" in namespace:
            result_value = namespace["result"]
            # Handle different result types appropriately
            if hasattr(result_value, "to_string"):  # pandas DataFrame/Series
                console.print(
                    Panel.fit(
                        str(result_value.to_string()),
                        title="[bold blue]📊 Result Output[/bold blue]",
                        border_style="blue",
                    )
                )
                return str(result_value.to_string())
            elif hasattr(result_value, "__len__") and len(str(result_value)) > 5000:
                # Truncate very long outputs
                truncated = str(result_value)[:5000] + "... [output truncated]"
                console.print(
                    Panel.fit(
                        truncated,
                        title="[bold blue]📊 Result Output (Truncated)[/bold blue]",
                        border_style="blue",
                    )
                )
                return truncated
            else:
                console.print(
                    Panel.fit(
                        str(result_value),
                        title="[bold blue]📊 Result Output[/bold blue]",
                        border_style="blue",
                    )
                )
                return str(result_value)
        else:
            console.print("[green]✅ Code executed successfully[/green]")
            return "Code executed successfully"

    except ImportError as e:
        error_msg = (
            f"Import error: {str(e)}. Only safe data analysis modules are allowed."
        )
        console.print(f"[red]❌ {error_msg}[/red]")
        return error_msg
    except Exception as e:
        error_msg = f"Code execution failed: {str(e)}"
        console.print(f"[red]❌ {error_msg}[/red]")
        return error_msg


@tool
async def sql_query(query: str) -> str:
    """Execute SQL queries against the e-commerce database.

    Available tables:
    - amazon_sale_report: Amazon sales data (order_id, date, status, sku, category, amount, etc.)
    - sale_report: Product inventory (sku_code, design_no, stock, category, size, color)
    - may_2022: Product catalog with MRPs across platforms (sku, category, ajio_mrp, amazon_mrp, etc.)
    - international_sale_report: B2B transactions (date, customer, sku, pcs, rate, gross_amt)
    - p__l_march_2021: Profit/loss data (sku, category, mrp values across platforms)
    - expense_iigf: Expense tracking
    - cloud_warehouse_compersion_chart: Logistics comparison

    Use LIMIT clause for large datasets. Query carefully as this contains real business data.
    """
    try:
        # Display the SQL query being executed
        console.print(
            Panel.fit(
                Syntax(query, "sql", theme="monokai", line_numbers=True),
                title="[bold yellow]🗃️ SQL Query Being Executed[/bold yellow]",
                border_style="yellow",
            )
        )

        start_time = time.time()

        # Basic SQL injection protection
        dangerous_keywords = [
            "DROP",
            "DELETE",
            "UPDATE",
            "INSERT",
            "ALTER",
            "CREATE",
            "TRUNCATE",
        ]
        query_upper = query.upper()
        for keyword in dangerous_keywords:
            if keyword in query_upper:
                error_msg = f"SQL query rejected: Contains dangerous keyword '{keyword}'. Only SELECT queries are allowed."
                console.print(f"[red]❌ {error_msg}[/red]")
                return error_msg

        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(query, conn)
        conn.close()

        execution_time = time.time() - start_time

        # Display execution results
        console.print(
            f"[yellow]⏱️ Query executed in {execution_time:.3f} seconds[/yellow]"
        )
        console.print(f"[yellow]📊 Returned {len(df)} rows[/yellow]")

        # Format and display results
        if len(df) == 0:
            result_msg = "Query executed successfully but returned no results."
            console.print(f"[yellow]ℹ️ {result_msg}[/yellow]")
            return result_msg
        elif len(df) > 100:
            # Show sample of results
            console.print(
                Panel.fit(
                    df.head(10).to_string(),
                    title="[bold blue]📋 Query Results (First 10 of "
                    + str(len(df))
                    + " rows)[/bold blue]",
                    border_style="blue",
                )
            )
            result = f"Query returned {len(df)} rows. Showing first 100:\n{df.head(100).to_string()}"
        else:
            # Show all results
            console.print(
                Panel.fit(
                    df.to_string(),
                    title=f"[bold blue]📋 Query Results ({len(df)} rows)[/bold blue]",
                    border_style="blue",
                )
            )
            result = f"Query returned {len(df)} rows:\n{df.to_string()}"

        return result

    except Exception as e:
        error_msg = f"SQL query failed: {str(e)}"
        console.print(f"[red]❌ {error_msg}[/red]")
        return error_msg
