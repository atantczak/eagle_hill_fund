import boto3
import logging
import time

import pandas as pd
from botocore.exceptions import ClientError
from boto3.dynamodb.conditions import Attr
from sqlalchemy import create_engine, text, MetaData, Table, Column, String, Integer, Float, DateTime, Boolean, Text
from sqlalchemy.exc import SQLAlchemyError

logging.basicConfig(level=logging.INFO)

from payfusion.server.payfusion.apps.integrations.aws.credentials.access_keys import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY_ID,
)


class RDS:
    """
    Class to manage PostgreSQL RDS read/write operations and integrations,
    optimized for high-volume querying and batch exports.
    """

    def __init__(
        self,
        db_name: str,
        db_user: str,
        db_password: str,
        host: str,
        table_name: str = None,
        region: str = "us-east-2",
    ):
        """
        Initialize the RDS connection and set up the SQLAlchemy engine.

        Args:
            db_name (str): The name of the database.
            table_name (str): The target table in the database.
            db_user (str): The RDS database username.
            db_password (str): The RDS database password.
            host (str): The RDS endpoint hostname (e.g., mydb.xxxxxx.us-east-2.rds.amazonaws.com).
            region (str): AWS region where the RDS instance is hosted.
        """
        self.db_name = db_name
        self.table_name = table_name
        self.region = region

        self.engine = create_engine(f"postgresql+psycopg2://{db_user}:{db_password}@{host}:5432/{db_name}")

    def push_dataframe_to_rds(self, df: pd.DataFrame, if_exists: str = "append"):
        """
        Push a pandas DataFrame to the RDS table.

        Args:
            df (pd.DataFrame): The DataFrame to insert.
            if_exists (str): 'append' to add rows, 'replace' to drop and recreate the table.
        """
        try:
            # Create a copy of the DataFrame with sanitized column names
            df_copy = df.copy()
            
            # Always sanitize column names to match the table schema
            column_mapping = {}
            for col in df_copy.columns:
                # Sanitize to match the same logic used in create_table_from_dataframe
                safe_col = col.replace(' ', '_').replace('-', '_').lower()
                if col != safe_col:
                    column_mapping[col] = safe_col
            
            if column_mapping:
                df_copy = df_copy.rename(columns=column_mapping)
            
            df_copy.to_sql(self.table_name, self.engine, if_exists=if_exists, index=False, chunksize=1000)
        except SQLAlchemyError as e:
            raise RuntimeError(f"Error pushing DataFrame to RDS: {e}")

    def add_single_row(self, row_data: dict, unique_field: str = None) -> bool:
        """
        Add or update a single row in the RDS table. If a unique_field is specified,
        it will check for existing records and update them instead of creating duplicates.

        Args:
            row_data (dict): Dictionary containing column names and values for the row.
                           Format: {"column_name": "value", ...}
            unique_field (str): Optional field to check for existing records.
                              If specified, will update existing record if found.

        Returns:
            bool: True if row was added/updated successfully, False otherwise.
        """
        try:
            # Validate row data against table schema
            valid_data = self.validate_row_data(row_data)
            
            if not valid_data:
                logging.error(f"No valid columns found for table {self.table_name}")
                return False

            # If no unique field specified, just insert as before
            if not unique_field:
                df = pd.DataFrame([valid_data])
                self.push_dataframe_to_rds(df, if_exists="append")
                logging.info(f"Single row added to table {self.table_name} successfully")
                return True

            # Check for existing record if unique_field specified
            if unique_field not in valid_data:
                logging.error(f"Specified unique field '{unique_field}' not found in row data")
                return False

            # Get existing data as DataFrame
            existing_df = self.pull_dataframe_from_rds()

            # Check if record exists
            mask = existing_df[unique_field] == valid_data[unique_field]
            matching_rows = existing_df[mask]

            if matching_rows.empty:
                # No existing record found, insert new one
                df = pd.DataFrame([valid_data])
                self.push_dataframe_to_rds(df, if_exists="append")
                logging.info(f"Single row added to table {self.table_name} successfully")
                return True
            else:
                # Compare existing record with new data
                existing_row = matching_rows.iloc[0].to_dict()
                if all(existing_row.get(k) == v for k, v in valid_data.items() if k in existing_row):
                    logging.info(f"Record already exists with identical values, no update needed")
                    return True
                else:
                    # Update existing record
                    for col, value in valid_data.items():
                        existing_df.loc[mask, col] = value
                    
                    # Write back entire DataFrame
                    existing_df.to_sql(self.table_name, self.engine, if_exists='replace', index=False)
                    logging.info(f"Existing record updated in table {self.table_name} successfully")
                    return True

        except SQLAlchemyError as e:
            logging.error(f"Error adding/updating row in table {self.table_name}: {e}")
            return False

    def add_single_row_sql(self, row_data: dict) -> bool:
        """
        Add a single row using direct SQL INSERT (more efficient for single rows).

        Args:
            row_data (dict): Dictionary containing column names and values for the row.
                           Format: {"column_name": "value", ...}

        Returns:
            bool: True if row was added successfully, False otherwise.
        """
        try:
            # Sanitize column names
            sanitized_data = {}
            for col, value in row_data.items():
                safe_col = col.replace(' ', '_').replace('-', '_').lower()
                sanitized_data[safe_col] = value
            
            # Build INSERT statement
            columns = list(sanitized_data.keys())
            values = list(sanitized_data.values())
            placeholders = [f":{i}" for i in range(len(values))]
            
            insert_sql = f"""
                INSERT INTO {self.table_name} ({', '.join(columns)})
                VALUES ({', '.join(placeholders)})
            """
            
            # Execute the insert
            with self.engine.connect() as conn:
                conn.execute(text(insert_sql), dict(zip(placeholders, values)))
                conn.commit()
            
            logging.info(f"Single row added to table {self.table_name} successfully")
            return True
            
        except SQLAlchemyError as e:
            logging.error(f"Error adding single row to table {self.table_name}: {e}")
            return False

    def add_single_row_with_return(self, row_data: dict, return_column: str = "id") -> any:
        """
        Add a single row and return the value of a specified column (e.g., auto-generated ID).

        Args:
            row_data (dict): Dictionary containing column names and values for the row.
            return_column (str): Column name to return (default: "id")

        Returns:
            any: Value of the specified column, or None if insertion failed.
        """
        try:
            # Sanitize column names
            sanitized_data = {}
            for col, value in row_data.items():
                safe_col = col.replace(' ', '_').replace('-', '_').lower()
                sanitized_data[safe_col] = value
            
            # Build INSERT statement with RETURNING clause
            columns = list(sanitized_data.keys())
            values = list(sanitized_data.values())
            placeholders = [f":{i}" for i in range(len(values))]
            
            insert_sql = f"""
                INSERT INTO {self.table_name} ({', '.join(columns)})
                VALUES ({', '.join(placeholders)})
                RETURNING {return_column}
            """
            
            # Execute the insert and get the returned value
            with self.engine.connect() as conn:
                result = conn.execute(text(insert_sql), dict(zip(placeholders, values)))
                returned_value = result.scalar()
                conn.commit()
            
            logging.info(f"Single row added to table {self.table_name} with returned {return_column}: {returned_value}")
            return returned_value
            
        except SQLAlchemyError as e:
            logging.error(f"Error adding single row to table {self.table_name}: {e}")
            return None

    def pull_dataframe_from_rds(self, query: str = None) -> pd.DataFrame:
        """
        Pull data from the RDS table or run a custom SQL query.

        Args:
            query (str, optional): Custom SQL query to execute. Defaults to SELECT *.

        Returns:
            pd.DataFrame: Query results as a DataFrame.
        """
        try:
            if query is None:
                query = f"SELECT * FROM {self.table_name}"
            return pd.read_sql(query, self.engine)
        except SQLAlchemyError as e:
            raise RuntimeError(f"Error pulling DataFrame from RDS: {e}")

    def filter_dataframe(self, filter_query: str) -> pd.DataFrame:
        """
        Filter data from the table using a raw SQL WHERE clause.

        Args:
            filter_query (str): SQL WHERE clause (without 'WHERE').

        Returns:
            pd.DataFrame: Filtered results as a DataFrame.
        """
        query = f"SELECT * FROM {self.table_name} WHERE {filter_query}"
        return self.pull_dataframe_from_rds(query)

    def filter_by_field(self, field: str, value: str, operator: str = "=", use_sql: bool = False) -> pd.DataFrame:
        """
        Filter data from the table by a specific field/value/operator.

        Args:
            field (str): Column name to filter by.
            value (str): Value to match.
            operator (str): SQL operator (e.g., '=', 'LIKE').

        Returns:
            pd.DataFrame: Filtered results as a DataFrame.
        """
        # Optional: sanitize field and operator inputs here if necessary
        if use_sql:
            query = f"SELECT * FROM {self.table_name} WHERE {field} {operator} :value"
            return self.pull_dataframe_from_rds(query)
        else:
            df = self.pull_dataframe_from_rds()
            df = df[df[field] == value].reset_index(drop=True)
            return df

    def create_table(self, table_name: str = None, columns: dict = None, if_exists: str = "fail"):
        """
        Create a new table in the RDS database.

        Args:
            table_name (str, optional): Name of the table to create. If None, uses self.table_name.
            columns (dict, optional): Dictionary defining column names and their SQL types.
                                    Format: {"column_name": "sql_type"}
                                    Example: {"id": "SERIAL PRIMARY KEY", "name": "VARCHAR(255)", "created_at": "TIMESTAMP"}
            if_exists (str): 'fail' (default), 'replace', or 'ignore'
        """
        if table_name is None:
            table_name = self.table_name

        self.table_name = table_name
        
        if columns is None:
            # Default columns for a basic table
            columns = {
                "id": "SERIAL PRIMARY KEY",
                "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                "updated_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            }
        
        # Build the CREATE TABLE SQL
        column_definitions = []
        for column_name, column_type in columns.items():
            column_definitions.append(f"{column_name} {column_type}")
        
        create_sql = f"CREATE TABLE {table_name} ({', '.join(column_definitions)})"
        
        try:
            with self.engine.connect() as conn:
                if if_exists == "replace":
                    # Drop table if it exists
                    conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
                    conn.commit()
                elif if_exists == "ignore":
                    # Check if table exists
                    result = conn.execute(text(f"""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = '{table_name}'
                        )
                    """))
                    if result.scalar():
                        logging.info(f"Table {table_name} already exists, skipping creation")
                        return
                
                conn.execute(text(create_sql))
                conn.commit()
                logging.info(f"Table {table_name} created successfully")
                
        except SQLAlchemyError as e:
            raise RuntimeError(f"Error creating table {table_name}: {e}")

    def create_table_from_dataframe(self, df: pd.DataFrame, table_name: str = None, if_exists: str = "fail"):
        """
        Create a table based on a pandas DataFrame structure.

        Args:
            df (pd.DataFrame): DataFrame to use as template for table structure
            table_name (str, optional): Name of the table to create. If None, uses self.table_name.
            if_exists (str): 'fail' (default), 'replace', or 'ignore'
        """
        if table_name is None:
            table_name = self.table_name
        
        # Map pandas dtypes to PostgreSQL types
        dtype_mapping = {
            'object': 'TEXT',
            'string': 'TEXT',
            'int64': 'BIGINT',
            'int32': 'INTEGER',
            'float64': 'DOUBLE PRECISION',
            'float32': 'REAL',
            'bool': 'BOOLEAN',
            'datetime64[ns]': 'TIMESTAMP',
            'category': 'TEXT'
        }
        
        columns = {}
        for column_name, dtype in df.dtypes.items():
            pg_type = dtype_mapping.get(str(dtype), 'TEXT')
            # Sanitize column names to be PostgreSQL-safe
            safe_column_name = column_name.replace(' ', '_').replace('-', '_').lower()
            columns[safe_column_name] = pg_type
        
        self.create_table(table_name, columns, if_exists)

    def drop_table(self, table_name: str = None, if_exists: bool = True):
        """
        Drop a table from the database.

        Args:
            table_name (str, optional): Name of the table to drop. If None, uses self.table_name.
            if_exists (bool): If True, adds IF EXISTS to prevent errors if table doesn't exist.
        """
        if table_name is None:
            table_name = self.table_name
        
        drop_sql = f"DROP TABLE {'IF EXISTS ' if if_exists else ''}{table_name}"
        
        try:
            with self.engine.connect() as conn:
                conn.execute(text(drop_sql))
                conn.commit()
                logging.info(f"Table {table_name} dropped successfully")
        except SQLAlchemyError as e:
            raise RuntimeError(f"Error dropping table {table_name}: {e}")

    def wipe_table(self, table_name: str = None, reset_sequence: bool = True) -> bool:
        """
        Wipe out all rows from a table while keeping the table structure intact.
        Optionally resets auto-increment sequences.

        Args:
            table_name (str, optional): Name of the table to wipe. If None, uses self.table_name.
            reset_sequence (bool): If True, resets auto-increment sequences to start from 1.

        Returns:
            bool: True if table was wiped successfully, False otherwise.
        """
        if table_name is None:
            table_name = self.table_name
        
        try:
            with self.engine.connect() as conn:
                # Delete all rows
                delete_sql = f"DELETE FROM {table_name}"
                conn.execute(text(delete_sql))
                
                # Reset auto-increment sequence if requested
                if reset_sequence:
                    # Get the primary key column (assuming it's auto-increment)
                    result = conn.execute(text(f"""
                        SELECT column_name 
                        FROM information_schema.columns 
                        WHERE table_name = '{table_name}' 
                        AND column_default LIKE 'nextval%'
                        LIMIT 1
                    """))
                    
                    sequence_col = result.scalar()
                    if sequence_col:
                        # Reset the sequence
                        reset_sql = f"""
                            SELECT setval(
                                pg_get_serial_sequence('{table_name}', '{sequence_col}'), 
                                1, 
                                false
                            )
                        """
                        conn.execute(text(reset_sql))
                
                conn.commit()
                logging.info(f"Table {table_name} wiped successfully")
                return True
                
        except SQLAlchemyError as e:
            logging.error(f"Error wiping table {table_name}: {e}")
            return False

    def truncate_table(self, table_name: str = None, cascade: bool = False) -> bool:
        """
        Truncate a table (faster than DELETE for large tables).
        This removes all rows and resets auto-increment sequences.

        Args:
            table_name (str, optional): Name of the table to truncate. If None, uses self.table_name.
            cascade (bool): If True, also truncate tables that reference this table.

        Returns:
            bool: True if table was truncated successfully, False otherwise.
        """
        if table_name is None:
            table_name = self.table_name
        
        try:
            with self.engine.connect() as conn:
                # TRUNCATE is faster than DELETE and automatically resets sequences
                truncate_sql = f"TRUNCATE TABLE {table_name}"
                if cascade:
                    truncate_sql += " CASCADE"
                
                conn.execute(text(truncate_sql))
                conn.commit()
                logging.info(f"Table {table_name} truncated successfully")
                return True
                
        except SQLAlchemyError as e:
            logging.error(f"Error truncating table {table_name}: {e}")
            return False

    def table_exists(self, table_name: str = None) -> bool:
        """
        Check if a table exists in the database.

        Args:
            table_name (str, optional): Name of the table to check. If None, uses self.table_name.

        Returns:
            bool: True if table exists, False otherwise.
        """
        if table_name is None:
            table_name = self.table_name
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(f"""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = '{table_name}'
                    )
                """))
                return result.scalar()
        except SQLAlchemyError as e:
            raise RuntimeError(f"Error checking if table {table_name} exists: {e}")

    def get_table_schema(self, table_name: str = None) -> dict:
        """
        Get the schema of a table.

        Args:
            table_name (str, optional): Name of the table. If None, uses self.table_name.

        Returns:
            dict: Dictionary with column names as keys and their data types as values.
        """
        if table_name is None:
            table_name = self.table_name
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(f"""
                    SELECT column_name, data_type, is_nullable, column_default
                    FROM information_schema.columns 
                    WHERE table_name = '{table_name}'
                    ORDER BY ordinal_position
                """))
                
                schema = {}
                for row in result:
                    schema[row[0]] = {
                        'data_type': row[1],
                        'is_nullable': row[2],
                        'column_default': row[3]
                    }
                return schema
        except SQLAlchemyError as e:
            raise RuntimeError(f"Error getting schema for table {table_name}: {e}")

    def get_table_columns(self, table_name: str = None) -> list:
        """
        Get a simple list of column names for a table.

        Args:
            table_name (str, optional): Name of the table. If None, uses self.table_name.

        Returns:
            list: List of column names.
        """
        if table_name is None:
            table_name = self.table_name
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(f"""
                    SELECT column_name
                    FROM information_schema.columns 
                    WHERE table_name = '{table_name}'
                    ORDER BY ordinal_position
                """))
                
                return [row[0] for row in result]
        except SQLAlchemyError as e:
            raise RuntimeError(f"Error getting columns for table {table_name}: {e}")

    def validate_row_data(self, row_data: dict, table_name: str = None) -> dict:
        """
        Validate row data against table schema and return only valid columns.

        Args:
            row_data (dict): Dictionary containing column names and values.
            table_name (str, optional): Name of the table. If None, uses self.table_name.

        Returns:
            dict: Dictionary with only valid columns that exist in the table.
        """
        if table_name is None:
            table_name = self.table_name
        
        # Get table columns
        table_columns = self.get_table_columns(table_name)
        
        # Filter row_data to only include columns that exist in the table
        valid_data = {}
        invalid_columns = []
        
        for col, value in row_data.items():
            # Sanitize column name to match table schema
            safe_col = col.replace(' ', '_').replace('-', '_').lower()
            
            if safe_col in table_columns:
                valid_data[safe_col] = value
            else:
                invalid_columns.append(f"{col} -> {safe_col}")
        
        if invalid_columns:
            logging.warning(f"Invalid columns for table {table_name}: {invalid_columns}")
        
        return valid_data
    
    def delete_rows(self, filter_query: str = None, field: str = None, value: any = None) -> int:
        """
        Delete rows from the table based on a raw SQL WHERE clause or field/value pair.

        Args:
            filter_query (str, optional): SQL WHERE clause (without 'WHERE').
            field (str, optional): Column name to filter by.
            value (any, optional): Value to match for the field.

        Returns:
            int: Number of rows deleted.
        """
        if filter_query is not None:
            delete_sql = f"DELETE FROM {self.table_name} WHERE {filter_query}"
            params = {}
        elif field is not None and value is not None:
            delete_sql = f"DELETE FROM {self.table_name} WHERE {field} = :value"
            params = {"value": value}
        else:
            raise ValueError("Must provide either filter_query or both field and value")
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(delete_sql), params)
                conn.commit()
                rowcount = result.rowcount
                logging.info(f"Deleted {rowcount} rows from table {self.table_name}")
                return rowcount
        except SQLAlchemyError as e:
            logging.error(f"Error deleting rows from table {self.table_name}: {e}")
            return 0
