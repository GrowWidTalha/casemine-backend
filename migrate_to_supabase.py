import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import pandas as pd

# Database connection strings
NEON_URL = "postgresql://saasdb_owner:hZm1Ql3RgJjs@ep-holy-voice-a2p4hd0z-pooler.eu-central-1.aws.neon.tech/saasdb?sslmode=require"
SUPABASE_URL = "postgresql://postgres:xpBS15qHiEWtm1pi@db.ciojbynixikujryugtvs.supabase.co:5432/postgres"

def get_all_tables(connection):
    """Get all table names from the database"""
    cursor = connection.cursor()
    cursor.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
    """)
    return [table[0] for table in cursor.fetchall()]

def get_table_schema(connection, table_name):
    """Get the complete schema for a table"""
    cursor = connection.cursor()
    cursor.execute(f"""
        SELECT column_name, data_type, character_maximum_length,
               column_default, is_nullable
        FROM information_schema.columns
        WHERE table_name = '{table_name}'
        ORDER BY ordinal_position
    """)
    return cursor.fetchall()

def get_table_data(connection, table_name):
    """Get all data from a table"""
    cursor = connection.cursor()
    cursor.execute(f"SELECT * FROM {table_name}")
    return cursor.fetchall()

def get_foreign_keys(connection, table_name):
    """Get foreign key constraints for a table"""
    cursor = connection.cursor()
    cursor.execute("""
        SELECT
            tc.constraint_name,
            kcu.column_name,
            ccu.table_name AS foreign_table_name,
            ccu.column_name AS foreign_column_name
        FROM
            information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
              ON tc.constraint_name = kcu.constraint_name
            JOIN information_schema.constraint_column_usage AS ccu
              ON ccu.constraint_name = tc.constraint_name
        WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_name = %s
    """, (table_name,))
    return cursor.fetchall()

def create_foreign_keys(connection, table_name, foreign_keys):
    """Create foreign key constraints"""
    cursor = connection.cursor()
    for fk in foreign_keys:
        constraint_name, column_name, foreign_table, foreign_column = fk
        try:
            cursor.execute(f"""
                ALTER TABLE {table_name}
                ADD CONSTRAINT {constraint_name}
                FOREIGN KEY ({column_name})
                REFERENCES {foreign_table}({foreign_column})
            """)
        except Exception as e:
            print(f"Warning: Could not create foreign key {constraint_name}: {e}")
    connection.commit()

def create_custom_types(connection):
    """Create custom types in the target database"""
    cursor = connection.cursor()
    try:
        # Grant necessary permissions
        cursor.execute("""
            GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO postgres;
            GRANT ALL ON ALL TABLES IN SCHEMA public TO postgres;
        """)

        # Create CourtType enum
        cursor.execute("""
            DO $$
            BEGIN
                IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'courttype') THEN
                    CREATE TYPE courttype AS ENUM ('supreme_court', 'high_court', 'tribunal');
                END IF;
            END $$;
        """)
        connection.commit()
    except Exception as e:
        print(f"Error creating custom types: {e}")
        connection.rollback()
        raise

def create_table_in_target(connection, table_name, schema):
    """Create table in target database"""
    cursor = connection.cursor()

    try:
        # First, create sequences if they don't exist
        cursor.execute(f"""
            DO $$
            BEGIN
                IF NOT EXISTS (SELECT 1 FROM pg_sequences WHERE sequencename = '{table_name}_id_seq') THEN
                    CREATE SEQUENCE {table_name}_id_seq;
                END IF;
            END $$;
        """)

        # Build CREATE TABLE statement
        columns = []
        for col in schema:
            col_name, data_type, max_length, default, nullable = col

            # Handle special cases
            if col_name == 'id' and data_type in ('integer', 'bigint'):
                col_def = f"{col_name} {data_type} DEFAULT nextval('{table_name}_id_seq')"
            elif data_type == 'ARRAY':
                # Handle array types - default to float array if not specified
                col_def = f"{col_name} float[]"
            elif data_type == 'USER-DEFINED':
                # Handle custom types
                if 'court_type' in col_name.lower():
                    col_def = f"{col_name} courttype"
                else:
                    col_def = f"{col_name} text"  # Default to text if unknown
            else:
                col_def = f"{col_name} {data_type}"
                if max_length:
                    col_def += f"({max_length})"
                if default and 'nextval' not in default:
                    col_def += f" DEFAULT {default}"
                if nullable == 'NO':
                    col_def += " NOT NULL"

            columns.append(col_def)

        # Handle reserved words
        if table_name.lower() == 'user':
            table_name = '"user"'  # Quote the table name

        # Drop table if exists
        cursor.execute(f'DROP TABLE IF EXISTS {table_name} CASCADE')

        # Create table
        create_table_sql = f"""
        CREATE TABLE {table_name} (
            {', '.join(columns)}
        )
        """

        cursor.execute(create_table_sql)

        # Set sequence ownership
        cursor.execute(f"ALTER SEQUENCE {table_name}_id_seq OWNED BY {table_name}.id")

        connection.commit()
        print(f"Successfully created table {table_name}")

    except Exception as e:
        print(f"Error creating table {table_name}: {e}")
        connection.rollback()
        raise

def create_sequences(connection, tables):
    """Create sequences for all tables"""
    cursor = connection.cursor()
    try:
        for table in tables:
            cursor.execute(f"""
                DO $$
                BEGIN
                    IF NOT EXISTS (SELECT 1 FROM pg_sequences WHERE sequencename = '{table}_id_seq') THEN
                        CREATE SEQUENCE {table}_id_seq;
                    END IF;
                END $$;
            """)
        connection.commit()
    except Exception as e:
        print(f"Error creating sequences: {e}")
        connection.rollback()
        raise

def migrate_complete_database():
    # Connect to source database
    source_conn = psycopg2.connect(NEON_URL)
    source_conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

    # Connect to target database
    target_conn = psycopg2.connect(SUPABASE_URL)
    target_conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

    # Define tables to skip (tables we don't want to migrate)
    tables_to_skip = {
        'amicus', 'attorney_profiles', 'blacklisted_tokens',
        'cases', 'contact', 'contact_messages', 'documents', 'folders',
        'languages_known', 'practice_areas', 'practice_courts',
        'professional_memberships', 'professional_experience', 'prompts', 'users',
        "user", "profiles", "case_documents", 'articles', "article_chunks", "article2"
    }

    try:
        # Create custom types first
        create_custom_types(target_conn)

        # Get all tables
        all_tables = get_all_tables(source_conn)

        # Filter out tables to skip
        tables = [table for table in all_tables if table not in tables_to_skip]

        print(f"Found {len(tables)} tables to migrate (excluding {len(tables_to_skip)} skipped tables)")
        print("Tables to migrate:", tables)

        # Create sequences for all tables first
        create_sequences(target_conn, tables)

        # Track successful migrations
        successful_tables = []

        # First pass: Create all tables
        for table in tables:
            try:
                print(f"\nMigrating table: {table}")
                schema = get_table_schema(source_conn, table)
                create_table_in_target(target_conn, table, schema)
                successful_tables.append(table)
            except Exception as e:
                print(f"Failed to create table {table}: {e}")
                continue

        # Second pass: Create foreign keys
        for table in successful_tables:
            try:
                foreign_keys = get_foreign_keys(source_conn, table)
                if foreign_keys:
                    create_foreign_keys(target_conn, table, foreign_keys)
            except Exception as e:
                print(f"Failed to create foreign keys for table {table}: {e}")
                continue

        # Third pass: Migrate data
        for table in successful_tables:
            try:
                print(f"\nMigrating data for table: {table}")
                data = get_table_data(source_conn, table)
                if data:
                    # Get column names
                    cursor = source_conn.cursor()
                    cursor.execute(f"SELECT * FROM {table} LIMIT 0")
                    columns = [desc[0] for desc in cursor.description]

                    # Create DataFrame for easier handling
                    df = pd.DataFrame(data, columns=columns)

                    # Insert data in batches
                    batch_size = 1000
                    for i in range(0, len(df), batch_size):
                        batch = df.iloc[i:i + batch_size]
                        batch.to_sql(
                            table,
                            create_engine(SUPABASE_URL),
                            if_exists='append',
                            index=False
                        )
                        print(f"Migrated {min(i + batch_size, len(df))}/{len(df)} rows")

                print(f"Completed migration of table: {table}")
            except Exception as e:
                print(f"Failed to migrate data for table {table}: {e}")
                continue

        print("\nDatabase migration completed!")
        print(f"Successfully migrated {len(successful_tables)} out of {len(tables)} tables")
        print("Successfully migrated tables:", successful_tables)

    except Exception as e:
        print(f"Error during migration: {e}")
        target_conn.rollback()
    finally:
        source_conn.close()
        target_conn.close()

def verify_migration():
    """Verify the migration by comparing row counts"""
    source_conn = psycopg2.connect(NEON_URL)
    target_conn = psycopg2.connect(SUPABASE_URL)

    # Tables to skip
    tables_to_skip = {
        'amicus', 'articles', 'attorney_profiles', 'blacklisted_tokens',
        'cases', 'contact', 'contact_messages', 'documents', 'folders',
        'languages_known', 'practice_areas', 'practice_courts',
        'professional_experience', 'professional_members', 'prompts', 'users'
    }

    try:
        all_tables = get_all_tables(source_conn)
        tables = [table for table in all_tables if table not in tables_to_skip]
        print("\nVerifying migration:")

        for table in tables:
            try:
                # Get row counts
                source_cursor = source_conn.cursor()
                target_cursor = target_conn.cursor()

                source_cursor.execute(f"SELECT COUNT(*) FROM {table}")
                target_cursor.execute(f"SELECT COUNT(*) FROM {table}")

                source_count = source_cursor.fetchone()[0]
                target_count = target_cursor.fetchone()[0]

                print(f"{table}:")
                print(f"Source count: {source_count}")
                print(f"Target count: {target_count}")
                print(f"Match: {source_count == target_count}\n")
            except Exception as e:
                print(f"Error verifying table {table}: {e}")
                continue

    finally:
        source_conn.close()
        target_conn.close()

def migrate_articles_data():
    # Connect to source database
    source_conn = psycopg2.connect(NEON_URL)
    source_conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

    # Connect to target database
    target_conn = psycopg2.connect(SUPABASE_URL)
    target_conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

    try:
        print("\nMigrating data for articles table...")

        # Get data from source
        data = get_table_data(source_conn, 'articles')
        if data:
            # Get column names
            cursor = source_conn.cursor()
            cursor.execute("SELECT * FROM articles LIMIT 0")
            columns = [desc[0] for desc in cursor.description]

            # Create DataFrame for easier handling
            df = pd.DataFrame(data, columns=columns)

            # Insert data in batches
            batch_size = 1000
            total_rows = len(df)
            print(f"Total rows to migrate: {total_rows}")

            for i in range(0, total_rows, batch_size):
                batch = df.iloc[i:i + batch_size]
                batch.to_sql(
                    'articles',
                    create_engine(SUPABASE_URL),
                    if_exists='append',
                    index=False
                )
                print(f"Migrated {min(i + batch_size, total_rows)}/{total_rows} rows")

            print("Completed migration of articles table!")

            # Verify the migration
            source_cursor = source_conn.cursor()
            target_cursor = target_conn.cursor()

            source_cursor.execute("SELECT COUNT(*) FROM articles")
            target_cursor.execute("SELECT COUNT(*) FROM articles")

            source_count = source_cursor.fetchone()[0]
            target_count = target_cursor.fetchone()[0]

            print("\nVerification:")
            print(f"Source count: {source_count}")
            print(f"Target count: {target_count}")
            print(f"Match: {source_count == target_count}")

    except Exception as e:
        print(f"Error during migration: {e}")
        target_conn.rollback()
    finally:
        source_conn.close()
        target_conn.close()

if __name__ == "__main__":
    # print("Starting database migration...")
    migrate_complete_database()
    print("\nVerifying migration...")
    verify_migration()
    # print("\nStarting articles data migration...")
    # migrate_articles_data()
