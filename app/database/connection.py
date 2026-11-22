import os

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector


def get_database_url(variable_name: str = "DATABASE_URL") -> str:
    # load_dotenv() reads .env but never overrides variables that are already set.
    load_dotenv()

    url = os.getenv(variable_name)
    if not url:
        raise RuntimeError(
            f"{variable_name} is not set. Copy .env.example to .env and fill in the passwords."
        )
    return url


def get_connection(variable_name: str = "DATABASE_URL") -> psycopg.Connection:
    connection = psycopg.connect(get_database_url(variable_name))

    try:
        # Lets us pass Python lists or numpy arrays as pgvector "vector" values.
        register_vector(connection)
    except psycopg.ProgrammingError:
        connection.close()
        raise RuntimeError(
            "The pgvector extension is not enabled in this database. "
            "Run: CREATE EXTENSION IF NOT EXISTS vector;"
        )

    return connection


def require_current_user(connection: psycopg.Connection, expected_user: str) -> None:
    """Stop if the connection is not logged in as the expected database role."""
    current_user = connection.execute("SELECT current_user").fetchone()[0]
    if current_user != expected_user:
        raise RuntimeError(f"Connected as {current_user}, but this tool must run as {expected_user}.")


def check_database(variable_name: str = "DATABASE_URL") -> dict:
    """Connect, run one read-only query and report what we are connected to."""
    with get_connection(variable_name) as connection:
        row = connection.execute(
            """
            SELECT current_database(),
                   current_user,
                   current_setting('server_version'),
                   (SELECT extversion FROM pg_extension WHERE extname = 'vector')
            """
        ).fetchone()

    return {
        "database": row[0],
        "user": row[1],
        "postgresql_version": row[2],
        "pgvector_version": row[3],
    }
