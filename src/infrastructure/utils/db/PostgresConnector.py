import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

class PostgresConnector:
    def __init__(self):
        self.host = os.getenv("DB_HOST")
        self.port = os.getenv("DB_PORT")
        self.user = os.getenv("DB_USER")
        self.password = os.getenv("DB_PASSWORD")
        self.dbname = os.getenv("DB_NAME")
        self.conn = None

    def connect(self):
        if self.conn is None or self.conn.closed:
            try:
                self.conn = psycopg2.connect(
                    host=self.host,
                    port=self.port,
                    user=self.user,
                    password=self.password,
                    dbname=self.dbname
                )
            except psycopg2.OperationalError as e:
                print(f"Error connecting to PostgreSQL: {e}")
                self.conn = None
        return self.conn

    def close(self):
        if self.conn and not self.conn.closed:
            self.conn.close()
            self.conn = None

    def execute_query(self, query, params=None):
        conn = self.connect()
        if not conn:
            return None
        
        cursor = conn.cursor()
        try:
            cursor.execute(query, params)
            conn.commit()
            return cursor
        except Exception as e:
            conn.rollback()
            print(f"Error executing query: {e}")
            return None
        finally:
            cursor.close()
