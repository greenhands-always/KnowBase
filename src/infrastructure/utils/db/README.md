# PostgreSQL Connection Utility

This utility provides a simple way to connect to a PostgreSQL database and execute queries.

## Usage

1.  **Set up your environment variables:**

    Create a `.env` file in the root of your project and add the following variables:

    ```
    POSTGRES_HOST=your_host
    POSTGRES_PORT=your_port
    POSTGRES_USER=your_user
    POSTGRES_PASSWORD=your_password
    POSTGRES_DB=your_database
    ```

2.  **Use the `PostgresConnector` in your code:**

    ```python
    from src.infrastructure.utils.db.PostgresConnector import PostgresConnector

    # Create a connector instance
    db_connector = PostgresConnector()

    # Connect to the database
    conn = db_connector.connect()

    if conn:
        # Execute a query
        cursor = db_connector.execute_query("SELECT * FROM your_table;")
        if cursor:
            for row in cursor:
                print(row)

        # Close the connection
        db_connector.close()
    ```
