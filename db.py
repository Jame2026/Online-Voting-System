import mysql.connector
from mysql.connector import Error

def get_db_connection():

    connection = None
    try:
        # Connection parameters based on a typical local setup (localhost:3307)
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",      # <<< REPLACE with your actual MySQL password if one is set
            database="onlinevoting_system",
            port=3307         # Port 3307 is often used by XAMPP/WAMP
        )
        if connection.is_connected():
            # In a final application, you might remove this print statement
            # print("Successfully connected to MySQL Database: online_votingsystem")
            return connection
        
    except Error as e:
        # Prints a user-friendly error message if the connection fails
        print(f"Error connecting to MySQL: {e}")
        # Return None if connection failed
        return None
