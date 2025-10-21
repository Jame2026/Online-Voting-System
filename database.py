import mysql.connector

def get_db_connection():
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",  # your MySQL password
        database="online_voting_system",
        port=3307
    )
    return connection
