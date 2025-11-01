from passlib.hash import bcrypt
import mysql.connector

# === Change these to match your MySQL setup ===
DB_HOST = "localhost"
DB_PORT = 3307   # adjust if needed
DB_USER = "root"
DB_PASS = ""     # your MySQL password
DB_NAME = "onlinevoting_system"

username = "tjame"
plain_pw = "Tjame19@Gmail.com"
full_name = "Tjame"
email = "tjame.topprograming@gmail.com"
role = "superadmin"

conn = mysql.connector.connect(
    host=DB_HOST,
    port=DB_PORT,
    user=DB_USER,
    password=DB_PASS,
    database=DB_NAME
)
cur = conn.cursor()

# create a proper bcrypt hash
pw_hash = bcrypt.hash(plain_pw)

# insert or update admin
cur.execute("""
    INSERT INTO admins (username, password_hash, full_name, email, role, created_at)
    VALUES (%s,%s,%s,%s,%s,NOW())
    ON DUPLICATE KEY UPDATE password_hash=VALUES(password_hash),
                            full_name=VALUES(full_name),
                            email=VALUES(email),
                            role=VALUES(role)
""", (username, pw_hash, full_name, email, role))

conn.commit()
cur.close()
conn.close()

print("✅ Admin created/updated")
print("Username:", username)
print("Password:", plain_pw)
