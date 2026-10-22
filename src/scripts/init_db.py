"""
File: init_db.py
Project: Intelligent Fruit Classification and Information System
Description:
    Initialises the local SQLite database by creating the schema
    and inserting the agreed initial fruit-information records.

Author: Abeeha Mariam
Responsible Pillar: Pillar 3 - Database & Information System
Maintainers: Abeeha Mariam

Usage:
    python src/scripts/init_db.py
"""
import sqlite3
from pathlib import Path
# Get the main project folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Path to the SQL schema file
SCHEMA_PATH = BASE_DIR / "database" / "schema.sql"

# Path where the SQLite database will be created
DATABASE_PATH = BASE_DIR / "database" / "fruit_classifier.db"
# Read the database schema
with open(SCHEMA_PATH, "r", encoding="utf-8") as file:
    schema_sql = file.read()
# Connect to the SQLite database
connection = sqlite3.connect(DATABASE_PATH)

# Execute all SQL commands from schema.sql
connection.executescript(schema_sql)

# Save the changes and close the connection
connection.commit()
connection.close()

print("Database created successfully.")
# Verify the tables created in the database
connection = sqlite3.connect(DATABASE_PATH)

cursor = connection.cursor()

cursor.execute("""
    SELECT name
    FROM sqlite_master
    WHERE type='table'
    ORDER BY name;
""")

tables = cursor.fetchall()

print("Tables in database:")
for table in tables:
    print(table[0])

connection.close()
