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
    python src/scripts/init_db.py            create the database (does nothing if it already exists)
    python src/scripts/init_db.py --reset    delete the database and rebuild it from schema.sql and
                                             seed_fruit.sql (this ERASES the prediction history)
"""
import argparse
import sqlite3
from pathlib import Path

# Define the src directory as the base location for database files.
BASE_DIR = Path(__file__).resolve().parent.parent

# Path to the SQL schema and seed data file
SCHEMA_PATH = BASE_DIR / "database" / "schema.sql"
SEED_PATH = BASE_DIR / "database" / "seed_fruit.sql"

# Path where the SQLite database will be created
DATABASE_PATH = BASE_DIR / "database" / "fruit_classifier.db"

parser = argparse.ArgumentParser(description="Create the local SQLite database.")
parser.add_argument(
    "--reset",
    action="store_true",
    help="delete an existing database first and rebuild it (erases the prediction history)",
)
args = parser.parse_args()

# The seed file uses plain INSERT statements, so running it twice on the same database fails with a
# primary-key error. Instead of failing, stop with a clear message unless the user asked for a reset.
if DATABASE_PATH.exists():
    if not args.reset:
        print(f"Database already exists: {DATABASE_PATH}")
        print("Nothing was changed. To rebuild it from scratch (erasing the prediction history), run:")
        print("    python src/scripts/init_db.py --reset")
        raise SystemExit(0)
    DATABASE_PATH.unlink()
    print("Existing database deleted.")

# Read the database schema
with open(SCHEMA_PATH, "r", encoding="utf-8") as file:
    schema_sql = file.read()

# Read the seed data
with open(SEED_PATH, "r", encoding="utf-8") as file:
    seed_sql = file.read()

# Connect to the SQLite database
connection = sqlite3.connect(DATABASE_PATH)

# Execute all SQL commands from schema.sql
connection.executescript(schema_sql)

# Insert the initial fruit information
connection.executescript(seed_sql)

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