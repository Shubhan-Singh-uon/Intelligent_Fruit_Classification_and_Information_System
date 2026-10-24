"""
File: database.py
Project: Intelligent Fruit Classification and Information System
Description:
    Provides SQLite connection management and shared database
    access functionality for the application's repositories.

Author: Abeeha Mariam
Responsible Pillar: Pillar 3 - Database & Information System
Maintainers: Abeeha Mariam
"""
import sqlite3
from pathlib import Path



# Define the path to the SQLite database relative to the src directory.
# Using pathlib keeps the database path consistent across different computers.
BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_PATH = BASE_DIR / "database" / "fruit_classifier.db"


# Create and return a connection to the SQLite database.
# sqlite3.Row allows query results to be accessed using column names.
# Foreign-key enforcement is enabled for each new SQLite connection.
def get_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON;")
    return connection
