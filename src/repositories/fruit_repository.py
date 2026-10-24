"""
File: fruit_repository.py
Project: Intelligent Fruit Classification and Information System
Description:
    Retrieves fruit records and their associated nutrition,
    health-benefit information, uses, fun facts, and source
    references from the SQLite database.

Author: Abeeha Mariam
Responsible Pillar: Pillar 3 - Database & Information System
Maintainers: Abeeha Mariam
"""
from src.repositories.database import get_connection



def get_fruit_by_model_label(model_label):
# Retrieve the fruit record that matches the class label returned by the ML model.
# model_label is unique, so only one record is expected.    
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute(
        "SELECT * FROM FRUIT WHERE model_label = ?", 
        (model_label,))
    fruit = cursor.fetchone()
    connection.close()
    return fruit


def get_nutrition_by_fruit_id(fruit_id):
# Retrieve the nutrition record for the selected fruit.
# Join with SOURCE so the application can display the nutrition source details.    
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute(
    """
    SELECT NUTRITION.*,
           SOURCE.source_name,
           SOURCE.source_url,
           SOURCE.access_date
    FROM NUTRITION
    JOIN SOURCE
        ON NUTRITION.source_id = SOURCE.source_id
    WHERE NUTRITION.fruit_id = ?
    """,
    (fruit_id,))
    nutrition = cursor.fetchone()
    connection.close()
    return nutrition


def get_uses_by_fruit_id(fruit_id):
# Retrieve all common uses associated with the selected fruit.
# A fruit can have multiple use records.    
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute(
        "SELECT * FROM USE WHERE fruit_id = ?", 
        (fruit_id,))
    uses = cursor.fetchall()
    connection.close()
    return uses


def get_health_benefits_by_fruit_id(fruit_id):
# Retrieve all health benefits associated with the selected fruit.
# Join with SOURCE because each health benefit has a supporting reference.
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute(
    """
    SELECT HEALTH_BENEFIT.benefit_description,
           SOURCE.source_name,
           SOURCE.source_url,
           SOURCE.access_date
    FROM HEALTH_BENEFIT
    JOIN SOURCE
        ON HEALTH_BENEFIT.source_id = SOURCE.source_id
    WHERE HEALTH_BENEFIT.fruit_id = ?
    """,
    (fruit_id,))
    health_benefits = cursor.fetchall()
    connection.close()
    return health_benefits


def get_fun_facts_by_fruit_id(fruit_id):
# Retrieve all fun facts associated with the selected fruit.
# Join with SOURCE so each fact can be displayed with its reference.
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute(
    """
    SELECT FUN_FACT.fact_text,
           SOURCE.source_name,
           SOURCE.source_url,
           SOURCE.access_date
    FROM FUN_FACT
    JOIN SOURCE
        ON FUN_FACT.source_id = SOURCE.source_id
    WHERE FUN_FACT.fruit_id = ?
    """,
    (fruit_id,))
    fun_facts = cursor.fetchall()
    connection.close()
    return fun_facts
