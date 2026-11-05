from src.repositories.fruit_repository import (
    get_fruit_by_model_label,
    get_nutrition_by_fruit_id,
    get_uses_by_fruit_id,
    get_health_benefits_by_fruit_id,
    get_fun_facts_by_fruit_id,
)


# ============================================
# APPLE REPOSITORY TESTS
# ============================================

# Verify that the Apple record can be retrieved using its model label.
def test_get_apple_by_model_label():
    fruit = get_fruit_by_model_label("apple")

    assert fruit is not None
    assert fruit["model_label"] == "apple"
    assert fruit["fruit_name"] == "Apple"


# Verify that nutrition information is available for Apple.
def test_get_apple_nutrition():
    nutrition = get_nutrition_by_fruit_id(1)

    assert nutrition is not None
    assert nutrition["fruit_id"] == 1


# Verify that Apple has one or more stored uses with descriptions.
def test_get_apple_uses():
    uses = get_uses_by_fruit_id(1)

    assert uses is not None
    assert len(uses) > 0

    for use in uses:
        assert use["fruit_id"] == 1
        assert use["use_description"] is not None


# Verify that Apple health benefits and their source information are returned.
def test_get_apple_health_benefits():
    benefits = get_health_benefits_by_fruit_id(1)

    assert benefits is not None
    assert len(benefits) > 0

    for benefit in benefits:
        assert benefit["benefit_description"] is not None
        assert benefit["source_name"] is not None
        assert benefit["source_url"] is not None
        assert benefit["access_date"] is not None


# Verify that Apple fun facts and their source information are returned.
def test_get_apple_fun_facts():
    facts = get_fun_facts_by_fruit_id(1)

    assert facts is not None
    assert len(facts) > 0

    for fact in facts:
        assert fact["fact_text"] is not None
        assert fact["source_name"] is not None
        assert fact["source_url"] is not None
        assert fact["access_date"] is not None


# ============================================
# BANANA REPOSITORY TESTS
# ============================================

# Verify that the Banana record can be retrieved using its model label.
def test_get_banana_by_model_label():
    fruit = get_fruit_by_model_label("banana")

    assert fruit is not None
    assert fruit["model_label"] == "banana"
    assert fruit["fruit_name"] == "Banana"


# Verify that nutrition information is available for Banana.
def test_get_banana_nutrition():
    nutrition = get_nutrition_by_fruit_id(2)

    assert nutrition is not None
    assert nutrition["fruit_id"] == 2


# Verify that Banana has one or more stored uses with descriptions.
def test_get_banana_uses():
    uses = get_uses_by_fruit_id(2)

    assert uses is not None
    assert len(uses) > 0

    for use in uses:
        assert use["fruit_id"] == 2
        assert use["use_description"] is not None


# Verify that Banana health benefits and their source information are returned.
def test_get_banana_health_benefits():
    benefits = get_health_benefits_by_fruit_id(2)

    assert benefits is not None
    assert len(benefits) > 0

    for benefit in benefits:
        assert benefit["benefit_description"] is not None
        assert benefit["source_name"] is not None
        assert benefit["source_url"] is not None
        assert benefit["access_date"] is not None


# Verify that Banana fun facts and their source information are returned.
def test_get_banana_fun_facts():
    facts = get_fun_facts_by_fruit_id(2)

    assert facts is not None
    assert len(facts) > 0

    for fact in facts:
        assert fact["fact_text"] is not None
        assert fact["source_name"] is not None
        assert fact["source_url"] is not None
        assert fact["access_date"] is not None