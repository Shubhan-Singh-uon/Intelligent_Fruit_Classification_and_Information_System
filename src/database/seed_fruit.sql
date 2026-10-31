-- File: seed_fruit.sql
-- Project: Intelligent Fruit Classification and Information System
-- Description:
--     Inserts the agreed initial fruit-information records and
--     supporting source records into the SQLite database.
--
-- Author(s): Group 8
-- Database Implementation Lead: Abeeha Mariam
-- Content Contributors: Wajahat Manzoor, Shubhan Singh, Abeeha Mariam, Hasnain Younas
-- Responsible Pillar: Pillar 3 - Database & Information System
--
-- Note:
--     All four members contribute and verify content across the
--     applicable tables. Prediction history is populated at runtime.
-- ============================================
-- 1. FRUIT
-- ============================================
INSERT INTO FRUIT (fruit_id, fruit_name, model_label, description) VALUES
    (1, 'Apple', 'apple', 'An apple is a round, edible fruit covered  with smooth skin in shades of red, green, or yellow and a crisp, whitish interior.'),
    (2, 'Banana', 'banana', 'A banana is an elongated, edible fruit with a soft white flesh covered by a thick, removable peel in shades of yellow or green.');

-- ============================================
-- 2. SOURCE
-- ============================================
INSERT INTO SOURCE (source_id, source_name, source_url, access_date) VALUES
    (1, 'F000110: Apple, red skin, unpeeled, raw', 'https://www.foodstandards.gov.au/science-data/food-nutrient-databases/afcd/search/food/F000110', '2026-10-16'),
    (2, 'F000262: Banana, cavendish, peeled, raw', 'https://www.foodstandards.gov.au/science-data/food-nutrient-databases/afcd/search/food/F000262', '2026-10-16'),
    (3, 'St. Vincent''s Health System- 5 Reasons to Eat More Apples', 'https://stvincents.org/about-us/news-press/news-detail?articleId=59557&publicid=395', '2026-10-16'),
    (4, 'Harvard Health Publishing – 11 Foods That Lower Cholesterol', 'https://www.health.harvard.edu/heart-health/11-foods-that-lower-cholesterol-4BTQX4EV', '2026-10-16'),
    (5, 'Harvard T.H. Chan School of Public Health – Apples', 'https://nutritionsource.hsph.harvard.edu/food-features/apples/', '2026-10-16'),
    (6, 'Harvard T.H. Chan School of Public Health – Bananas', 'https://nutritionsource.hsph.harvard.edu/food-features/bananas/', '2026-10-16'),
    (7, 'Cleveland Clinic-Peel It Back: Let’s Look at the Health Benefits of Bananas', 'https://health.clevelandclinic.org/benefits-of-bananas', '2026-10-16'),
    (8, 'Canadian Geographic-Canada’s Long History of Apple Growing', 'https://canadiangeographic.ca/articles/canadas-long-history-of-apple-growing/', '2026-10-16'),
    (9, 'Wikipedia-Rosaceae', 'https://en.wikipedia.org/wiki/Rosaceae', '2026-10-16'),
    (10, 'Washington State University (WSU) Tree Fruit-Varrieties Apple', 'https://treefruit.wsu.edu/web-article/apple-varieties/', '2026-10-16'),
    (11, 'BBC-The birthplace of the modern apple', 'https://www.bbc.com/travel/article/20181120-the-birthplace-of-the-modern-apple', '2026-10-16'),
    (12, 'Live Science-Why are bananas berries but strawberries aren''t?', 'https://www.livescience.com/57477-why-are-bananas-considered-berries.html', '2026-10-16'),
    (13, 'Michigan State Univeristy -Plant science at the dinner table: bananas', 'https://www.canr.msu.edu/news/plant-science-at-the-dinner-table-bananas', '2026-10-16'),
    (14, 'The Guardian', 'https://www.theguardian.com/commentisfree/2024/mar/22/1000-varieties-banana-lack-of-diversity-extinction', '2026-10-16'),
    (15, 'Australian Bananas-Mind Bending Facts', 'https://www.australianbananas.com.au/pages/all-about-bananas/mind-bending-facts', '2026-10-16'),
    (16, 'City of Ryde - Granny Smith', 'https://www.ryde.nsw.gov.au/Library/Local-and-Family-History/Historic-Ryde/Historic-Figures/Granny-Smith', '2026-10-30'),
    (17, 'University of Vermont Extension - All About Apples', 'https://www.uvm.edu/extension/news/all-about-apples', '2026-10-30'),
    (18, 'Kew Gardens - Cavendish Banana', 'https://www.kew.org/plants/cavendish-banana', '2026-10-30'),
    (19, 'San Diego Zoo - Banana', 'https://animals.sandiegozoo.org/plants/banana', '2026-10-30'),
    (20, 'University of Saskatchewan - Apple Pollination', 'https://gardening.usask.ca/gardening-advice/gardenline-nested-pages/food-plant-pages/fruit/apples.php', '2026-10-30'),
    (21, 'USDA - Apple Nutrition and Fun Facts', 'https://www.govinfo.gov/content/pkg/GOVPUB-A98-PURL-gpo87472/pdf/GOVPUB-A98-PURL-gpo87472.pdf', '2026-10-30'),
    (22, 'Texas A&M AgriLife Extension - Banana', 'https://aggie-horticulture.tamu.edu/fruit-nut/fact-sheets/banana/', '2026-10-30'),
    (23, 'New York Botanical Garden - Banana', 'https://www.nybg.org/around-the-table/banana/', '2026-10-30'),
    (24, 'Michigan State University Extension - Taste of the Tropics: Ten tropical fruits', 'https://msu-prod.dotcmscloud.com/news/taste-of-the-tropics-ten-tropical-fruits', '2026-10-29'),
    (25, 'Harvard T.H. Chan School of Public Health - Vitamin B6', 'https://nutritionsource.hsph.harvard.edu/vitamin-b6/', '2026-10-29');    
-- ============================================
-- 3. NUTRITION
-- ============================================
INSERT INTO NUTRITION (nutrition_id, fruit_id, source_id, serving_basis, energy_kcal, protein_g, carbohydrate_g, sugars_g, fibre_g, fat_g, potassium_mg, vitamin_c_mg) VALUES
    (1, 1, 1, '100g', 57.8, 0.2, 12.7, 12.4, 2.5, 0, 115, 7),
    (2, 2, 2, '100g', 95.4, 1.4, 19.9, 13.4, 2.2, 0.2, 348, 5);

-- ============================================
-- 4. USE
-- ============================================
INSERT INTO USE (use_id, fruit_id, use_description) VALUES
    (1, 1, 'Eat Fresh'),
    (2, 1, 'Baking'),
    (3, 1, 'Smoothies'),
    (4, 1, 'Juices'),
    (5, 2, 'Eat Fresh'),
    (6, 2, 'Baking'),
    (7, 2, 'Smoothies'),
    (8, 2, 'Desserts'),
    (9, 1, 'Spreads'),
    (10, 1, 'Salads'),
    (11, 2, 'Snacks'),
    (12, 2, 'Toppings');

-- ============================================
-- 5. HEALTH BENEFIT
-- ============================================
INSERT INTO HEALTH_BENEFIT (benefit_id, fruit_id, source_id, benefit_description) VALUES
    (1, 1, 3, 'Apples contain dietary fibre, which supports healthy digestion.'),
    (2, 1, 4, 'Apples contain fibre called pectin which can help lower cholesterol levels .'),
    (3, 1, 5, 'Research suggests apple consumption may have beneficial effects on some cardiovascular risk factors.'),
    (4, 2, 6, 'Bananas are rich in potassium, which supports normal heart function, muscle contraction, and healthy blood pressure.'),
    (5, 2, 7, 'Bananas provide carbohydrates that supply energy for physical activity, making them a convenient pre-workout snack.'),
    (6, 2, 6, 'Unripe bananas contain resistant starch, which acts as a prebiotic and helps nourish beneficial gut bacteria.'),
    (7, 1, 5, 'Observational research has associated apple consumption with a lower risk of type 2 diabetes.'),
    (8, 2, 25, 'Bananas provide vitamin B6, which supports enzyme reactions involved in protein, carbohydrate, and fat metabolism.');
    

-- ============================================
-- 6. FUN FACT
-- ============================================
INSERT INTO FUN_FACT (fact_id, fruit_id, source_id, fact_text) VALUES
    (1, 1, 8, 'The first apple tree in Canada was cultivated by early French settlers in Nova Scotia’s Annapolis Valley around 1633.'),
    (2, 1, 9, 'Apples belong to the rose family (Rosaceae), alongside pears, strawberries and peaches.'),
    (3, 1, 10, 'There are more than 7,500 apple varieties worldwide!'),
    (4, 1, 11, 'Kazakhstan is linked to the ancestry of modern domestic apples.The name of its largest city, Almaty, translates to "Father of Apples".'),
    (5, 2, 12, 'Bananas are botanically classified as berries, but strawberries are not true botanical berries!'),
    (6, 2, 13, 'The word “banana” comes from the Arabic word banan, meaning “finger.”'),
    (7, 2, 14, 'There are more than 1,000 different varieties of bananas worldwide, divided into about 50 groups'),
    (8, 2, 15, 'Once developed, instead of growing towards the ground, bananas turn towards the sun. The fruit continues growing against gravity, giving the banana its familiar curved shape.'),
    (9, 1, 16, 'The Granny Smith apple was discovered by Maria Ann Smith in 1868 in Eastwood, New South Wales, Australia.'),
    (10, 1, 17, 'Apple trees grown from seeds do not necessarily produce fruit identical to the parent apple tree. Commercial apple trees are therefore propagated through grafting.'),
    (11, 2, 18, 'Banana plants are giant herbs rather than trees. Their apparent trunks are called pseudostems and are formed from tightly packed leaf bases.'),
    (12, 2, 19, 'A banana plant generally produces around 40 leaves before reaching the fruiting stage.'),
    (13, 1, 20, 'Apple trees generally require pollen from another apple or crabapple variety to produce fruit successfully.'),
    (14, 1, 21, 'An apple has five seed chambers, also called carpels, located near its core.'),
    (15, 2, 22, 'Banana flowers grow in groups called hands along the flower stalk, and the female flowers develop into bananas.'),
    (16, 2, 23, 'Many wild bananas contain large, hard seeds, unlike the familiar cultivated bananas that are usually seedless.'),
    (17, 1, 17, 'An apple is about 25% air, which helps explain why apples float in water.'),
    (18, 1, 17, 'It takes about 40 apples to make one gallon of apple cider.'),
    (19, 2, 24, 'Cultivated bananas are seedless and are propagated through cloning.');
