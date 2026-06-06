WITH raw_data AS (
    SELECT *
    FROM {{ source('raw_data', 'insurance_raw') }}
),
   cleaned AS (
    SELECT 
        age,
        CASE sex = 'female' WHEN TRUE THEN 1 ELSE 0 END AS sex,
        bmi,
        children,
        smoker::int AS smoker,
        CASE LOWER(region) = 'northeast' WHEN TRUE THEN 1 ELSE 0 END AS northeast,
        CASE LOWER(region) = 'northwest' WHEN TRUE THEN 1 ELSE 0 END AS northwest,
        CASE LOWER(region) = 'southeast' WHEN TRUE THEN 1 ELSE 0 END AS southeast,
        CASE LOWER(region) = 'southwest' WHEN TRUE THEN 1 ELSE 0 END AS southwest,
        charges
    FROM raw_data
)

SELECT *
FROM cleaned