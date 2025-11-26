select * from performance

WITH ordered AS (
    SELECT test_name, response_time,
           ROW_NUMBER() OVER (PARTITION BY test_name ORDER BY response_time) AS rn,
           COUNT(*) OVER (PARTITION BY test_name) AS cnt
    FROM performance
)
SELECT test_name,
       MIN(response_time) AS min_response_time,
       MAX(response_time) AS max_response_time,
       AVG(response_time) AS avg_response_time,
       (SELECT response_time FROM ordered o2 
        WHERE o2.test_name = o1.test_name AND rn = (cnt + 1)/2) AS median_response_time,
       (SELECT response_time FROM ordered o2 
        WHERE o2.test_name = o1.test_name AND rn = CAST(0.9 * cnt AS INT)) AS percentile_90,
       (SELECT response_time FROM ordered o2 
        WHERE o2.test_name = o1.test_name AND rn = CAST(0.95 * cnt AS INT)) AS percentile_95,
       (SELECT response_time FROM ordered o2 
        WHERE o2.test_name = o1.test_name AND rn = CAST(0.99 * cnt AS INT)) AS percentile_99
FROM ordered o1
GROUP BY test_name;



WITH ordered AS (
    SELECT suite_name, test_name, response_time,
           ROW_NUMBER() OVER (PARTITION BY suite_name, test_name ORDER BY response_time) AS rn,
           COUNT(*) OVER (PARTITION BY suite_name, test_name) AS cnt
    FROM performance
)
SELECT suite_name, test_name,
       MIN(response_time) AS min_response_time,
       MAX(response_time) AS max_response_time,
       AVG(response_time) AS avg_response_time,
       (SELECT response_time FROM ordered o2 
        WHERE o2.suite_name = o1.suite_name AND o2.test_name = o1.test_name AND rn = (cnt + 1)/2) AS median_response_time,
       (SELECT response_time FROM ordered o2 
        WHERE o2.suite_name = o1.suite_name AND o2.test_name = o1.test_name AND rn = CAST(0.9 * cnt AS INT)) AS percentile_90,
       (SELECT response_time FROM ordered o2 
        WHERE o2.suite_name = o1.suite_name AND o2.test_name = o1.test_name AND rn = CAST(0.95 * cnt AS INT)) AS percentile_95,
       (SELECT response_time FROM ordered o2 
        WHERE o2.suite_name = o1.suite_name AND o2.test_name = o1.test_name AND rn = CAST(0.99 * cnt AS INT)) AS percentile_99
FROM ordered o1
GROUP BY suite_name, test_name;



all

WITH ordered AS (
    SELECT response_time,
           ROW_NUMBER() OVER (ORDER BY response_time) AS rn,
           COUNT(*) OVER () AS cnt
    FROM performance
)
SELECT
    MIN(response_time) AS min_response_time,
    MAX(response_time) AS max_response_time,
    AVG(response_time) AS avg_response_time,
    (SELECT response_time FROM ordered WHERE rn = (cnt + 1)/2) AS median_response_time,
    (SELECT response_time FROM ordered WHERE rn = CAST(0.9 * cnt AS INT)) AS percentile_90,
    (SELECT response_time FROM ordered WHERE rn = CAST(0.95 * cnt AS INT)) AS percentile_95,
    (SELECT response_time FROM ordered WHERE rn = CAST(0.99 * cnt AS INT)) AS percentile_99
FROM performance;