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



all all all
WITH ordered AS (
    SELECT suite_name, test_name, response_time/1000 as response_time, max_threads as test,
           ROW_NUMBER() OVER (PARTITION BY suite_name, test_name ORDER BY response_time/1000) AS rn,
           COUNT(*) OVER (PARTITION BY suite_name, test_name) AS cnt
    FROM performance
)
SELECT suite_name, test, test_name,
       MIN(response_time) AS min_response_time,
       MAX(response_time) AS max_response_time,
       AVG(response_time) AS avg_response_time,
       (SELECT response_time FROM ordered o2 
        WHERE o2.suite_name = o1.suite_name AND o2.test_name = o1.test_name AND rn = (cnt + 1)/2) AS median_response_time,
        (SELECT response_time FROM ordered o2 
        WHERE o2.suite_name = o1.suite_name AND o2.test_name = o1.test_name AND rn = CAST(0.1 * cnt AS INT)) AS percentile_10,
        (SELECT response_time FROM ordered o2 
        WHERE o2.suite_name = o1.suite_name AND o2.test_name = o1.test_name AND rn = CAST(0.2 * cnt AS INT)) AS percentile_20,
        (SELECT response_time FROM ordered o2 
        WHERE o2.suite_name = o1.suite_name AND o2.test_name = o1.test_name AND rn = CAST(0.3 * cnt AS INT)) AS percentile_30,
        (SELECT response_time FROM ordered o2 
        WHERE o2.suite_name = o1.suite_name AND o2.test_name = o1.test_name AND rn = CAST(0.4 * cnt AS INT)) AS percentile_40,
        (SELECT response_time FROM ordered o2 
        WHERE o2.suite_name = o1.suite_name AND o2.test_name = o1.test_name AND rn = CAST(0.5 * cnt AS INT)) AS percentile_50,
        (SELECT response_time FROM ordered o2 
        WHERE o2.suite_name = o1.suite_name AND o2.test_name = o1.test_name AND rn = CAST(0.6 * cnt AS INT)) AS percentile_60,
        (SELECT response_time FROM ordered o2 
        WHERE o2.suite_name = o1.suite_name AND o2.test_name = o1.test_name AND rn = CAST(0.7 * cnt AS INT)) AS percentile_70,
        (SELECT response_time FROM ordered o2 
        WHERE o2.suite_name = o1.suite_name AND o2.test_name = o1.test_name AND rn = CAST(0.8 * cnt AS INT)) AS percentile_80,
       (SELECT response_time FROM ordered o2 
        WHERE o2.suite_name = o1.suite_name AND o2.test_name = o1.test_name AND rn = CAST(0.9 * cnt AS INT)) AS percentile_90,
       (SELECT response_time FROM ordered o2 
        WHERE o2.suite_name = o1.suite_name AND o2.test_name = o1.test_name AND rn = CAST(0.95 * cnt AS INT)) AS percentile_95,
       (SELECT response_time FROM ordered o2 
        WHERE o2.suite_name = o1.suite_name AND o2.test_name = o1.test_name AND rn = CAST(0.99 * cnt AS INT)) AS percentile_99
FROM ordered o1
GROUP BY suite_name, test, test_name
order by test 
