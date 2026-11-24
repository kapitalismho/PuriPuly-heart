WITH buckets AS (
  SELECT version_id, ingress_colo, cohort, status, upper_ms,
         SUM(requests) AS n, MIN(day_utc) AS first_day, MAX(day_utc) AS last_day
    FROM puri_token_baseline
   WHERE day_utc BETWEEN date('now', '-6 days') AND date('now')
   GROUP BY version_id, ingress_colo, cohort, status, upper_ms
), ranked AS (
  SELECT *,
         SUM(n) OVER cohort_group AS total,
         SUM(n) OVER (cohort_group ORDER BY upper_ms = -1, upper_ms) AS cumulative
    FROM buckets
  WINDOW cohort_group AS (PARTITION BY version_id, ingress_colo, cohort, status)
)
SELECT version_id, ingress_colo, cohort, status,
       MIN(first_day) AS first_utc_day, MAX(last_day) AS last_utc_day, MAX(total) AS requests,
       MAX(CASE WHEN total >= 2 AND (cumulative - n) * 2 < total AND cumulative * 2 >= total THEN upper_ms END) AS p50_upper_ms,
       MAX(CASE WHEN total >= 20 AND (cumulative - n) * 100 < total * 95 AND cumulative * 100 >= total * 95 THEN upper_ms END) AS p95_upper_ms
  FROM ranked
 GROUP BY version_id, ingress_colo, cohort, status
 ORDER BY version_id, ingress_colo, cohort, status;
