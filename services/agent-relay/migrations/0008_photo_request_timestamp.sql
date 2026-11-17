ALTER TABLE photo_admissions ADD COLUMN requested_at INTEGER;
CREATE INDEX photo_admissions_requested_date ON photo_admissions (COALESCE(requested_at, period_start));
