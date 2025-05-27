@echo off
:: ========================================
:: Neon → Supabase DB Migration Script (.bat)
:: Requires: pg_dump and psql in PATH
:: Migrates only judgments and acts tables
:: ========================================

:: === Step 1: Set your database URLs ===
:: Replace the placeholders below with actual values

set OLD_DB_URL=postgresql://saasdb_owner:hZm1Ql3RgJjs@ep-holy-voice-a2p4hd0z-pooler.eu-central-1.aws.neon.tech/saasdb?sslmode=require
set NEW_DB_URL=postgresql://postgres:xpBS15qHiEWtm1pi@db.ciojbynixikujryugtvs.supabase.co:5432/postgres

:: === Step 2: Confirm values ===
echo Migrating judgments and acts tables from:
echo %OLD_DB_URL%
echo to:
echo %NEW_DB_URL%
pause

:: === Step 3: Export from Neon ===
echo Dumping judgments and acts tables from Neon DB to dump.sql...
pg_dump "%OLD_DB_URL%" ^
  --clean ^
  --if-exists ^
  --quote-all-identifiers ^
  --no-owner ^
  --no-privileges ^
  --table=acts ^
  -f dump.sql

IF %ERRORLEVEL% NEQ 0 (
  echo Failed to export tables from Neon.
  pause
  exit /b %ERRORLEVEL%
)

:: === Step 4: Import to Supabase ===
echo Importing judgments and acts tables to Supabase...
psql -d "%NEW_DB_URL%" -f dump.sql

IF %ERRORLEVEL% NEQ 0 (
  echo Failed to import tables into Supabase.
  pause
  exit /b %ERRORLEVEL%
)

echo ✅ Migration of judgments and acts tables completed successfully!
pause
