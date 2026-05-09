@echo off
REM Start the application on localhost and suppress the LangGraph warning.
set PYTHONWARNINGS=ignore:.*allowed_objects.*
cd /d %~dp0
".\.venv\Scripts\uvicorn.exe" app.api.server:app --host 127.0.0.1 --port 8000
