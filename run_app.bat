@echo off
if not exist .venv\Scripts\python.exe (
  echo Create the virtual environment first: python -m venv .venv
  exit /b 1
)
.venv\Scripts\python.exe -m streamlit run app.py
