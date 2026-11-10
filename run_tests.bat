@echo off
set HF_HUB_OFFLINE=1
echo Running core, reconstructed pipeline, and Phase 3 evaluation tests...
set PYTHONPATH=.
.venv\Scripts\python.exe -m pytest tests -q
if errorlevel 1 exit /b 1

echo Running Member B verifier helper and relevance tests...
set PYTHONPATH=member_b
.venv\Scripts\python.exe -m pytest member_b\tests -q
if errorlevel 1 exit /b 1

echo All tests passed successfully.
