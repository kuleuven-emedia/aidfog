@echo on
call .venv\Scripts\activate
set PYTHONPATH="%cd%"

set "FILE=trial_auto_id.txt"
if exist "%FILE%" (
    < "%FILE%" set /p "TRIAL_ID="
) else (
    set "TRIAL_ID=0"
)
set /a TRIAL_ID=%TRIAL_ID% + 1
echo %TRIAL_ID% > "%FILE%"

hermes-cli -o .\data -f resources\buds.yml -e project=AidFOG trial=%TRIAL_ID%
