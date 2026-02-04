@echo off
echo ============================================================
echo   Сборка TextStageProcessor.exe
echo ============================================================
echo.
echo Убедитесь, что установлены зависимости:
echo   pip install -r requirements.txt
echo   pip install pyinstaller pymorphy3 pymorphy3-dicts-ru
echo.

python build_exe.py

echo.
echo Готово! Результат в папке dist\TextStageProcessor\
pause
