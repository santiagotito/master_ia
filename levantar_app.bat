@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo  Clasificador UIDE - Dashboard (Nivel 1 en vivo + hibrido)
echo ============================================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo No se encontro "python" en el PATH. Instala Python o abri este
    echo .bat desde una consola donde "python --version" funcione.
    pause
    exit /b 1
)

echo Instalando/actualizando dependencias (rapido si ya estan)...
python -m pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo.
    echo Fallo la instalacion de dependencias. Revisa el error de arriba.
    pause
    exit /b 1
)

echo.
echo Levantando Streamlit... se abre solo en el navegador.
echo Para cerrar: volve a esta ventana y apreta Ctrl+C.
echo.
python -m streamlit run app.py

pause
