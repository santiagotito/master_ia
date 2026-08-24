Después de instalar Python

1) Crear un ambiente virtual de Python 3.10 ejecutando en mcd :: 
		py -3.10 -m venv .venv_clasificador
2) Ingresar al ambiente virtual:    .\.venv_clasificador\Scripts\Activate.ps1

3) Instalar todos los componentes adicionales que están dentro de requirements.txt
		powershell:   pip install -r requirements.txt
4) Instalar Streamlit en el ambiente virtual: .venv_clasificador 
5) Descargar e Instalar Cursor
6) Descomprimir la carpeta del proyecto zip en el directorio preferente : C:\User\clasificador_piezas
7) En powershell desde ambiente virtual ingresa al directorio del proyecto: cd C:\User\clasificador_piezas
8) En powershell Ejecutar Streamlit dentro del directorio del proyecto:  streamlit run app.py
9) Va abrir el proyecto en el nevegador
