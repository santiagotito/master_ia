"""Ingesta y enrutamiento de piezas reales de RAM por tipo de archivo.

Ratifica el bucle de recorrido y el EXT_MAP de
PROYECTO/Clasificador_Piezas_ML.ipynb (celda 37) y la función
parse_filename() (celda 35). A diferencia del notebook, aquí sí se abre
el archivo real (para poder pasarlo al enriquecimiento IA) en vez de
solo leer el nombre.
"""
import logging
import os
import re

from config import EXT_MAP, FOLDER_2026, MESES_MAP

logger = logging.getLogger(__name__)


def parse_filename(filename):
    """Extrae anunciante y versión del nombre de archivo.

    Formato esperado: ANUNCIANTE - VERSIÓN(FORMATO).ext
    Ratificado tal cual de la celda 35 del notebook.
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    if ' - ' in base:
        anunciante, resto = base.split(' - ', 1)
        version = re.sub(r'\([^)]*\)\s*$', '', resto).strip()
        version = re.sub(r'[-_]\s*(FB|CIG|IG|TW|YT|\d+X\d+)\s*$', '', version).strip()
    else:
        anunciante, version = '', base
    return anunciante.strip(), version.strip()


def clasificar_tipo(filename):
    """Devuelve el tipo de pieza (Imagen/Video/Audio/PDF) según EXT_MAP,
    o None si la extensión no está soportada (AD-2: EXT_MAP es la única
    fuente de verdad para el tipo de archivo)."""
    ext = os.path.splitext(filename)[1].lower()
    return EXT_MAP.get(ext)


def listar_piezas(folder_2026=None):
    """Recorre FOLDER_2026/{mes}/... (solo carpetas de mes conocidas, ver
    MESES_MAP) y devuelve la lista de piezas reales.

    El recorrido dentro de cada mes es RECURSIVO: en producción no todos
    los meses organizan las piezas de la misma forma -- algunos las tienen
    sueltas directamente en la carpeta del mes, otros las anidan varios
    niveles por universidad/área (ej. `MAR/{universidad}/{Prensa|VP}/`).
    El código original del notebook asumía un único nivel y por eso nunca
    procesaba el contenido real de los meses con estructura anidada.

    Cada pieza: {ruta, archivo, mes, tipo, anunciante, version}.

    Un archivo con extensión no soportada, vacío o ilegible se omite con
    una advertencia registrada -- nunca detiene el resto del lote (AC #2).
    `desktop.ini` (metadato de sincronización de Google Drive Desktop) se
    ignora silenciosamente, sin generar advertencia de error.
    """
    folder_2026 = folder_2026 or FOLDER_2026
    piezas = []

    if not os.path.isdir(folder_2026):
        logger.warning("FOLDER_2026 no existe: %s", folder_2026)
        return piezas

    for mes_dir in MESES_MAP:
        mes_path = os.path.join(folder_2026, mes_dir)
        if not os.path.isdir(mes_path):
            continue

        for root, _dirs, files in os.walk(mes_path):
            for fname in sorted(files):
                if fname.startswith('~') or fname.startswith('.') or fname.lower() == 'desktop.ini':
                    continue

                ruta = os.path.join(root, fname)
                try:
                    tipo = clasificar_tipo(fname)
                    if tipo is None:
                        logger.warning("Extensión no soportada, se omite: %s", fname)
                        continue
                    if not os.path.isfile(ruta) or os.path.getsize(ruta) == 0:
                        logger.warning("Archivo corrupto o vacío, se omite: %s", fname)
                        continue

                    anunciante, version = parse_filename(fname)
                    piezas.append({
                        'ruta': ruta,
                        'archivo': fname,
                        'mes': mes_dir,
                        'tipo': tipo,
                        'anunciante': anunciante,
                        'version': version,
                    })
                except OSError:
                    logger.warning("No se pudo leer el archivo, se omite: %s", fname, exc_info=True)
                    continue

    return piezas
