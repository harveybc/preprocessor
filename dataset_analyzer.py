# Importar librerías necesarias
import pandas as pd
import numpy as np
import warnings
import kagglehub
import os
from pathlib import Path

# Configuración para ignorar advertencias innecesarias
warnings.filterwarnings("ignore")

# Función principal para descargar y procesar datasets
def descargar_y_procesar_datasets():
    print("[INFO] Iniciando la descarga y procesamiento de los datasets...")
    datasets = [
        "jkalamar/eurusd-foreign-exchange-fx-intraday-1minute",
        "stijnvanleeuwen/eurusd-forex-pair-15min-2002-2019",
        "meehau/EURUSD",
        "imetomi/eur-usd-forex-pair-historical-data-2002-2019",
        "gabrielmv/eurusd-daily-historical-data-20012019"
    ]

    resumen_general = []
    for dataset in datasets:
        try:
            print(f"[INFO] Descargando dataset: {dataset}")
            path = kagglehub.dataset_download(dataset)
            path = Path(path)

            # Verificar si la descarga fue exitosa
            if not os.path.exists(path):
                print(f"[ERROR] La ruta de descarga no existe: {path}")
                continue

            # Asumir que el dataset tiene un archivo CSV principal
            csv_files = [file for file in path.glob('**/*.csv')]
            if csv_files:
                print(f"[INFO] Analizando el archivo CSV: {csv_files[0]}")
                analizar_archivo_csv(csv_files[0], 4500)
            else:
                print(f"[ERROR] No se encontró archivo CSV en el dataset {dataset}")
        except Exception as e:
            print(f"[ERROR] Error durante la descarga o procesamiento del dataset {dataset}: {e}")

# Función para analizar el archivo CSV sin depender de los nombres de columnas
def analizar_archivo_csv(ruta_archivo_csv, limite_filas=None):
    try:
        print(f"[DEBUG] Cargando el archivo CSV desde la ruta: {ruta_archivo_csv}")
        data = pd.read_csv(ruta_archivo_csv, skiprows=3)  # Saltar las primeras tres filas si son metadatos

        # Eliminar la primera columna (asumimos que es la fecha)
        data = data.iloc[:, 1:]

        # Limitar los datos a las últimas 'limite_filas' si se especifica
        if limite_filas is not None and len(data) > limite_filas:
            data = data.tail(limite_filas)
            print(f"[DEBUG] Limitando los datos a las últimas {limite_filas} filas.")

        # Convertir todas las columnas a numéricas (ignorar errores)
        for i in range(data.shape[1]):
            print(f"[DEBUG] Convirtiendo la columna {i + 1} a numérico.")
            data.iloc[:, i] = pd.to_numeric(data.iloc[:, i], errors='coerce')

        # Eliminar filas con valores NaN
        data.dropna(inplace=True)
        print(f"[DEBUG] Datos después de eliminar filas con NaN: {data.shape}")

        # Validar que todavía hay suficientes filas después de la limpieza
        if data.shape[0] < 2:
            print("[ERROR] No hay suficientes datos para el análisis después de la limpieza de valores nulos.")
            return

        # Iterar sobre cada columna (por índice) para analizarla
        for i in range(data.shape[1]):
            try:
                print(f"[INFO] Analizando columna {i + 1}")
                serie = data.iloc[:, i]
                media = serie.mean()
                desviacion = serie.std()
                snr = (media / desviacion) ** 2 if desviacion != 0 else np.nan

                print(f"[DEBUG] Media: {media}, Desviación Estándar: {desviacion}, SNR: {snr}")

            except Exception as e:
                print(f"[ERROR] Error al analizar la columna {i + 1}: {e}")

    except FileNotFoundError:
        print("[ERROR] El archivo especificado no se encontró. Por favor verifique la ruta.")
    except pd.errors.EmptyDataError:
        print("[ERROR] El archivo CSV está vacío.")
    except Exception as e:
        print(f"[ERROR] Ocurrió un error inesperado: {e}")

# Ejecutar la función principal
descargar_y_procesar_datasets()
