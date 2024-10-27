# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.fft import fft
import warnings
import kagglehub
import os
from pathlib import Path

# Configure to ignore warnings
warnings.filterwarnings("ignore")

# Ensure output directory exists
def ensure_output_directory():
    output_dir = Path("output")
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    print(f"[INFO] Output directory verified at: {output_dir}")

# Function to download and process datasets
def descargar_y_procesar_datasets():
    ensure_output_directory()
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

            # Verify if download was successful
            if not path.exists():
                print(f"[ERROR] La ruta de descarga no existe: {path}")
                continue

            # Find CSV files
            csv_files = [file for file in path.glob('**/*.csv')]
            if csv_files:
                print(f"[INFO] Analizando el archivo CSV: {csv_files[0]}")
                resumen_dataset = analizar_archivo_csv(csv_files[0], 4500)
                if resumen_dataset is not None:
                    resumen_general.append(resumen_dataset)
            else:
                print(f"[ERROR] No se encontró archivo CSV en el dataset {dataset}")
        except Exception as e:
            print(f"[ERROR] Error durante la descarga o procesamiento del dataset {dataset}: {e}")

    # Generate summary table for all datasets
    if resumen_general:
        generar_tabla_resumen(resumen_general)
        generar_csv_resumen(resumen_general)
    else:
        print("[INFO] No se generó ningún resumen general debido a errores en el procesamiento de los datasets.")

# Function to analyze CSV file
def analizar_archivo_csv(ruta_archivo_csv, limite_filas=None):
    try:
        # Load CSV without headers
        data = pd.read_csv(ruta_archivo_csv, header=True, skiprows=1, index_col=False)

        # Limit rows if specified
        if limite_filas is not None and len(data) > limite_filas:
            data = data.tail(limite_filas)

        # Drop the first column (assumed to be date)
        data.drop(data.columns[0], axis=1, inplace=True)

        # Ensure there are enough columns for analysis
        if len(data.columns) < 4:
            print("[ERROR] No hay suficientes columnas para el análisis.")
            return None

        # Convert the fourth column to numeric
        serie = pd.to_numeric(data.iloc[:, 3], errors='coerce')
        serie.dropna(inplace=True)

        # Ensure there are enough rows after cleaning
        if len(serie) < 2:
            print("[ERROR] No hay suficientes datos para el análisis después de la limpieza de valores nulos.")
            return None

        # Calculate statistics
        media = serie.mean() if not serie.empty else 'E'
        desviacion = serie.std() if not serie.empty else 'E'
        snr = (media / desviacion) ** 2 if desviacion != 0 else 'E'
        promedio_retornos = serie.diff().abs().mean() if not serie.empty else 'E'

        # Decompose time series into trend, seasonal, and residual components
        decomposition = sm.tsa.seasonal_decompose(serie, model='additive', period=30)
        plt.figure()
        decomposition.plot()
        plt.suptitle(f"Trend, Seasonality, and Residuals - {ruta_archivo_csv.name}")
        plt.savefig(f"output/{ruta_archivo_csv.stem}_decomposition.png")

        # Fourier analysis
        espectro = np.abs(fft(serie))
        plt.figure()
        plt.plot(espectro[:len(espectro)//2])
        plt.title(f"Espectro de Fourier - {ruta_archivo_csv.name}")
        plt.savefig(f"output/{ruta_archivo_csv.stem}_fourier_spectrum.png")

        # Prepare the summary for this dataset
        resumen = {
            "dataset": str(ruta_archivo_csv),
            "media": media,
            "desviacion": desviacion,
            "snr": snr,
            "promedio_retornos": promedio_retornos
        }

        return resumen

    except Exception as e:
        print(f"[ERROR] Ocurrió un error inesperado: {e}")
        return {
            "dataset": str(ruta_archivo_csv),
            "media": 'E',
            "desviacion": 'E',
            "snr": 'E',
            "promedio_retornos": 'E'
        }

# Function to generate summary CSV
def generar_csv_resumen(resumen_general):
    df_resumen = pd.DataFrame(resumen_general)
    df_resumen.to_csv("output/resumen_general.csv", index=False)
    print("[INFO] Resumen general generado y guardado en: output/resumen_general.csv")

# Function to generate summary table
def generar_tabla_resumen(resumen_general):
    print("\n*********************************************")
    for resumen in resumen_general:
        print(f"Estadísticas para dataset {resumen['dataset']}:")
        print(f"  Media: {resumen['media']}")
        print(f"  Desviación estándar: {resumen['desviacion']}")
        print(f"  SNR: {resumen['snr']}")
        print(f"  Promedio de retornos: {resumen['promedio_retornos']}")
        print("*********************************************")

# Execute the script
descargar_y_procesar_datasets()
