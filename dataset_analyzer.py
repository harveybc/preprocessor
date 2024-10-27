# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.signal import find_peaks
import statsmodels.api as sm
import nolds
import warnings
import kagglehub
import os
from pathlib import Path

# Configure to ignore warnings
warnings.filterwarnings("ignore")

# Function to ensure output directory exists
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
        print_resumen_separador(resumen_general)

# Function to analyze CSV file
def analizar_archivo_csv(ruta_archivo_csv, limite_filas=None):
    try:
        print(f"[DEBUG] LABEL: Loading CSV file")
        # Load CSV without headers
        print(f"[DEBUG] Cargando el archivo CSV desde la ruta: {ruta_archivo_csv}")
        data = pd.read_csv(ruta_archivo_csv, header=None, skiprows=1, index_col=False)

        # Debug the initial dataset load
        print(f"[DEBUG] LABEL: Initial dataset load")
        print(f"[DEBUG] Primeras 10 filas del dataset cargado:\n{data.head(10)}")

        # Limit rows if specified
        if limite_filas is not None and len(data) > limite_filas:
            print(f"[DEBUG] LABEL: Limiting dataset rows")
            data = data.tail(limite_filas)
            print(f"[DEBUG] Limitando los datos a las últimas {limite_filas} filas.")

        # Drop the first column (assumed to be date)
        print(f"[DEBUG] LABEL: Dropping date column")
        print(f"[DEBUG] Columnas antes de eliminar la columna de fecha: {data.columns}")
        data.drop(data.columns[0], axis=1, inplace=True)

        # Debug after dropping the date column
        print(f"[DEBUG] LABEL: Dataset after dropping date column")
        print(f"[DEBUG] Dataset después de eliminar la columna de fecha:\n{data.head(10)}")

        # Ensure there are enough columns for analysis
        if len(data.columns) < 4:
            print("[ERROR] No hay suficientes columnas para el análisis.")
            return None

        # Convert the fourth column to numeric
        print(f"[DEBUG] LABEL: Converting fourth column to numeric")
        print(f"[DEBUG] Valores antes de convertir la cuarta columna a numérico:\n{data.iloc[:, 3].head(10)}")
        serie = pd.to_numeric(data.iloc[:, 3], errors='coerce')

        # Drop NaN values from the series
        print(f"[DEBUG] LABEL: Dropping NaN values from series")
        print(f"[DEBUG] Serie antes de eliminar NaN: {serie.shape}")
        serie.dropna(inplace=True)
        print(f"[DEBUG] Serie después de eliminar NaN: {serie.shape}")

        # Ensure there are enough rows after cleaning
        if len(serie) < 2:
            print("[ERROR] No hay suficientes datos para el análisis después de la limpieza de valores nulos.")
            return None

        # Calculate statistics
        print(f"[DEBUG] LABEL: Calculating statistics for series")
        print(f"[DEBUG] Calculando estadísticas para la serie...")
        media = serie.mean()
        desviacion = serie.std()
        snr = (media / desviacion) ** 2 if desviacion != 0 else np.nan

        # Debug statistics
        print(f"[DEBUG] Media calculada: {media}")
        print(f"[DEBUG] Desviación calculada: {desviacion}")
        print(f"[DEBUG] SNR calculado: {snr}")

        # Calculate returns
        print(f"[DEBUG] LABEL: Calculating returns")
        retornos = serie.diff().abs().dropna()
        promedio_retornos = retornos.mean()

        # Debug returns
        print(f"[DEBUG] Promedio de retornos: {promedio_retornos}")

        # Additional analysis
        print(f"[DEBUG] LABEL: Calculating Hurst exponent")
        print(f"[DEBUG] Calculando Hurst exponent...")
        hurst_exponent = nolds.hurst_rs(serie)
        print(f"[DEBUG] Hurst exponent calculado: {hurst_exponent}")

        print(f"[DEBUG] LABEL: Calculating DFA")
        print(f"[DEBUG] Calculando DFA...")
        dfa = nolds.dfa(serie)
        print(f"[DEBUG] DFA calculado: {dfa}")

        # Fourier analysis
        print(f"[DEBUG] LABEL: Calculating Fourier spectrum")
        print(f"[DEBUG] Calculando espectro de Fourier...")
        espectro = np.abs(fft(serie))

        # Debug Fourier spectrum before normalization
        print(f"[DEBUG] Espectro de Fourier (sin normalizar): {espectro[:10]}")

        espectro_normalizado = espectro / espectro.sum()

        # Debug normalized Fourier spectrum
        print(f"[DEBUG] Espectro de Fourier (normalizado): {espectro_normalizado[:10]}")

        entropia_espectral = -np.sum(espectro_normalizado * np.log2(espectro_normalizado + 1e-10))

        # Debug Fourier analysis
        print(f"[DEBUG] Entropía espectral calculada: {entropia_espectral}")

        # Find peaks in Fourier spectrum
        print(f"[DEBUG] LABEL: Finding peaks in Fourier spectrum")
        try:
            print(f"[DEBUG] Buscando picos en el espectro de Fourier...")
            freqs = np.fft.fftfreq(len(serie))

            # Debug frequencies array
            print(f"[DEBUG] Frecuencias calculadas: {freqs[:10]}")

            peaks, _ = find_peaks(espectro)

            # Debug peaks found in the Fourier spectrum
            print(f"[DEBUG] Índices de picos encontrados: {peaks[:10]}")
            print(f"[DEBUG] Valores de picos encontrados: {espectro[peaks[:10]]}")

            if len(peaks) == 0:
                print("[ERROR] No se encontraron picos en el espectro de Fourier.")
                return None

            peak_indices = np.argsort(espectro[peaks])[-5:][::-1]
            peak_freqs = freqs[peaks][peak_indices]

        except IndexError as ie:
            print(f"[ERROR] Error durante el análisis de picos del espectro de Fourier: {ie}")
            return None

        # Debug Fourier peaks
        print(f"[DEBUG] Picos principales del espectro de Fourier: {peak_freqs}")

        # Rest of the analysis continues as before...

    except Exception as e:
        print(f"[ERROR] Ocurrió un error inesperado: {e}")
        print(f"[DEBUG] Contexto del error: Dataset - {ruta_archivo_csv}, Serie - {serie if 'serie' in locals() else 'No definida'}")
        return None

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
        print(f"  Hurst Exponent: {resumen['hurst_exponent']}")
        print(f"  DFA: {resumen['dfa']}")
        print(f"  Promedio de retornos: {resumen['promedio_retornos']}")
        print(f"  Autocorrelación (lag 1): {resumen['autocorr_1']}")
        print(f"  Pico Frecuencia 1: {resumen['pico_frecuencia 1']}")
        print(f"  Pico Frecuencia 2: {resumen['pico_frecuencia 2']}")
        print(f"  Pico Frecuencia 3: {resumen['pico_frecuencia 3']}")
        print(f"  Pico Frecuencia 4: {resumen['pico_frecuencia 4']}")
        print(f"  Pico Frecuencia 5: {resumen['pico_frecuencia 5']}")
        print("*********************************************")

# Function to print the summary separated by asterisks
def print_resumen_separador(resumen_general):
    print("\n*********************************************")
    for resumen in resumen_general:
        print(f"Dataset {resumen['dataset']}: Media={resumen['media']}, Desviación={resumen['desviacion']}, SNR={resumen['snr']}")
    print("*********************************************")

# Execute the script
descargar_y_procesar_datasets()
