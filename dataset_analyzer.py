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
        # Load CSV without headers
        print(f"[DEBUG] Cargando el archivo CSV desde la ruta: {ruta_archivo_csv}")
        data = pd.read_csv(ruta_archivo_csv, header=None, skiprows=1, index_col=False)

        # Debug the initial dataset load
        print(f"[DEBUG] Primeras 10 filas del dataset cargado:\n{data.head(10)}")

        # Limit rows if specified
        if limite_filas is not None and len(data) > limite_filas:
            data = data.tail(limite_filas)
            print(f"[DEBUG] Limitando los datos a las últimas {limite_filas} filas.")

        # Drop the first column (assumed to be date)
        print(f"[DEBUG] Columnas antes de eliminar la columna de fecha: {data.columns}")
        data.drop(data.columns[0], axis=1, inplace=True)

        # Debug after dropping the date column
        print(f"[DEBUG] Dataset después de eliminar la columna de fecha:\n{data.head(10)}")

        # Ensure there are enough columns for analysis
        if len(data.columns) < 4:
            print("[ERROR] No hay suficientes columnas para el análisis.")
            return None

        # Convert the fourth column to numeric
        print(f"[DEBUG] Valores antes de convertir la cuarta columna a numérico:\n{data.iloc[:, 3].head(10)}")
        serie = pd.to_numeric(data.iloc[:, 3], errors='coerce')

        # Drop NaN values from the series
        print(f"[DEBUG] Serie antes de eliminar NaN: {serie.shape}")
        serie.dropna(inplace=True)
        print(f"[DEBUG] Serie después de eliminar NaN: {serie.shape}")

        # Ensure there are enough rows after cleaning
        if len(serie) < 2:
            print("[ERROR] No hay suficientes datos para el análisis después de la limpieza de valores nulos.")
            return None

        # Calculate statistics
        print(f"[DEBUG] Calculando estadísticas para la serie...")
        media = serie.mean()
        desviacion = serie.std()
        snr = (media / desviacion) ** 2 if desviacion != 0 else np.nan

        # Calculate returns
        retornos = serie.diff().abs().dropna()
        promedio_retornos = retornos.mean()

        # Additional analysis
        hurst_exponent = nolds.hurst_rs(serie)
        dfa = nolds.dfa(serie)

        # Fourier analysis
        espectro = np.abs(fft(serie))
        espectro_normalizado = espectro / espectro.sum()
        entropia_espectral = -np.sum(espectro_normalizado * np.log2(espectro_normalizado + 1e-10))

        # Find peaks in Fourier spectrum
        freqs = np.fft.fftfreq(len(serie))
        peaks, _ = find_peaks(espectro)
        peak_indices = np.argsort(espectro[peaks])[-5:][::-1]
        peak_freqs = freqs[peaks][peak_indices]

        # Debug Fourier peaks
        print(f"[DEBUG] Picos principales del espectro de Fourier: {peak_freqs}")

        # Autocorrelation analysis
        autocorr_1 = serie.autocorr(lag=1)
        print(f"[DEBUG] Autocorrelación (lag 1): {autocorr_1}")

        # Seasonal decomposition
        print(f"[DEBUG] Realizando descomposición estacional...")
        decomposition = sm.tsa.seasonal_decompose(serie, period=int(len(serie) / 2), model='additive')

        # Plot seasonal decomposition
        plt.figure(figsize=(10, 8))
        plt.subplot(411)
        plt.plot(decomposition.trend)
        plt.title('Tendencia')
        plt.subplot(412)
        plt.plot(decomposition.seasonal)
        plt.title('Estacionalidad')
        plt.subplot(413)
        plt.plot(decomposition.resid)
        plt.title('Residuales')
        plt.tight_layout()
        plt.savefig(f"output/estacionalidad_{Path(ruta_archivo_csv).stem}_col4.png")
        plt.close()
        print(f"[INFO] Guardado el gráfico de descomposición estacional para: {Path(ruta_archivo_csv).stem}")

        # Save Fourier spectrum plot
        plt.figure()
        plt.plot(freqs, espectro)
        plt.title('Espectro de Fourier')
        plt.xlabel('Frecuencia')
        plt.ylabel('Potencia')
        plt.savefig(f"output/fourier_{Path(ruta_archivo_csv).stem}_col4.png")
        plt.close()
        print(f"[INFO] Guardado el gráfico del espectro de Fourier para: {Path(ruta_archivo_csv).stem}")

        # Return dataset summary
        resumen_dataset = {
            "dataset": Path(ruta_archivo_csv).stem,
            "media": media,
            "desviacion": desviacion,
            "snr": snr,
            "hurst_exponent": hurst_exponent,
            "dfa": dfa,
            "promedio_retornos": promedio_retornos,
            "autocorr_1": autocorr_1,
            "pico_frecuencia 1": peak_freqs[0] if len(peak_freqs) > 0 else np.nan,
            "pico_frecuencia 2": peak_freqs[1] if len(peak_freqs) > 1 else np.nan,
            "pico_frecuencia 3": peak_freqs[2] if len(peak_freqs) > 2 else np.nan,
            "pico_frecuencia 4": peak_freqs[3] if len(peak_freqs) > 3 else np.nan,
            "pico_frecuencia 5": peak_freqs[4] if len(peak_freqs) > 4 else np.nan
        }
        return resumen_dataset

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
