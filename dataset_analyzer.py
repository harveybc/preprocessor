# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.fft import fft
from scipy.signal import find_peaks
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
        # 1min data
        ("jkalamar/eurusd-foreign-exchange-fx-intraday-1minute", "1min"),
        # 5min data
        ("stijnvanleeuwen/eurusd-forex-pair-15min-2002-2019", "5min"),
        # 15min data
        ("meehau/EURUSD", "15min"),
        # 1 hour data
        ("imetomi/eur-usd-forex-pair-historical-data-2002-2019", "1h"),
        # 4 hour data
        ("chandrimad31/eurusd-forex-trading-data-20032021", "4h"),
        # Daily data
        ("gabrielmv/eurusd-daily-historical-data-20012019", "daily")
    ]

    resumen_general = []

    for dataset, periodicity in datasets:
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
                resumen_dataset = analizar_archivo_csv(csv_files[0], 4500, periodicity)
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
def analizar_archivo_csv(ruta_archivo_csv, limite_filas=None, periodicity="unknown"):
    try:
        # Load CSV without headers
        data = pd.read_csv(ruta_archivo_csv, header=None, skiprows=1, index_col=False)

        # Limit rows if specified
        if limite_filas is not None and len(data) > limite_filas:
            data = data.tail(limite_filas)

        # Drop the first column (assumed to be date)
        data.drop(data.columns[0], axis=1, inplace=True)

        # Ensure there are enough columns for analysis
        if len(data.columns) < 4:
            print("[ERROR] No hay suficientes columnas para el análisis.")
            return None

        # Convert the correct column to numeric
        dataset_name = ruta_archivo_csv.name
        if dataset_name == 'eur-usd-forex-pair-historical-data-2002-2019.csv':
            serie = pd.to_numeric(data.iloc[:, 5], errors='coerce')
        else:
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
        potencia_error = 1 / snr if snr != 'E' and snr != 0 else 'E'
        desviacion_error = np.sqrt(potencia_error) if potencia_error != 'E' else 'E'
        media_error = (desviacion_error * (np.sqrt(2/np.pi))) if desviacion_error != 'E' else 'E'
        promedio_retornos = serie.diff().abs().mean() if not serie.empty else 'E'

        # Decompose time series into trend, seasonal, and residual components
        decomposition = sm.tsa.seasonal_decompose(serie, model='additive', period=30)
        plt.figure()
        decomposition.plot()
        plt.suptitle(f"Trend, Seasonality, and Residuals - {periodicity}")
        plt.savefig(f"output/{periodicity}_decomposition.png")

        # Fourier analysis
        espectro = np.abs(fft(serie))
        espectro_db = 20 * np.log10(espectro + 1e-10)  # Adding small value to avoid log(0)
        freqs = np.fft.fftfreq(len(espectro_db))
        plt.figure()
        plt.plot(freqs[:len(freqs)//2], espectro_db[:len(espectro_db)//2])
        plt.title(f"Espectro de Fourier - {periodicity}")
        plt.xlabel('Frecuencia (Hz)')
        plt.ylabel('Potencia (dB)')

        # Find top 5 peaks in the Fourier spectrum
        peaks, _ = find_peaks(espectro_db[:len(espectro_db)//2], height=None, distance=5, prominence=10)
        top_5_peaks = sorted(peaks, key=lambda x: espectro_db[x], reverse=True)[:5]
        top_5_peak_freqs = freqs[top_5_peaks] if len(top_5_peaks) > 0 else 'E'

        # Mark the top 5 peaks on the Fourier plot
        if top_5_peaks != 'E':
            plt.plot(freqs[top_5_peaks], espectro_db[top_5_peaks], "x")
        plt.savefig(f"output/{periodicity}_fourier_spectrum.png")

        # Autocorrelation
        autocorrelacion = [serie.autocorr(lag) for lag in range(1, 11)] if not serie.empty else 'E'

        # Plot autocorrelation
        plt.figure()
        pd.plotting.autocorrelation_plot(serie)
        plt.title(f"Autocorrelación - {periodicity}")
        plt.savefig(f"output/{periodicity}_autocorrelation.png")

        # Prepare the summary for this dataset
        resumen = {
            "dataset": str(ruta_archivo_csv),
            "media": media,
            "desviacion": desviacion,
            "snr": snr,
            "potencia_error": potencia_error,
            "desviacion_error": desviacion_error,
            "media_error": media_error,
            "promedio_retornos": promedio_retornos,
            "autocorrelacion": autocorrelacion,
            "top_5_peak_freqs": top_5_peak_freqs
        }

        return resumen

    except Exception as e:
        print(f"[ERROR] Ocurrió un error inesperado: {e}")
        return {
            "dataset": str(ruta_archivo_csv),
            "media": 'E',
            "desviacion": 'E',
            "snr": 'E',
            "potencia_error": 'E',
            "desviacion_error": 'E',
            "media_error": 'E',
            "promedio_retornos": 'E',
            "autocorrelacion": 'E',
            "top_5_peak_freqs": 'E'
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
        print(f"  Potencia del Error (PE): {resumen['potencia_error']}")
        print(f"  Desviación del Error (DE): {resumen['desviacion_error']}")
        print(f"  Media del Error: {resumen['media_error']}")
        print(f"  Promedio de retornos: {resumen['promedio_retornos']}")
        print(f"  Autocorrelación (lags 1-10): {resumen['autocorrelacion']}")
        print(f"  Frecuencias de los top 5 picos del espectro de Fourier: {resumen['top_5_peak_freqs']}")
        print("*********************************************")

# Execute the script
descargar_y_procesar_datasets()
