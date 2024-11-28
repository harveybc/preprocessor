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
                resumen_dataset = analizar_archivo_csv(csv_files[0], periodicity)
                if resumen_dataset is not None:
                    resumen_general.append(resumen_dataset)
            else:
                print(f"[ERROR] No se encontró archivo CSV en el dataset {dataset}")
        except Exception as e:
            print(f"[ERROR] Error durante la descarga o procesamiento del dataset {dataset}: {e}")

    # Generate summary table for all datasets
    if resumen_general:
        generar_tabla_resumen(resumen_general)
    else:
        print("[INFO] No se generó ningún resumen general debido a errores en el procesamiento de los datasets.")

# Function to analyze CSV file
def analizar_archivo_csv(ruta_archivo_csv, periodicity="unknown"):
    try:
        # Load CSV without headers
        data = pd.read_csv(ruta_archivo_csv, header=None, skiprows=1, index_col=False)

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

        # Normalize returns (new behavior)
        retornos = serie.diff().dropna()  # Compute returns as the difference between consecutive values
        normalizacion_retornos = (retornos - retornos.mean()) / retornos.std()  # Normalizing returns

        # Calculate statistics
        media = serie.mean() if not serie.empty else 'E'
        desviacion = serie.std() if not serie.empty else 'E'
        snr = (normalizacion_retornos.mean() / normalizacion_retornos.std()) ** 2 if normalizacion_retornos.std() != 0 else 'E'
        potencia_error = 1 / snr if snr != 'E' and snr != 0 else 'E'
        desviacion_error = np.sqrt(potencia_error) if potencia_error != 'E' else 'E'
        media_error = (desviacion_error * (np.sqrt(2/np.pi))) if desviacion_error != 'E' else 'E'
        promedio_retornos = normalizacion_retornos.mean() if not normalizacion_retornos.empty else 'E'

        # Get the size of the dataset dynamically (instead of hardcoding 4500)
        dataset_size = len(serie)

        # Calculate sampling frequency in Hz
        periodicity_seconds_map = {
            "1min": 60,
            "5min": 5 * 60,
            "15min": 15 * 60,
            "1h": 60 * 60,
            "4h": 4 * 60 * 60,
            "daily": 24 * 60 * 60
        }
        sampling_period_seconds = periodicity_seconds_map.get(periodicity, None)
        sampling_frequency = 1 / sampling_period_seconds if sampling_period_seconds else 'E'

        # Calculate Shannon-Hartley channel capacity and noise-free information in bits
        if snr != 'E' and sampling_frequency != 'E':
            channel_capacity = sampling_frequency * np.log2(1 + snr)
            information_bits = channel_capacity * dataset_size * sampling_period_seconds  # Use dataset_size dynamically
        else:
            channel_capacity = 'E'
            information_bits = 'E'

        # Fourier analysis
        espectro = np.abs(fft(serie))
        espectro_db = 10 * np.log10(espectro + 1e-10)  # Using 10*log10 for better peak perception
        freqs = np.fft.fftfreq(len(espectro_db))
        plt.figure()
        plt.plot(freqs[:len(freqs)//2], espectro_db[:len(espectro_db)//2])
        plt.title(f"Espectro de Fourier - {periodicity}")
        plt.xlabel('Frecuencia (Hz)')
        plt.ylabel('Amplitud (dB)')
        plt.savefig(f"output/{periodicity}_fourier_spectrum.png")

        # Find peaks in the spectrum
        peaks, _ = find_peaks(espectro_db[:len(espectro_db)//2], height=0)
        top_peaks = sorted(peaks, key=lambda x: espectro_db[x], reverse=True)[:5]
        top_5_peak_freqs = [freqs[peak] for peak in top_peaks]

        # Calculate autocorrelation for lags 1-10
        autocorrelation = [serie.autocorr(lag) for lag in range(1, 11)]

        # Prepare the summary of results
        resumen = {
            'dataset': ruta_archivo_csv.name,
            'media': media,
            'desviacion': desviacion,
            'snr': snr,
            'potencia_error': potencia_error,
            'desviacion_error': desviacion_error,
            'media_error': media_error,
            'promedio_retornos': promedio_retornos,
            'autocorrelacion': autocorrelation,
            'top_5_peak_freqs': top_5_peak_freqs,
            'sampling_frequency': sampling_frequency,
            'channel_capacity': channel_capacity,
            'information_bits': information_bits
        }

        return resumen

    except Exception as e:
        print(f"[ERROR] Error durante el análisis del archivo CSV: {e}")
        return None

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
        print(f"  Frecuencia de muestreo (Hz): {resumen['sampling_frequency']}")
        print(f"  Capacidad del Canal (bits/seg): {resumen['channel_capacity']}")
        print(f"  Información libre de ruido (bits): {resumen['information_bits']}")
        print("*********************************************")

# Execute the script
descargar_y_procesar_datasets()
