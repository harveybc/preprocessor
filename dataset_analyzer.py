# Import necessary libraries
import pandas as pd
import numpy as np
import warnings
import kagglehub
import os
from pathlib import Path
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Configuration to ignore numpy/pandas warnings that do not affect processing
warnings.filterwarnings("ignore")

# Define the function to download and process the latest 4500 rows of each dataset
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
            
            # Convert path to Path object if necessary
            path = Path(path)

            # Verify if download was successful
            if not os.path.exists(path):
                print(f"[ERROR] La ruta de descarga no existe: {path}")
                continue
            
            # Assuming the dataset contains a main CSV file
            csv_files = [file for file in path.glob('**/*.csv')]
            if csv_files:
                print(f"[INFO] Analizando el archivo CSV: {csv_files[0]}")
                resumen = analizar_archivo_csv(csv_files[0], 4500)
                if resumen:
                    resumen_general.append(resumen)
            else:
                print(f"[ERROR] No se encontró archivo CSV en el dataset {dataset}")
        except Exception as e:
            print(f"[ERROR] Error durante la descarga o procesamiento del dataset {dataset}: {e}")
    
    # Print the overall summary
    print("\n*********************************************")
    print("Resumen de uso:")
    for resumen in resumen_general:
        print(f"Dataset {resumen['periodicidad']}: mejor para {resumen['mejor_uso']} porque tiene calificación de trading ({resumen['calificaciones']['Trading']:.2f}), calificación de portafolio ({resumen['calificaciones']['Portafolio']:.2f}) y calificación de predicción ({resumen['calificaciones']['Predicción']:.2f})")
    print("\nTotal de datasets analizados:")
    for resumen in resumen_general:
        print(f"{resumen['periodicidad']} = {resumen['filas']} filas")
    print("*********************************************")

# Define the main function that analyzes the CSV file
def analizar_archivo_csv(ruta_archivo_csv, limite_filas=None):
    try:
        # Determine the periodicity of the dataset based on the file name
        if "1minute" in str(ruta_archivo_csv).lower():
            periodicidad = "1min"
        elif "15min" in str(ruta_archivo_csv).lower():
            periodicidad = "15min"
        elif "1h" in str(ruta_archivo_csv).lower():
            periodicidad = "1h"
        elif "1d" in str(ruta_archivo_csv).lower():
            periodicidad = "1d"
        else:
            periodicidad = "desconocido"
        
        # Load the CSV file using pandas
        try:
            print(f"[DEBUG] Cargando el archivo CSV desde la ruta: {ruta_archivo_csv}")
            data = pd.read_csv(ruta_archivo_csv, header=None, skiprows=1)  # Skip the first row which might be header
        except Exception as e:
            print(f"[ERROR] Error al cargar el archivo CSV: {e}")
            return None
        
        # Limit the data to the last 'limite_filas' if specified
        if limite_filas is not None and len(data) > limite_filas:
            data = data.tail(limite_filas)
            print(f"[DEBUG] Limitando los datos a las últimas {limite_filas} filas.")
        
        # Drop the first column (date)
        data = data.iloc[:, 1:]
        
        # Convert all columns to numeric
        data = data.apply(pd.to_numeric, errors='coerce')
        
        # Show the first 10 rows of the dataset
        print("Primeras 10 filas del dataset antes del procesamiento:")
        print(data.head(10))
        
        # Count NaN values per column and display
        nan_counts = data.isna().sum()
        print("Conteo de valores NaN por columna:")
        print(nan_counts)
        
        # Drop columns with all NaN values
        data.dropna(axis=1, how='all', inplace=True)
        
        # Validate that there are still enough rows after cleaning
        if data.shape[0] < 2:
            print("[ERROR] No hay suficientes datos para el análisis después de la limpieza de valores nulos.")
            return None
        
        # Proceed with further analysis (placeholder for additional steps)
        # Fourier Transform and other analysis steps
        # Use only column 4 (counting from 0)
        if data.shape[1] < 4:
            print("[ERROR] No hay suficientes columnas para realizar el análisis de la columna 4.")
            return None
        serie = data.iloc[:, 3]
        
        # Validate that the series has enough data points
        if len(serie) < 2:
            print("[ERROR] La columna seleccionada no tiene suficientes datos para el análisis.")
            return None

        # Calculate statistics
        media = serie.mean()
        desviacion = serie.std()
        snr = (media / desviacion) ** 2 if desviacion != 0 else np.nan
        ruido_normalizado = 1 / snr if snr != 0 else np.nan
        desviacion_ruido = np.sqrt(ruido_normalizado) * desviacion if ruido_normalizado != 0 else np.nan
        amplitud_promedio = desviacion_ruido * np.sqrt(2 / np.pi) if ruido_normalizado != 0 else np.nan
        retornos = serie.diff().abs().dropna()
        promedio_retornos = retornos.mean()

        # Perform Fourier Transform
        espectro = np.abs(np.fft.fft(serie))
        frecuencias = np.fft.fftfreq(len(serie))
        picos, _ = find_peaks(espectro)
        potencias_picos = espectro[picos]
        indices_ordenados = np.argsort(potencias_picos)[-5:][::-1]
        picos_principales = frecuencias[picos][indices_ordenados]
        potencias_principales = potencias_picos[indices_ordenados]
        
        # Store results
        resultados = {
            "media": media,
            "desviacion_std": desviacion,
            "SNR": snr,
            "ruido_normalizado": ruido_normalizado,
            "desviacion_ruido": desviacion_ruido,
            "amplitud_promedio": amplitud_promedio,
            "promedio_retornos": promedio_retornos,
            "frecuencias_picos": picos_principales,
            "potencia_picos": potencias_principales
        }

        # Print summary statistics for each dataset
        print("*********************************************")
        print(f"Estadísticas para dataset {periodicidad}:")
        for key, value in resultados.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: {value[:5]}... (truncado)")  # Print only first 5 elements if array
            else:
                print(f"  {key}: {value}")
        print("*********************************************")

        # Plot Fourier Spectrum for the column
        plt.figure(figsize=(10, 6))
        plt.plot(frecuencias, espectro)
        plt.title(f"Espectro de Fourier para la columna 4 - Dataset {periodicidad}")
        plt.xlabel("Frecuencia")
        plt.ylabel("Potencia")
        plt.grid(True)
        plt.savefig(f"output/espectro_fourier_{periodicidad}_col4.png")
        plt.close()

        # Decompose the series into trend, seasonal, and residual components
        descomposicion = sm.tsa.seasonal_decompose(serie, model='additive', period=30)
        descomposicion.plot()
        plt.savefig(f"output/descomposicion_{periodicidad}_col4.png")
        plt.close()

        # Scenario scores
        calificaciones = {
            "Trading": snr,
            "Predicción": media,
            "Portafolio": -desviacion  # Lower deviation is better for portfolio balancing
        }

        # Determine best use case for dataset based on the highest score in each scenario
        mejor_uso = max(calificaciones, key=calificaciones.get)
        
        # Return summary for general use
        return {
            "periodicidad": periodicidad,
            "mejor_columna": 4,  # Fixed column selection
            "mejor_snr": snr,
            "mejor_uso": mejor_uso,
            "filas": data.shape[0],
            "calificaciones": calificaciones
        }

    except FileNotFoundError:
        print("[ERROR] El archivo especificado no se encontró. Por favor verifique la ruta.")
    except pd.errors.EmptyDataError:
        print("[ERROR] El archivo CSV está vacío.")
    except Exception as e:
        print(f"[ERROR] Ocurrió un error inesperado: {e}")

# Start the process
descargar_y_procesar_datasets()
