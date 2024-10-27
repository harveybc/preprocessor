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
                analizar_archivo_csv(csv_files[0], 4500)
            else:
                print(f"[ERROR] No se encontró archivo CSV en el dataset {dataset}")
        except Exception as e:
            print(f"[ERROR] Error durante la descarga o procesamiento del dataset {dataset}: {e}")

# Define the main function that analyzes the CSV file
def analizar_archivo_csv(ruta_archivo_csv, limite_filas=None):
    try:
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
        columnas = data.columns
        mejor_columna = None
        mejor_snr = -np.inf
        resultados = {}

        for i, columna in enumerate(columnas):
            try:
                print(f"[INFO] Analizando columna {i + 1}")
                serie = data[columna]
                if len(serie) < 2:
                    print(f"[ERROR] La columna {i + 1} no tiene suficientes datos para el análisis.")
                    continue

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
                
                # Decompose the time series using seasonal_decompose
                decomposition = sm.tsa.seasonal_decompose(serie, model='additive', period=30)
                tendencia = decomposition.trend
                estacionalidad = decomposition.seasonal
                residuales = decomposition.resid
                
                # Plot and save decomposition
                plt.figure()
                plt.plot(tendencia, label='Tendencia')
                plt.plot(estacionalidad, label='Estacionalidad')
                plt.plot(residuales, label='Residuales')
                plt.legend()
                plt.title(f"Descomposición de la serie - Dataset Periodicidad {ruta_archivo_csv.stem} Columna {columna}")
                plt.savefig(f"output/decomposition_{ruta_archivo_csv.stem}_col{columna}.png")
                plt.close()

                # Plot and save Fourier Spectrum for the best column
                plt.figure()
                plt.plot(frecuencias, espectro)
                plt.title(f"Espectro de Fourier - Dataset Periodicidad {ruta_archivo_csv.stem} Columna {columna}")
                plt.xlabel('Frecuencia')
                plt.ylabel('Potencia')
                plt.savefig(f"output/fourier_spectrum_{ruta_archivo_csv.stem}_col{columna}.png")
                plt.close()
                
                # Store results
                resultados[columna] = {
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
                
                # Select the best column based on SNR
                if snr > mejor_snr:
                    mejor_snr = snr
                    mejor_columna = columna
            
            except Exception as e:
                print(f"[ERROR] Error al analizar la columna {i + 1}: {e}")

        # Print summary statistics for each dataset
        print("*********************************************")
        print(f"Estadísticas para dataset:")
        for columna, stats in resultados.items():
            print(f"Columna: {columna}")
            for key, value in stats.items():
                if isinstance(value, np.ndarray):
                    print(f"  {key}: {value[:5]}... (truncado)")  # Print only first 5 elements if array
                else:
                    print(f"  {key}: {value}")
        print("*********************************************")

        # Print summary for the best dataset for each scenario
        print("Resumen de uso:")
        if mejor_columna is not None:
            print(f"Mejor columna para predicción de tendencias: Columna {mejor_columna} del dataset {ruta_archivo_csv.stem}")
            print(f"Características de la mejor columna:")
            for key, value in resultados[mejor_columna].items():
                if isinstance(value, np.ndarray):
                    print(f"  {key}: {value[:5]}... (truncado)")
                else:
                    print(f"  {key}: {value}")
        print("*********************************************")

    except FileNotFoundError:
        print("[ERROR] El archivo especificado no se encontró. Por favor verifique la ruta.")
    except pd.errors.EmptyDataError:
        print("[ERROR] El archivo CSV está vacío.")
    except Exception as e:
        print(f"[ERROR] Ocurrió un error inesperado: {e}")

# Start the process
descargar_y_procesar_datasets()
