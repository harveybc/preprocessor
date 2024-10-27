# Importar librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import kagglehub
import os
from pathlib import Path
from scipy.fft import fft
from scipy.signal import find_peaks
import statsmodels.api as sm
import nolds

# Configuración para ignorar advertencias de numpy/pandas que no afectan el procesamiento
warnings.filterwarnings("ignore")

# Definir la función para descargar y procesar los últimos 4500 datos de cada dataset
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

            # Aquí asumimos que el dataset tiene un archivo CSV principal
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

    # Generar tabla resumen de todos los datasets
    if resumen_general:
        generar_tabla_resumen(resumen_general)

# Definir la función principal que analizará el archivo CSV
def analizar_archivo_csv(ruta_archivo_csv, limite_filas=None):
    try:
        periodicidad = "1min" if "1minute" in str(ruta_archivo_csv).lower() else "15min" if "15min" in str(ruta_archivo_csv).lower() else "1h" if "1h" in str(ruta_archivo_csv).lower() else "1d"
        
        print(f"[DEBUG] Cargando el archivo CSV desde la ruta: {ruta_archivo_csv}")
        data = pd.read_csv(ruta_archivo_csv, skiprows=3)

        if limite_filas is not None and len(data) > limite_filas:
            data = data.tail(limite_filas)

        if data.shape[1] < 2:
            print("[ERROR] El archivo CSV debe tener al menos dos columnas.")
            return None

        # Eliminar la primera columna (fecha) y convertir el resto a numérico
        data = data.iloc[:, 1:]
        data = data.apply(pd.to_numeric, errors='coerce')
        data.dropna(axis=1, how='any', inplace=True)

        if data.shape[1] < 1:
            print("[ERROR] No hay suficientes columnas para el análisis después de limpiar los valores nulos.")
            return None

        columnas = data.columns
        resultados = {}
        mejor_columna = None
        mejor_snr = -np.inf

        for idx, columna in enumerate(columnas):
            try:
                print(f"[INFO] Analizando columna {idx + 1}")
                serie = data[columna]
                if len(serie) < 2:
                    print(f"[ERROR] La columna '{columna}' no tiene suficientes datos para el análisis.")
                    continue

                # Cálculo de estadísticas
                media = serie.mean()
                desviacion = serie.std()
                snr = (media / desviacion) ** 2 if desviacion != 0 else np.nan
                ruido_normalizado = 1 / snr if snr != 0 else np.nan
                desviacion_ruido = np.sqrt(ruido_normalizado) * desviacion if ruido_normalizado != 0 else np.nan
                amplitud_promedio = desviacion_ruido * np.sqrt(2 / np.pi) if ruido_normalizado != 0 else np.nan

                # Cálculo de retornos
                retornos = serie.diff().abs().dropna()
                promedio_retornos = retornos.mean()

                # Exponente de Hurst
                hurst_exponent = nolds.hurst_rs(serie)

                # Detrended Fluctuation Analysis (DFA)
                dfa = nolds.dfa(serie)

                # Espectro de Fourier y análisis de picos
                espectro = np.abs(fft(serie))
                frecuencias = np.fft.fftfreq(len(serie))
                picos, _ = find_peaks(espectro)
                picos_mas_altos = picos[np.argsort(espectro[picos])[-5:]]
                frecuencias_picos = frecuencias[picos_mas_altos]
                potencias_picos = espectro[picos_mas_altos]

                # Graficar el espectro de Fourier
                plt.figure(figsize=(10, 5))
                plt.plot(frecuencias, espectro)
                plt.title(f'Espectro de Fourier para dataset {periodicidad}, columna {columna}')
                plt.xlabel('Frecuencia')
                plt.ylabel('Amplitud')
                plt.savefig(f'output/espectro_fourier_{periodicidad}_col{idx + 1}.png')
                plt.close()

                # Análisis de estacionalidad, tendencia y residuales
                descomposicion = sm.tsa.seasonal_decompose(serie, model='additive', period=30)
                descomposicion.plot()
                plt.suptitle(f'Descomposición de estacionalidad para dataset {periodicidad}, columna {columna}')
                plt.savefig(f'output/descomposicion_estacionalidad_{periodicidad}_col{idx + 1}.png')
                plt.close()

                resultados[columna] = {
                    "media": media,
                    "desviacion_std": desviacion,
                    "SNR": snr,
                    "ruido_normalizado": ruido_normalizado,
                    "desviacion_ruido": desviacion_ruido,
                    "amplitud_promedio": amplitud_promedio,
                    "promedio_retornos": promedio_retornos,
                    "hurst_exponent": hurst_exponent,
                    "dfa": dfa,
                    "frecuencias_picos": frecuencias_picos,
                    "potencias_picos": potencias_picos
                }

                # Seleccionar la mejor columna para predicción (con mayor SNR)
                if snr > mejor_snr:
                    mejor_snr = snr
                    mejor_columna = columna

            except Exception as e:
                print(f"[ERROR] Error al analizar la columna '{columna}': {e}")

        # Evaluar la calidad del dataset para cada escenario
        calificaciones = evaluar_dataset(resultados)

        # Generar resumen del dataset
        generar_resumen(resultados, periodicidad)

        if mejor_columna:
            resumen_dataset = {
                "dataset": periodicidad,
                "mejor_columna": mejor_columna,
                "mejor_snr": mejor_snr,
                "calificaciones": calificaciones
            }
            return resumen_dataset
        return None

    except Exception as e:
        print(f"[ERROR] Ocurrió un error inesperado: {e}")
    return None

# Evaluar la calidad del dataset
def evaluar_dataset(resultados):
    calificaciones = {}
    for columna, stats in resultados.items():
        prediccion_tendencias = "Alta calidad" if stats['SNR'] > 10 and stats['hurst_exponent'] > 0.5 else "Baja calidad"
        balanceo_portafolios = "Adecuada" if stats['dfa'] < 1.5 else "No adecuada"
        trading_automatico = "Buena" if stats['potencias_picos'].mean() > 1000 else "Baja"

        calificaciones[columna] = {
            "Predicción de Tendencias": prediccion_tendencias,
            "Balanceo de Portafolios": balanceo_portafolios,
            "Trading Automático": trading_automatico
        }
    return calificaciones

# Resumen del dataset
def generar_resumen(resultados, periodicidad):
    print("\n*********************************************")
    print(f"Estadísticas para dataset {periodicidad}:")
    for columna, stats in resultados.items():
        print(f"Columna: {columna}")
        print(f"  Media: {stats['media']}")
        print(f"  Desviación estándar: {stats['desviacion_std']}")
        print(f"  SNR: {stats['SNR']}")
        print(f"  Exponente de Hurst: {stats['hurst_exponent']}")
        print(f"  DFA: {stats['dfa']}")
        print(f"  Frecuencias de los 5 picos principales: {stats['frecuencias_picos']}")
        print(f"  Potencia de los 5 picos principales: {stats['potencias_picos']}")
    print("*********************************************")

# Generar la tabla resumen de todos los datasets
def generar_tabla_resumen(resumen_general):
    resumen_df = pd.DataFrame(resumen_general)
    resumen_df.to_csv('output/resumen_general.csv', index=False)
    print("\nResumen general generado y guardado en: output/resumen_general.csv")
    print(resumen_df)

# Ejecutar la función principal
descargar_y_procesar_datasets()