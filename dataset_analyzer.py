# Importar librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fft import fft
from scipy.signal import find_peaks
import pywt
import nolds
import kagglehub
import os
from pathlib import Path

# Definir la función para descargar y cargar los últimos 4500 datos de cada dataset
def descargar_y_procesar_datasets():
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
            path = kagglehub.dataset_download(dataset)
            # Convertir path en un objeto Path si es necesario
            path = Path(path)

            # Verificar si la descarga fue exitosa
            if not os.path.exists(path):
                continue
            
            # Aquí asumimos que el dataset tiene un archivo CSV principal
            csv_files = [file for file in path.glob('**/*.csv')]
            if csv_files:
                resumen_dataset = analizar_archivo_csv(csv_files[0], 4500)
                if resumen_dataset is not None:
                    resumen_general.append(resumen_dataset)
        except Exception as e:
            continue
    
    # Generar tabla resumen de todos los datasets
    if resumen_general:
        generar_tabla_resumen(resumen_general)

# Definir la función principal que analizará el archivo CSV
# Ahora también acepta un parámetro de límite de filas
def analizar_archivo_csv(ruta_archivo_csv, limite_filas=None):
    try:
        # Determinar la periodicidad del dataset a partir del nombre del archivo
        periodicidad = "1min" if "1minute" in str(ruta_archivo_csv).lower() else "15min" if "15min" in str(ruta_archivo_csv).lower() else "1h" if "1h" in str(ruta_archivo_csv).lower() else "1d"
        
        # Cargar el archivo CSV usando pandas
        try:
            data = pd.read_csv(ruta_archivo_csv, skiprows=3)  # Saltar las primeras tres filas
        except Exception as e:
            return None
        
        # Limitar los datos a las últimas 'limite_filas' si se especifica
        if limite_filas is not None and len(data) > limite_filas:
            data = data.tail(limite_filas)
        
        # Eliminar la primera columna (considerada fecha) y convertir las restantes a numéricas
        data = data.iloc[:, 1:]
        for i in range(data.shape[1]):
            data.iloc[:, i] = pd.to_numeric(data.iloc[:, i], errors='coerce')
        
        # Eliminar columnas con valores NaN
        data.dropna(axis=1, inplace=True)
        
        # Validar que todavía hay suficientes columnas después de la limpieza
        if data.shape[1] < 1:
            return None
        
        # Extraer las columnas excepto la fecha
        columnas = data.columns
        
        # Diccionario para almacenar resultados de cada columna
        resultados = {}
        mejor_columna = None
        mejor_snr = -np.inf
        
        # Iterar sobre cada columna para analizarla
        for i, columna in enumerate(columnas):
            try:
                # Validar que la serie tiene datos suficientes para el análisis
                serie = data[columna]
                if len(serie) < 2:
                    continue
                
                # Calcular estadísticas
                media = serie.mean()
                desviacion = serie.std()
                snr = (media / desviacion) ** 2 if desviacion != 0 else np.nan
                ruido_normalizado = 1 / snr if snr != 0 else np.nan
                desviacion_ruido = np.sqrt(ruido_normalizado) * desviacion if ruido_normalizado != 0 else np.nan
                amplitud_promedio = desviacion_ruido * np.sqrt(2 / np.pi) if ruido_normalizado != 0 else np.nan
                
                # Cálculo de retornos
                retornos = serie.diff().abs().dropna()
                promedio_retornos = retornos.mean()
                
                # Análisis adicional
                # Exponente de Hurst
                hurst_exponent = nolds.hurst_rs(serie)
                
                # Detrended Fluctuation Analysis (DFA)
                dfa = nolds.dfa(serie)
                
                # Análisis de coherencia Wavelet
                coeficientes, _ = pywt.cwt(serie, scales=np.arange(1, 128), wavelet='morl')
                potencia_wavelet = np.sum(np.abs(coeficientes) ** 2, axis=1)
                
                # Análisis de Fourier
                espectro = np.abs(fft(serie))
                frecuencias = np.fft.fftfreq(len(serie))
                picos, _ = find_peaks(espectro)
                picos_principales = sorted(picos, key=lambda x: espectro[x], reverse=True)[:5]
                
                # Almacenar resultados en el diccionario
                resultados[f'col{i}'] = {
                    "media": media,
                    "desviacion_std": desviacion,
                    "SNR": snr,
                    "ruido_normalizado": ruido_normalizado,
                    "desviacion_ruido": desviacion_ruido,
                    "amplitud_promedio": amplitud_promedio,
                    "promedio_retornos": promedio_retornos,
                    "hurst_exponent": hurst_exponent,
                    "dfa": dfa,
                    "potencia_wavelet": potencia_wavelet,
                    "picos_principales": [frecuencias[pico] for pico in picos_principales]
                }
                
                # Seleccionar la mejor columna para predicción (con mayor SNR)
                if snr > mejor_snr:
                    mejor_snr = snr
                    mejor_columna = f'col{i}'
                
            except Exception as e:
                continue
        
        # Evaluar la calidad del dataset para cada escenario
        calificaciones = evaluar_dataset(resultados)
        
        # Generar resumen en una tabla y graficar análisis
        generar_resumen(resultados, periodicidad)
        
        # Resumen del dataset para la mejor columna
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
        return None

# Funciones auxiliares para visualización y análisis
def evaluar_dataset(resultados):
    calificaciones = {}
    for columna, stats in resultados.items():
        # Criterios de evaluación para cada escenario
        prediccion_tendencias = "Alta calidad" if stats['SNR'] > 10 and stats['hurst_exponent'] > 0.5 else "Baja calidad"
        balanceo_portafolios = "Adecuada" if stats['dfa'] < 1.5 else "No adecuada"
        trading_automatico = "Buena" if stats['potencia_wavelet'].mean() > 1000 else "Baja"
        
        calificaciones[columna] = {
            "Predicción de Tendencias": prediccion_tendencias,
            "Balanceo de Portafolios": balanceo_portafolios,
            "Trading Automático": trading_automatico
        }
    return calificaciones

def generar_resumen(resultados, periodicidad):
    resumen = pd.DataFrame(resultados).T
    resumen_path = f"output/resumen_{periodicidad}.csv"
    resumen.to_csv(resumen_path)
    print(f"Resumen generado y guardado en: {resumen_path}")
    print(resumen)

def generar_tabla_resumen(resumen_general):
    # Crear un DataFrame resumen de todos los datasets
    df_resumen = pd.DataFrame(resumen_general)
    resumen_path = "output/resumen_general.csv"
    df_resumen.to_csv(resumen_path, index=False)
    print(f"Resumen general generado y guardado en: {resumen_path}")
    print(df_resumen)

# Llamar a la función principal para iniciar el análisis
descargar_y_procesar_datasets()