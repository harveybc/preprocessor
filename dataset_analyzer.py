# Importar librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fft import fft
from scipy.signal import find_peaks
import pywt
import nolds
import warnings
import kagglehub
import os
from pathlib import Path

# Configuración para ignorar advertencias de numpy/pandas que no afectan el procesamiento
warnings.filterwarnings("ignore")

# Definir la función para descargar y cargar los últimos 4500 datos de cada dataset

def descargar_y_procesar_datasets():
    print("[INFO] Iniciando la descarga y procesamiento de los datasets...")  # Agregar mensaje de inicio
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
            print(f"[INFO] Descargando dataset: {dataset}")  # Mensaje de descarga
            path = kagglehub.dataset_download(dataset)
            
            # Convertir path en un objeto Path si es necesario
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
# Ahora también acepta un parámetro de límite de filas

def analizar_archivo_csv(ruta_archivo_csv, limite_filas=None):
    try:
        # Determinar la periodicidad del dataset a partir del nombre del archivo
        periodicidad = "1min" if "1minute" in str(ruta_archivo_csv).lower() else "15min" if "15min" in str(ruta_archivo_csv).lower() else "1h" if "1h" in str(ruta_archivo_csv).lower() else "1d"
        
        # Cargar el archivo CSV usando pandas
        try:
            print(f"[DEBUG] Cargando el archivo CSV desde la ruta: {ruta_archivo_csv}")
            data = pd.read_csv(ruta_archivo_csv, skiprows=3)  # Saltar las primeras tres filas
        except Exception as e:
            print(f"[ERROR] Error al cargar el archivo CSV: {e}")
            return None
        
        # Limitar los datos a las últimas 'limite_filas' si se especifica
        if limite_filas is not None and len(data) > limite_filas:
            data = data.tail(limite_filas)
            print(f"[DEBUG] Limitando los datos a las últimas {limite_filas} filas.")
        
        # Validar que el archivo tiene al menos dos columnas (fecha y datos)
        if data.shape[1] < 2:
            print("[ERROR] El archivo CSV debe tener al menos dos columnas: fecha y una columna de datos.")
            return None
        
        # Eliminar espacios de los encabezados y convertir todas las columnas excepto la primera a datos numéricos
        data.columns = data.columns.str.strip()  # Eliminar espacios del encabezado
        for columna in data.columns[1:]:
            print(f"[DEBUG] Convirtiendo la columna '{columna}' a numérico.")
            data[columna] = pd.to_numeric(data[columna], errors='coerce')
        
        # Eliminar filas con valores NaN después de la conversión
        data.dropna(inplace=True)
        print(f"[DEBUG] Datos después de eliminar filas con NaN: {data.shape}")
        
        # Validar que todavía hay suficientes filas después de la limpieza
        if data.shape[0] < 2:
            print("[ERROR] No hay suficientes datos para el análisis después de la limpieza de valores nulos.")
            return None
        
        # Asegurarse de que las columnas no tengan nombres duplicados manualmente
        print(f"[DEBUG] Asegurándose de que no haya nombres duplicados en las columnas.")
        print(f"[DEBUG] Columnas antes de la deduplicación: {data.columns.tolist()}")

        # Alternativa para eliminar nombres de columnas duplicados manualmente
        seen = set()
        new_columns = []
        for col in data.columns:
            if col in seen:
                count = 1
                new_col = f"{col}_{count}"
                while new_col in seen:
                    count += 1
                    new_col = f"{col}_{count}"
                new_columns.append(new_col)
                seen.add(new_col)
            else:
                new_columns.append(col)
                seen.add(col)
        data.columns = new_columns

        print(f"[DEBUG] Columnas después de la deduplicación: {data.columns.tolist()}")

        # Extraer las columnas excepto la fecha
        columnas = data.columns[1:]
        
        # Diccionario para almacenar resultados de cada columna
        resultados = {}
        mejor_columna = None
        mejor_snr = -np.inf
        
        # Iterar sobre cada columna para analizarla
        for columna in columnas:
            try:
                print(f"[INFO] Analizando columna: {columna}")  # Mensaje de información
                
                # Validar que la serie tiene datos suficientes para el análisis
                serie = data[columna]
                if len(serie) < 2:
                    print(f"[ERROR] La columna '{columna}' no tiene suficientes datos para el análisis.")
                    continue
                
                # Calcular estadísticas
                print(f"[DEBUG] Calculando estadísticas para la columna: {columna}")
                media = serie.mean()
                desviacion = serie.std()
                snr = (media / desviacion) ** 2 if desviacion != 0 else np.nan
                ruido_normalizado = 1 / snr if snr != 0 else np.nan
                desviacion_ruido = np.sqrt(ruido_normalizado) * desviacion if ruido_normalizado != 0 else np.nan
                amplitud_promedio = desviacion_ruido * np.sqrt(2 / np.pi) if ruido_normalizado != 0 else np.nan
                
                # Cálculo de retornos
                print(f"[DEBUG] Calculando retornos para la columna: {columna}")
                retornos = serie.diff().abs().dropna()
                promedio_retornos = retornos.mean()
                
                # Análisis adicional
                print(f"[DEBUG] Calculando análisis adicionales para la columna: {columna}")
                # Exponente de Hurst
                hurst_exponent = nolds.hurst_rs(serie)
                
                # Detrended Fluctuation Analysis (DFA)
                dfa = nolds.dfa(serie)
                
                # Análisis de autocorrelación (Manual usando pandas)
                plt.figure(figsize=(12, 6))
                pd.plotting.autocorrelation_plot(serie)
                plt.title(f'{periodicidad} - Función de Autocorrelación (ACF) - {columna}')
                plt.xlabel('Lags')
                plt.ylabel('Autocorrelación')
                plt.grid(True)
                plt.savefig(f'output/{periodicidad}_acf_plot_{columna}.png')
                plt.close()
                
                # Entropía espectral
                espectro = np.abs(fft(serie))
                espectro_normalizado = espectro / espectro.sum()
                entropia_espectral = -np.sum(espectro_normalizado * np.log2(espectro_normalizado + 1e-10))
                
                # Análisis de coherencia Wavelet
                coeficientes, _ = pywt.cwt(serie, scales=np.arange(1, 128), wavelet='morl')
                potencia_wavelet = np.sum(np.abs(coeficientes) ** 2, axis=1)
                
                # Almacenar resultados en el diccionario
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
                    "entropia_espectral": entropia_espectral,
                    "potencia_wavelet": potencia_wavelet
                }
                
                # Seleccionar la mejor columna para predicción (con mayor SNR)
                if snr > mejor_snr:
                    mejor_snr = snr
                    mejor_columna = columna
                
                # Visualización de la columna
                print(f"[DEBUG] Realizando visualizaciones para la columna: {columna}")
                analizar_tendencia_estacionalidad_residuos(serie, columna, periodicidad, save_path=f'output/{periodicidad}_trend_{columna}.png')
                analizar_distribucion(serie, retornos, columna, periodicidad, save_path=f'output/{periodicidad}_distribution_{columna}.png')
                analizar_fourier(serie, columna, periodicidad, save_path=f'output/{periodicidad}_fourier_{columna}.png')
                
            except ValueError as ve:
                print(f"[ERROR] {ve}")
            except Exception as e:
                print(f"[ERROR] Error al analizar la columna '{columna}': {e}")  # Mensaje de error para la columna
        
        # Evaluar la calidad del dataset para cada escenario
        calificaciones = evaluar_dataset(resultados)
        
        # Generar resumen en una tabla
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
    
    except FileNotFoundError:
        print("[ERROR] El archivo especificado no se encontró. Por favor verifique la ruta.")
    except pd.errors.EmptyDataError:
        print("[ERROR] El archivo CSV está vacío.")
    except Exception as e:
        print(f"[ERROR] Ocurrió un error inesperado: {e}")
    return None

# Funciones auxiliares para visualización y análisis

def analizar_tendencia_estacionalidad_residuos(serie, columna, periodicidad, save_path):
    try:
        # Graficar la tendencia y residuos de la serie
        plt.figure(figsize=(10, 6))
        plt.plot(serie, label='Serie Temporal')
        plt.title(f'{periodicidad} - Tendencia, Estacionalidad y Residuos - {columna}')
        plt.xlabel('Tiempo')
        plt.ylabel('Valor')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"[ERROR] Error al graficar tendencia, estacionalidad y residuos para '{columna}': {e}")

def analizar_distribucion(serie, retornos, columna, periodicidad, save_path):
    try:
        # Graficar distribución de la serie
        plt.figure(figsize=(10, 6))
        sns.histplot(serie, kde=True)
        plt.title(f'{periodicidad} - Distribución de {columna}')
        plt.xlabel('Valor')
        plt.ylabel('Frecuencia')
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        
        # Graficar distribución de retornos
        plt.figure(figsize=(10, 6))
        sns.histplot(retornos, kde=True)
        plt.title(f'{periodicidad} - Distribución de Retornos - {columna}')
        plt.xlabel('Retorno')
        plt.ylabel('Frecuencia')
        plt.grid(True)
        plt.savefig(save_path.replace("distribution", "returns_distribution"))
        plt.close()
    except Exception as e:
        print(f"[ERROR] Error al graficar distribución para '{columna}': {e}")

def analizar_fourier(serie, columna, periodicidad, save_path):
    try:
        # Realizar la Transformada de Fourier a la serie
        espectro = np.abs(fft(serie))
        frecuencias = np.fft.fftfreq(len(serie))
        
        # Encontrar los picos principales en el espectro de potencia
        picos, _ = find_peaks(espectro)
        picos_principales = sorted(picos, key=lambda x: espectro[x], reverse=True)[:5]
        
        # Graficar el espectro de Fourier y marcar los picos principales
        plt.figure(figsize=(10, 6))
        plt.plot(frecuencias, espectro)
        plt.scatter(frecuencias[picos_principales], espectro[picos_principales], color='red')
        plt.title(f'{periodicidad} - Espectro de Fourier - {columna}')
        plt.xlabel('Frecuencia')
        plt.ylabel('Potencia')
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        
        # Mostrar las frecuencias de los picos principales
        print(f"Principales frecuencias para {columna}:")
        for i, pico in enumerate(picos_principales, start=1):
            print(f"Pico {i}: Frecuencia = {frecuencias[pico]}, Potencia = {espectro[pico]}")
    except Exception as e:
        print(f"[ERROR] Error en el análisis de Fourier para '{columna}': {e}")

def evaluar_dataset(resultados):
    calificaciones = {}
    print("\n[CALIFICACIÓN DEL DATASET]")
    for columna, stats in resultados.items():
        print(f"\nColumna: {columna}")
        print(f"  Media: {stats['media']}")
        print(f"  Desviación estándar: {stats['desviacion_std']}")
        print(f"  SNR: {stats['SNR']}")
        print(f"  Entropía Espectral: {stats['entropia_espectral']}")
        print(f"  Exponente de Hurst: {stats['hurst_exponent']}")
        print(f"  DFA: {stats['dfa']}")
        print(f"  Potencia Wavelet: {stats['potencia_wavelet']}")
        
        # Criterios de evaluación para cada escenario
        prediccion_tendencias = "Alta calidad" if stats['SNR'] > 10 and stats['hurst_exponent'] > 0.5 else "Baja calidad"
        balanceo_portafolios = "Adecuada" if stats['dfa'] < 1.5 else "No adecuada"
        trading_automatico = "Buena" if stats['potencia_wavelet'].mean() > 1000 else "Baja"
        
        calificaciones[columna] = {
            "Predicción de Tendencias": prediccion_tendencias,
            "Balanceo de Portafolios": balanceo_portafolios,
            "Trading Automático": trading_automatico
        }
        
        print(f"  [PREDICCIÓN DE TENDENCIAS]: {prediccion_tendencias}")
        print(f"  [BALANCEO DE PORTAFOLIOS]: {balanceo_portafolios}")
        print(f"  [TRADING AUTOMÁTICO]: {trading_automatico}")
    return calificaciones

def generar_resumen(resultados, periodicidad):
    resumen = pd.DataFrame(resultados).T
    resumen_path = f"output/resumen_{periodicidad}.csv"
    resumen.to_csv(resumen_path)
    print(f"[INFO] Resumen generado y guardado en: {resumen_path}")

def generar_tabla_resumen(resumen_general):
    # Crear un DataFrame resumen de todos los datasets
    df_resumen = pd.DataFrame(resumen_general)
    resumen_path = "output/resumen_general.csv"
    df_resumen.to_csv(resumen_path, index=False)
    print(f"[INFO] Resumen general generado y guardado en: {resumen_path}")

# Llamar a la función principal para iniciar el análisis
descargar_y_procesar_datasets()
