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
        "rsalaschile/forex-eurusd-dataset",
        "gabrielmv/eurusd-daily-historical-data-20012019"
    ]
    
    for dataset in datasets:
        try:
            print(f"[INFO] Descargando dataset: {dataset}")  # Mensaje de descarga
            path = kagglehub.dataset_download(dataset)
            print("[DEBUG] Path to dataset files:", path)
            
            # Convertir path en un objeto Path si es necesario
            path = Path(path)

            # Verificar si la descarga fue exitosa
            if not os.path.exists(path):
                print(f"[ERROR] La ruta de descarga no existe: {path}")
                continue
            
            # Aquí asumimos que el dataset tiene un archivo CSV principal
            csv_files = [file for file in path.glob('**/*.csv')]
            print(f"[DEBUG] Archivos CSV encontrados: {csv_files}")  # Agregar chequeo para archivos CSV
            if csv_files:
                print(f"[INFO] Analizando el archivo CSV: {csv_files[0]}")
                analizar_archivo_csv(csv_files[0], 4500)
            else:
                print(f"[ERROR] No se encontró archivo CSV en el dataset {dataset}")
        except Exception as e:
            print(f"[ERROR] Error durante la descarga o procesamiento del dataset {dataset}: {e}")

# Definir la función principal que analizará el archivo CSV
# Ahora también acepta un parámetro de límite de filas

def analizar_archivo_csv(ruta_archivo_csv, limite_filas=None):
    try:
        # Cargar el archivo CSV usando pandas
        try:
            data = pd.read_csv(ruta_archivo_csv)
            print(f"[DEBUG] Tamaño del DataFrame original: {data.shape}")  # Debug: Tamaño inicial del archivo
        except Exception as e:
            print(f"[ERROR] Error al cargar el archivo CSV: {e}")
            return
        
        # Limitar los datos a las últimas 'limite_filas' si se especifica
        if limite_filas is not None and len(data) > limite_filas:
            data = data.tail(limite_filas)
            print(f"[DEBUG] Tamaño del DataFrame después de limitar a {limite_filas} filas: {data.shape}")
        
        # Validar que el archivo tiene al menos dos columnas (fecha y datos)
        if data.shape[1] < 2:
            print("[ERROR] El archivo CSV debe tener al menos dos columnas: fecha y una columna de datos.")
            return
        
        # Extraer la fecha y eliminar la primera fila (que asumimos que es el encabezado)
        data = data.iloc[1:]  # Ignorar la primera fila, que es el encabezado
        data.columns = data.columns.str.strip()  # Eliminar espacios del encabezado
        print(f"[DEBUG] Tamaño del DataFrame sin la primera fila: {data.shape}")  # Debug: Tamaño tras eliminar la primera fila
        
        # Extraer las columnas excepto la fecha
        columnas = data.columns[1:]
        
        # Diccionario para almacenar resultados de cada columna
        resultados = {}
        
        # Iterar sobre cada columna para analizarla
        for columna in columnas:
            try:
                print(f"[INFO] Analizando columna: {columna}")  # Debug: Nombre de la columna actual
                
                # Convertir a datos numéricos y eliminar valores nulos
                serie = pd.to_numeric(data[columna], errors='coerce').dropna()
                print(f"[DEBUG] Tamaño de la serie sin nulos para columna '{columna}': {len(serie)}")  # Debug: Tamaño de la serie
                
                # Validar que la serie tiene datos suficientes para el análisis
                if len(serie) < 2:
                    print(f"[ERROR] La columna '{columna}' no tiene suficientes datos para el análisis.")
                    continue
                
                # Calcular estadísticas
                media = serie.mean()
                desviacion = serie.std()
                snr = (media / desviacion) ** 2 if desviacion != 0 else np.nan
                ruido_normalizado = 1 / snr if snr != 0 else np.nan
                desviacion_ruido = np.sqrt(ruido_normalizado) * desviacion if ruido_normalizado != 0 else np.nan
                amplitud_promedio = desviacion_ruido * np.sqrt(2 / np.pi) if ruido_normalizado != 0 else np.nan
                print(f"[DEBUG] Estadísticas calculadas para '{columna}': Media={media}, Desviación={desviacion}, SNR={snr}")
                
                # Cálculo de retornos
                retornos = serie.diff().abs().dropna()
                promedio_retornos = retornos.mean()
                print(f"[DEBUG] Promedio de retornos para '{columna}': {promedio_retornos}")
                
                # Análisis adicional
                # Exponente de Hurst
                hurst_exponent = nolds.hurst_rs(serie)
                print(f"[DEBUG] Exponente de Hurst para '{columna}': {hurst_exponent}")
                
                # Detrended Fluctuation Analysis (DFA)
                dfa = nolds.dfa(serie)
                print(f"[DEBUG] DFA para '{columna}': {dfa}")
                
                # Análisis de autocorrelación (Manual usando pandas)
                plt.figure(figsize=(12, 6))
                pd.plotting.autocorrelation_plot(serie)
                plt.title(f'Función de Autocorrelación (ACF) - {columna}')
                plt.xlabel('Lags')
                plt.ylabel('Autocorrelación')
                plt.grid(True)
                plt.savefig(f'output/{columna}_acf_plot.png')
                plt.close()
                
                # Entropía espectral
                espectro = np.abs(fft(serie))
                espectro_normalizado = espectro / espectro.sum()
                entropia_espectral = -np.sum(espectro_normalizado * np.log2(espectro_normalizado + 1e-10))
                print(f"[DEBUG] Entropía espectral para '{columna}': {entropia_espectral}")
                
                # Análisis de coherencia Wavelet
                coeficientes, _ = pywt.cwt(serie, scales=np.arange(1, 128), wavelet='morl')
                potencia_wavelet = np.sum(np.abs(coeficientes) ** 2, axis=1)
                print(f"[DEBUG] Potencia wavelet para '{columna}': {potencia_wavelet.mean()}")
                
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
                
                # Visualización de la columna
                analizar_tendencia_estacionalidad_residuos(serie, columna, save_path=f'output/{columna}_trend.png')
                analizar_distribucion(serie, retornos, columna, save_path=f'output/{columna}_distribution.png')
                analizar_fourier(serie, columna, save_path=f'output/{columna}_fourier.png')
                
            except Exception as e:
                print(f"[ERROR] Error al analizar la columna '{columna}': {e}")  # Debug: Mensaje de error para la columna
        
        # Evaluar la calidad del dataset para cada escenario
        evaluar_dataset(resultados)
        
        # Generar resumen en una tabla
        generar_resumen(resultados, ruta_archivo_csv)
        
        return resultados
    
    except FileNotFoundError:
        print("[ERROR] El archivo especificado no se encontró. Por favor verifique la ruta.")
    except pd.errors.EmptyDataError:
        print("[ERROR] El archivo CSV está vacío.")
    except Exception as e:
        print(f"[ERROR] Ocurrió un error inesperado: {e}")

# Funciones auxiliares para visualización y análisis

def analizar_tendencia_estacionalidad_residuos(serie, columna, save_path):
    try:
        # Graficar la tendencia y residuos de la serie
        plt.figure(figsize=(10, 6))
        plt.plot(serie, label='Serie Temporal')
        plt.title(f'Tendencia, Estacionalidad y Residuos - {columna}')
        plt.xlabel('Tiempo')
        plt.ylabel('Valor')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"[ERROR] Error al graficar tendencia, estacionalidad y residuos para '{columna}': {e}")

def analizar_distribucion(serie, retornos, columna, save_path):
    try:
        # Graficar distribución de la serie
        plt.figure(figsize=(10, 6))
        sns.histplot(serie, kde=True)
        plt.title(f'Distribución de {columna}')
        plt.xlabel('Valor')
        plt.ylabel('Frecuencia')
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        
        # Graficar distribución de retornos
        plt.figure(figsize=(10, 6))
        sns.histplot(retornos, kde=True)
        plt.title(f'Distribución de Retornos - {columna}')
        plt.xlabel('Retorno')
        plt.ylabel('Frecuencia')
        plt.grid(True)
        plt.savefig(save_path.replace("distribution", "returns_distribution"))
        plt.close()
    except Exception as e:
        print(f"[ERROR] Error al graficar distribución para '{columna}': {e}")

def analizar_fourier(serie, columna, save_path):
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
        plt.title(f'Espectro de Fourier - {columna}')
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
        if stats['SNR'] > 10 and stats['hurst_exponent'] > 0.5:
            print("  [PREDICCIÓN DE TENDENCIAS]: Alta calidad para predicción de tendencias.")
        else:
            print("  [PREDICCIÓN DE TENDENCIAS]: Baja calidad para predicción de tendencias.")
        
        if stats['dfa'] < 1.5:
            print("  [BALANCEO DE PORTAFOLIOS]: La serie muestra características estables, adecuada para balanceo de portafolios.")
        else:
            print("  [BALANCEO DE PORTAFOLIOS]: Serie altamente volátil, menor estabilidad para balanceo.")
        
        if stats['potencia_wavelet'].mean() > 1000:
            print("  [TRADING AUTOMÁTICO]: Buena coherencia en frecuencias de corto plazo, adecuado para trading en vivo.")
        else:
            print("  [TRADING AUTOMÁTICO]: Baja coherencia en frecuencias de corto plazo, menos adecuado para trading.")

def generar_resumen(resultados, ruta_archivo_csv):
    resumen = pd.DataFrame(resultados).T
    resumen_path = f"output/resumen_{Path(ruta_archivo_csv).stem}.csv"
    resumen.to_csv(resumen_path)
    print(f"[INFO] Resumen generado y guardado en: {resumen_path}")

# Llamar a la función principal para iniciar el análisis
descargar_y_procesar_datasets()
