# Importar librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fft import fft
from scipy.signal import find_peaks
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pywt
import nolds
import warnings

# Configuración para ignorar advertencias de numpy/pandas que no afectan el procesamiento
warnings.filterwarnings("ignore")

# Definir la función principal que analizará el archivo CSV
def analizar_archivo_csv(ruta_archivo_csv):
    try:
        # Cargar el archivo CSV usando pandas
        data = pd.read_csv(ruta_archivo_csv)
        print(f"[DEBUG] Tamaño del DataFrame original: {data.shape}")  # Debug: Tamaño inicial del archivo
        
        # Validar que el archivo tiene al menos dos columnas (fecha y datos)
        if data.shape[1] < 2:
            raise ValueError("El archivo CSV debe tener al menos dos columnas: fecha y una columna de datos.")
        
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
                print(f"[DEBUG] Analizando columna: {columna}")  # Debug: Nombre de la columna actual
                
                # Convertir a datos numéricos y eliminar valores nulos
                serie = pd.to_numeric(data[columna], errors='coerce').dropna()
                print(f"[DEBUG] Tamaño de la serie sin nulos para columna '{columna}': {len(serie)}")  # Debug: Tamaño de la serie
                
                # Validar que la serie tiene datos suficientes para el análisis
                if len(serie) < 2:
                    raise ValueError(f"La columna '{columna}' no tiene suficientes datos para el análisis.")
                
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
                
                # Análisis de autocorrelación (ACF/PACF)
                acf_fig = plot_acf(serie, lags=20)
                pacf_fig = plot_pacf(serie, lags=20)
                plt.show()
                
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
                
                # Visualización de la columna
                analizar_tendencia_estacionalidad_residuos(serie, columna)
                analizar_distribucion(serie, retornos, columna)
                analizar_fourier(serie, columna)
                
            except Exception as e:
                print(f"[ERROR] Error al analizar la columna '{columna}': {e}")  # Debug: Mensaje de error para la columna
        
        # Evaluar la calidad del dataset para cada escenario
        evaluar_dataset(resultados)
        
        return resultados
    
    except FileNotFoundError:
        print("[ERROR] El archivo especificado no se encontró. Por favor verifique la ruta.")
    except pd.errors.EmptyDataError:
        print("[ERROR] El archivo CSV está vacío.")
    except Exception as e:
        print(f"[ERROR] Ocurrió un error inesperado: {e}")

# Funciones auxiliares para visualización y análisis

def analizar_tendencia_estacionalidad_residuos(serie, columna):
    try:
        # Graficar la tendencia y residuos de la serie
        plt.figure(figsize=(10, 6))
        plt.plot(serie, label='Serie Temporal')
        plt.title(f'Tendencia, Estacionalidad y Residuos - {columna}')
        plt.xlabel('Tiempo')
        plt.ylabel('Valor')
        plt.legend()
        plt.grid(True)
        plt.show()
    except Exception as e:
        print(f"[ERROR] Error al graficar tendencia, estacionalidad y residuos para '{columna}': {e}")

def analizar_distribucion(serie, retornos, columna):
    try:
        # Graficar distribución de la serie
        plt.figure(figsize=(10, 6))
        sns.histplot(serie, kde=True)
        plt.title(f'Distribución de {columna}')
        plt.xlabel('Valor')
        plt.ylabel('Frecuencia')
        plt.grid(True)
        plt.show()
        
        # Graficar distribución de retornos
        plt.figure(figsize=(10, 6))
        sns.histplot(retornos, kde=True)
        plt.title(f'Distribución de Retornos - {columna}')
        plt.xlabel('Retorno')
        plt.ylabel('Frecuencia')
        plt.grid(True)
        plt.show()
    except Exception as e:
        print(f"[ERROR] Error al graficar distribución para '{columna}': {e}")

def analizar_fourier(serie, columna):
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
        plt.show()
        
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

# Ejemplo de uso del programa
if __name__ == "__main__":
    ruta_archivo = 'ruta_al_archivo.csv'  # Cambiar por la ruta del archivo CSV
    resultados = analizar_archivo_csv(ruta_archivo)
    if resultados:
        print("\n[RESULTADOS FINALES]")
        for columna, stats in resultados.items():
            print(f"\nColumna: {columna}")
            for key, value in stats.items():
                print(f"  {key}: {value}")