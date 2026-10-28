import time
import pandas as pd
import numpy as np
import torch
from Censurador import Censurador

RUTA_DATASET_292K = r"C:\Users\hpsg2\Downloads\0000(1).parquet"  # Soporta .csv, .tsv o .parquet
COLUMNA_TEXTO = "text"                      # Nombre de la columna de texto
COLUMNA_BINARIA = "foe_class"                   # Nombre de la columna binaria única {0, 1}
RUTA_SALIDA_PT = "pesos_foe_binario.pt"


def main():
    t0 = time.perf_counter()
    print(f"--> [1/4] Cargando dataset desde '{RUTA_DATASET_292K}'...")

    # Carga eficiente solo de las 2 columnas requeridas (ignorando la terna {0,1}^3)
    if RUTA_DATASET_292K.endswith(".parquet"):
        df = pd.read_parquet(RUTA_DATASET_292K, columns=[COLUMNA_TEXTO, COLUMNA_BINARIA])
    elif RUTA_DATASET_292K.endswith(".tsv"):
        df = pd.read_csv(RUTA_DATASET_292K, sep="\t", usecols=[COLUMNA_TEXTO, COLUMNA_BINARIA])
    else:
        df = pd.read_csv(RUTA_DATASET_292K, usecols=[COLUMNA_TEXTO, COLUMNA_BINARIA])

    # Limpieza de valores nulos y conversión estricta a binario {0, 1}
    df = df.dropna(subset=[COLUMNA_TEXTO, COLUMNA_BINARIA])
    lista_textos = df[COLUMNA_TEXTO].astype(str).tolist()
    vector_binario_y = df[COLUMNA_BINARIA].astype(np.float32).to_numpy()

    total_muestras = len(lista_textos)
    positivos = int(np.sum(vector_binario_y == 1.0))
    negativos = total_muestras - positivos
    print(f"--> Total cargadas: {total_muestras:,} | D.D.O. (1): {positivos:,} | Neutras (0): {negativos:,}")

    # Inicializar Censurador (carga spanlp y prepara la estructura en GPU)
    print("--> [2/4] Inicializando arquitectura ClasificadorBinarioFoE en GPU...")
    censurador = Censurador(pais='MX', ruta_pesos_foe=RUTA_SALIDA_PT)

    # Entrenar sobre las 292K muestras y serializar pesos_foe_binario.pt
    print("--> [3/4] Ejecutando entrenamiento FoE Binario...")
    t_train_ini = time.perf_counter()
    censurador.entrenar_foe_offline(
        lista_textos_292k=lista_textos,
        vector_binario_y=vector_binario_y,
        epocas=25,
        batch_size=4096,
        lr=1e-2
    )
    t_train_fin = time.perf_counter()
    print(f"--> Tiempo de extracción + entrenamiento en GPU: {t_train_fin - t_train_ini:.2f} s")

    # Verificación rápida del archivo generado
    print(f"--> [4/4] Verificando integridad del archivo '{RUTA_SALIDA_PT}'...")
    oracion_prueba = "esa gente son una plaga inferior y deberian eliminarlos"
    palabras_simuladas = [
        {"word": w, "start": i * 0.4, "end": (i + 1) * 0.4, "probability": 0.98, "gamma": 1.35}
        for i, w in enumerate(oracion_prueba.split())
    ]

    texto_cens, silenciadas = censurador.evaluar_multimodal_foe(oracion_prueba, palabras_simuladas)
    print(f"    Oración de prueba : {oracion_prueba}")
    print(f"    Salida censurada  : {texto_cens}")
    print(f"    Palabras a cortar : {[p['word'] for p in silenciadas]}")
    print(f"--> Proceso completado en {time.perf_counter() - t0:.2f} s.")


if __name__ == "__main__":
    main()