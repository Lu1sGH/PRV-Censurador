import time #Medicion de tiempos de entrenamiento
import pandas as pd #Lectura eficiente de conjuntos de datos
import numpy as np #Manejo de arreglos numericos
import torch #Calculo tensorial en tarjeta grafica
from Censurador import Censurador #Censurador con modelo neuronal

RUTA_DATASET_292K = r"C:\Users\hpsg2\Downloads\0000(1).parquet" #Soporta .csv, .tsv o .parquet
COLUMNA_TEXTO = "text" #Columna de texto de entrada
COLUMNA_BINARIA = "foe_class" #Columna de decision binaria unica
RUTA_SALIDA_PT = "pesos_foe_binario.pt"
MODELO_BASE = "dccuchile/distilbert-base-spanish-uncased"


def main():
    """Ejecuta la lectura del conjunto de datos y el ajuste fino del clasificador binario"""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    t0 = time.perf_counter()
    print(f"--> [1/4] Leyendo conjunto de datos desde '{RUTA_DATASET_292K}'...")

    if RUTA_DATASET_292K.endswith(".parquet"):
        df = pd.read_parquet(RUTA_DATASET_292K, columns=[COLUMNA_TEXTO, COLUMNA_BINARIA])
    elif RUTA_DATASET_292K.endswith(".tsv"):
        df = pd.read_csv(RUTA_DATASET_292K, sep="\t", usecols=[COLUMNA_TEXTO, COLUMNA_BINARIA])
    else:
        df = pd.read_csv(RUTA_DATASET_292K, usecols=[COLUMNA_TEXTO, COLUMNA_BINARIA])

    df = df.dropna(subset=[COLUMNA_TEXTO, COLUMNA_BINARIA])
    listaTextos = df[COLUMNA_TEXTO].astype(str).tolist()
    vectorBinarioY = df[COLUMNA_BINARIA].astype(np.float32).to_numpy()

    total = len(listaTextos)
    positivos = int(np.sum(vectorBinarioY == 1.0))
    print(f"--> Muestras validas: {total:,} | Positivos D.D.O. (1): {positivos:,} | Neutros (0): {total - positivos:,}")

    print("--> [2/4] Inicializando Censurador con arquitectura Transformer + MLP...")
    censurador = Censurador(
        pais='MX',
        ruta_pesos_foe=RUTA_SALIDA_PT,
        modelo_transformer=MODELO_BASE
    )

    print("--> [3/4] Iniciando ajuste fino con precision mixta...")
    tTrain = time.perf_counter()
    censurador.entrenar_foe_offline(
        lista_textos_292k=listaTextos,
        vector_binario_y=vectorBinarioY,
        epocas=2,
        batch_size=128,
        lr=3e-4
    )
    print(f"--> Entrenamiento completado en {(time.perf_counter() - tTrain) / 60.0:.2f} minutos.")

    print(f"--> [4/4] Verificando inferencia multimodal sobre '{RUTA_SALIDA_PT}'...")
    oracionPrueba = "esa gente son una plaga inferior y deberian eliminarlos"
    palabrasSimuladas = [
        {"word": w, "start": i * 0.4, "end": (i + 1) * 0.4, "probability": 0.98, "gamma": 1.35}
        for i, w in enumerate(oracionPrueba.split())
    ]

    textoCens, silenciadas = censurador.evaluar_multimodal_foe(oracionPrueba, palabrasSimuladas)
    print(f"    Entrada : {oracionPrueba}")
    print(f"    Salida  : {textoCens}")
    print(f"    Cortes  : {[p['word'] for p in silenciadas]}")
    print(f"--> Proceso total finalizado en {(time.perf_counter() - t0) / 60.0:.2f} minutos.")


if __name__ == "__main__":
    main()

    main()