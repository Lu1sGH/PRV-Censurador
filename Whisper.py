import whisper #Modelo de voz a texto
import numpy as np #Manejo de arreglos
import torch #Tensores para inteligencia artificial
import torchaudio #Respaldo de remuestreo si la entrada no viene a 16kHz


class WhisperTranscriptor:
    def __init__(self, modeloNombre="small"):
        """Inicializa el modelo de Whisper y lo carga en memoria"""
        #Detectar si hay tarjeta grafica disponible
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"--> Usando dispositivo: {self.device.upper()}")
        print(f"--> Cargando modelo Whisper '{modeloNombre}' (puede tardar la primera vez)...")
        self.modelo = whisper.load_model(modeloNombre, device=self.device)
        print("--> Modelo Whisper cargado exitosamente.")

    def transcribir(self, audioData, sampleRate=16000):
        """Transcribe el audio preprocesado con Whisper devolviendo texto, marcas de tiempo y probabilidades"""
        if audioData is None or len(audioData) == 0:
            return "", []

        print("--> Transcribiendo y obteniendo tiempos con Whisper...")

        audio1D = audioData.flatten().astype(np.float32)

        #Respaldo en caso de que el audio no haya pasado antes por preprocesarAudio a 16000 Hz
        if sampleRate != 16000:
            tensorAudio = torch.from_numpy(audio1D).unsqueeze(0).to(self.device)
            resampler = torchaudio.transforms.Resample(
                orig_freq=sampleRate,
                new_freq=16000
            ).to(self.device)
            audio1D = resampler(tensorAudio).squeeze(0).cpu().numpy()

        #La bandera word_timestamps obtiene el intervalo exacto de cada palabra
        usarFp16 = self.device == "cuda"
        resultado = self.modelo.transcribe(
            audio1D,
            language="es",
            word_timestamps=True,
            fp16=usarFp16
        )

        textoCompleto = resultado["text"].strip()
        palabrasConTiempo = []

        #Extraemos cada palabra con su inicio, fin en segundos y probabilidad del modelo
        for segmento in resultado.get("segments", []):
            for palabraInfo in segmento.get("words", []):
                palabrasConTiempo.append({
                    "word": palabraInfo["word"].strip(),
                    "start": float(palabraInfo["start"]),
                    "end": float(palabraInfo["end"]),
                    "probability": float(palabraInfo.get("probability", 1.0))
                })

        return textoCompleto, palabrasConTiempo
