import whisper  # Modelo de voz a texto
import numpy as np  # Manejo de arreglos
import torch  # Tensores para inteligencia artificial
import torchaudio  # Procesamiento de audio para tensores


class WhisperTranscriptor:
    def __init__(self, modeloNombre="small"):
        """Inicializa el modelo de Whisper y lo carga en memoria"""
        # Detectar si hay GPU disponible
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"--> Usando dispositivo: {self.device.upper()}")
        print(f"--> Cargando modelo Whisper '{modeloNombre}' (puede tardar la primera vez)...")
        self.modelo = whisper.load_model(modeloNombre, device=self.device)
        print("--> Modelo Whisper cargado exitosamente.")

    def transcribir(self, audioData, sampleRate):
        """Transcribe audio usando Whisper y devuelve texto, tiempos y probabilidad de cada palabra"""
        if audioData is None or len(audioData) == 0:
            return "", []

        print("--> Transcribiendo y obteniendo tiempos con Whisper...")

        audio1D = audioData.flatten().astype(np.float32)

        # Whisper requiere que el audio este a 16000 Hz
        # Hacemos un resample temporal a 16kHz en el dispositivo activo
        if sampleRate != 16000:
            tensorAudio = torch.from_numpy(audio1D).unsqueeze(0).to(self.device)
            resampler = torchaudio.transforms.Resample(
                orig_freq=sampleRate, new_freq=16000
            ).to(self.device)
            tensorResampleado = resampler(tensorAudio)
            audio1D = tensorResampleado.squeeze(0).cpu().numpy()

        # La bandera word_timestamps permite saber en que milisegundo ocurre la palabra
        usar_fp16 = self.device == "cuda"
        resultado = self.modelo.transcribe(
            audio1D,
            language="es",
            word_timestamps=True,
            fp16=usar_fp16
        )

        textoCompleto = resultado["text"].strip()
        palabrasConTiempo = []

        # Extraemos cada palabra con su inicio, fin en segundos y probabilidad del ASR
        for segmento in resultado.get("segments", []):
            for palabraInfo in segmento.get("words", []):
                palabrasConTiempo.append({
                    "word": palabraInfo["word"].strip(),
                    "start": float(palabraInfo["start"]),
                    "end": float(palabraInfo["end"]),
                    "probability": float(palabraInfo.get("probability", 1.0))
                })

        return textoCompleto, palabrasConTiempo