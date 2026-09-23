import whisper #Modelo de voz a texto
import numpy as np #Manejo de arreglos
import torch #Tensores para inteligencia artificial
import torchaudio #Procesamiento de audio para tensores

class WhisperTranscriptor:
    def __init__(self, modeloNombre="small"):
        """Inicializa el modelo de Whisper y lo carga en memoria"""
        print(f"--> Cargando modelo Whisper '{modeloNombre}' (puede tardar la primera vez)...")
        self.modelo = whisper.load_model(modeloNombre)
        print("--> Modelo Whisper cargado exitosamente.")

    def transcribir(self, audioData, sampleRate):
        """Transcribe audio usando Whisper y devuelve texto y tiempos en milisegundos de cada palabra"""
        if audioData is None or len(audioData) == 0:
            return "", []
            
        print("--> Transcribiendo y obteniendo tiempos con Whisper...")
        
        audio1D = audioData.flatten().astype(np.float32)
        
        #Whisper requiere que el audio este a 16000 Hz
        #Hacemos un resample temporal a 16kHz solo para Whisper
        if sampleRate != 16000:
            #Convertimos a tensor de PyTorch
            tensorAudio = torch.from_numpy(audio1D).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(orig_freq=sampleRate, new_freq=16000)
            tensorResampleado = resampler(tensorAudio)
            audio1D = tensorResampleado.squeeze(0).numpy()
        
        #La bandera word_timestamps es clave para saber en que milisegundo ocurre la palabra
        resultado = self.modelo.transcribe(audio1D, language="es", word_timestamps=True)
        
        textoCompleto = resultado["text"].strip()
        palabrasConTiempo = []
        
        #Extraemos cada palabra con su inicio y fin en segundos
        for segmento in resultado.get("segments", []):
            for palabraInfo in segmento.get("words", []):
                palabrasConTiempo.append({
                    "word": palabraInfo["word"].strip(),
                    "start": palabraInfo["start"],
                    "end": palabraInfo["end"]
                })
                
        return textoCompleto, palabrasConTiempo

