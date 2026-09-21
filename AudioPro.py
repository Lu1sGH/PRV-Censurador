import sounddevice as sd
import soundfile as sf
import numpy as np
import queue
import whisper

class AudioPro:
    def __init__(self, sampleRate=16000, channels=1):
        #Utilizamos 16000 Hz debido a que Whisper funciona mejor con esa frecuencia y es suficiente para voz
        self.sampleRate = sampleRate
        self.channels = channels
        self.q = queue.Queue()
        self.stream = None
        
        #Arreglos para el audio original y el censurado
        self.audioData = np.array([], dtype='float32')
        self.audioCensurado = np.array([], dtype='float32')
        
        print("--> Cargando modelo Whisper (puede tardar la primera vez)...")
        #Cambiamos a 'small' para mucha mayor precisión en español (similar a Google)
        self.modelo = whisper.load_model("small")
        print("--> Modelo Whisper cargado.")

    def callbackGrabacion(self, inData, frames, time, status):
        """Callback que almacena en la cola los bloques del micrófono."""
        if status:
            print(status)
        self.q.put(inData.copy())

    def iniciarGrabacion(self):
        """Inicia el flujo de entrada desde el micrófono."""
        self.audioData = np.array([], dtype='float32') #Reiniciar datos
        self.audioCensurado = np.array([], dtype='float32')
        self.q = queue.Queue()
        self.stream = sd.InputStream(samplerate=self.sampleRate, 
                                     channels=self.channels,
                                     callback=self.callbackGrabacion)
        self.stream.start()
        print("--> Grabación iniciada...")

    def detenerGrabacion(self):
        """Detiene el micrófono y concatena los bloques capturados."""
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            
            #Recopilamos todos los bloques
            bloques = []
            while not self.q.empty():
                bloques.append(self.q.get())
                
            if bloques:
                self.audioData = np.concatenate(bloques, axis=0)
                #Inicialmente, el censurado es igual al original
                self.audioCensurado = self.audioData.copy()
            
            print("--> Grabación detenida.")

    def normalizarAudio(self):
        """Normaliza la señal de audio dividiéndola por su amplitud máxima."""
        if self.audioData is not None and len(self.audioData) > 0:
            amplitudMaxima = np.max(np.abs(self.audioData))
            if amplitudMaxima > 0:
                self.audioData = self.audioData / amplitudMaxima
            print("--> Audio normalizado.")

    def transcribirAudio(self):
        """Transcribe usando Whisper para obtener texto y tiempos (ms) de cada palabra."""
        if self.audioData is None or len(self.audioData) == 0:
            return "", []
            
        print("--> Transcribiendo y obteniendo tiempos con Whisper...")
        
        #Se normalizar el audio para que el volumen bajo no afecte a Whisper
        self.normalizarAudio()
        
        audio1D = self.audioData.flatten().astype(np.float32)
        
        #word_timestamps=True es clave para saber en qué milisegundo ocurre la palabra
        resultado = self.modelo.transcribe(audio1D, language="es", word_timestamps=True)
        
        textoCompleto = resultado["text"].strip()
        palabrasConTiempo = []
        
        #Extraemos cada palabra con su inicio y fin en segundos
        for segmento in resultado.get("segments", []):
            for palabra_info in segmento.get("words", []):
                palabrasConTiempo.append({
                    "word": palabra_info["word"].strip(),
                    "start": palabra_info["start"],
                    "end": palabra_info["end"]
                })
                
        return textoCompleto, palabrasConTiempo

    def aplicarCensuraAudio(self, palabrasASilenciar):
        """Silencia (0) los intervalos de audio correspondientes a las palabras censuradas."""
        self.audioCensurado = self.audioData.copy()
        
        for p in palabrasASilenciar:
            #Convertimos el tiempo (segundos) a índice de muestra (multiplicando por sampleRate)
            inicioMuestra = int(p["start"] * self.sampleRate)
            finMuestra = int(p["end"] * self.sampleRate)
            
            #Volvemos 0 (silencio) ese fragmento del arreglo
            self.audioCensurado[inicioMuestra:finMuestra] = 0
            
        print(f"--> Audio censurado: se silenciaron {len(palabrasASilenciar)} palabra(s).")

    def reproducirAudioOriginal(self):
        """Reproduce la señal original."""
        if self.audioData is not None and len(self.audioData) > 0:
            print("--> Reproduciendo audio original...")
            sd.play(self.audioData, self.sampleRate)
        else:
            print("--> No hay audio original para reproducir.")

    def reproducirAudioCensurado(self):
        """Reproduce la señal censurada (silenciada)."""
        if self.audioCensurado is not None and len(self.audioCensurado) > 0:
            print("--> Reproduciendo audio censurado...")
            sd.play(self.audioCensurado, self.sampleRate)
        else:
            print("--> No hay audio censurado para reproducir.")
