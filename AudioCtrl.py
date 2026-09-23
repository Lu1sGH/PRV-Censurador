import sounddevice as sd
import soundfile as sf
import numpy as np
import queue
import librosa
import librosa.display

class AudioCtrl:
    def __init__(self, sampleRate=16000, channels=1):
        #Utilizamos 16000 Hz debido a que Whisper funciona mejor con esa frecuencia y es suficiente para voz
        self.sampleRate = sampleRate
        self.channels = channels
        self.q = queue.Queue()
        self.stream = None
        
        #Arreglos para el audio original y el censurado
        self.audioData = np.array([], dtype='float32')
        self.audioCensurado = np.array([], dtype='float32')

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
            
            bloques = []
            while not self.q.empty():
                bloques.append(self.q.get())
                
            if bloques:
                self.audioData = np.concatenate(bloques, axis=0)
                self.audioCensurado = self.audioData.copy()
                
                # Precalcular representaciones costosas (como el espectrograma)
                self.precalcularGraficas()
            
            print("--> Grabación detenida.")

    def precalcularGraficas(self):
        """Calcula el STFT original de una sola vez para no recalcularlo."""
        if self.audioData is None or len(self.audioData) == 0:
            return
            
        self.n_fft = 2048
        self.hop_length = 512
        
        # Calculamos STFT original
        self.stftOriginal = librosa.stft(self.audioData.flatten(), n_fft=self.n_fft, hop_length=self.hop_length)
        # La versión censurada iniciará como una copia del original
        self.stftCensurado = self.stftOriginal.copy()

    def normalizarAudio(self):
        """Normaliza la señal de audio dividiéndola por su amplitud máxima."""
        if self.audioData is not None and len(self.audioData) > 0:
            amplitudMaxima = np.max(np.abs(self.audioData))
            if amplitudMaxima > 0:
                self.audioData = self.audioData / amplitudMaxima
                
                # Al normalizar la onda, también debemos actualizar los cálculos base
                self.audioCensurado = self.audioData.copy()
                self.precalcularGraficas()
            print("--> Audio normalizado.")

    def aplicarCensuraAudio(self, palabrasASilenciar):
        """Silencia la onda y modifica el espectrograma precalculado."""
        self.audioCensurado = self.audioData.copy()
        self.stftCensurado = self.stftOriginal.copy()
        
        for p in palabrasASilenciar:
            # 1. Modificar la forma de onda
            inicioMuestra = int(p["start"] * self.sampleRate)
            finMuestra = int(p["end"] * self.sampleRate)
            self.audioCensurado[inicioMuestra:finMuestra] = 0
            
            # 2. Modificar el espectrograma precalculado (STFT)
            # Cada columna del STFT representa hop_length muestras
            inicioCol = int(inicioMuestra / self.hop_length)
            finCol = int(finMuestra / self.hop_length)
            
            # Volver las frecuencias de ese bloque casi 0 (1e-10 para evitar errores en log)
            self.stftCensurado[:, inicioCol:finCol] = 1e-10
            
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

    def graficarAudio(self, ax, tipo="onda", censurado=False):
        """Usa librosa para graficar usando datos precalculados en los ejes de Matplotlib."""
        if self.audioData is None or len(self.audioData) == 0:
            return
            
        if tipo == "onda":
            audio = self.audioCensurado if censurado else self.audioData
            librosa.display.waveshow(audio.flatten(), sr=self.sampleRate, ax=ax, color='red' if censurado else 'blue')
            ax.set_ylabel("Amplitud")
            
        elif tipo == "espectrograma":
            stft = self.stftCensurado if censurado else self.stftOriginal
            magnitud = np.abs(stft)
            espectrograma_db = librosa.amplitude_to_db(magnitud, ref=np.max)
            
            librosa.display.specshow(espectrograma_db, 
                                     sr=self.sampleRate, 
                                     hop_length=self.hop_length, 
                                     x_axis="time", 
                                     y_axis="log", 
                                     ax=ax, 
                                     cmap='inferno' if censurado else 'viridis')
            ax.set_ylabel("Frec. (Hz)")
            
        titulo = "Censurado" if censurado else "Original"
        ax.set_title(f"Forma de onda {titulo}" if tipo == "onda" else f"Espectrograma {titulo}")
        ax.set_xlabel("Tiempo (s)")
