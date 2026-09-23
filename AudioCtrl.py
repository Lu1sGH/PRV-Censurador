import sounddevice as sd #Captura de audio del microfono
import soundfile as sf #Lectura y escritura de archivos de audio
import numpy as np #Manejo de arreglos matematicos
import queue #Estructura de datos para audio
import librosa #Libreria de procesamiento de audio
import librosa.display #Visualizacion de audio

class AudioCtrl:
    def __init__(self, sampleRate=16000, channels=1):
        """Inicializa el controlador de audio y establece los arreglos base"""
        #Utilizamos 16000 Hz debido a que Whisper funciona mejor
        self.sampleRate = sampleRate
        self.channels = channels
        self.q = queue.Queue()
        self.stream = None
        
        #Arreglos para el audio original y el censurado
        self.audioData = np.array([], dtype='float32')
        self.audioCensurado = np.array([], dtype='float32')

    def callbackGrabacion(self, inData, frames, time, status):
        """Almacena en la cola los bloques de audio recibidos del microfono"""
        if status:
            print(status)
        self.q.put(inData.copy())

    def iniciarGrabacion(self):
        """Inicia el flujo de entrada desde el microfono"""
        self.audioData = np.array([], dtype='float32') #Reiniciar datos
        self.audioCensurado = np.array([], dtype='float32')
        self.q = queue.Queue()
        self.stream = sd.InputStream(samplerate=self.sampleRate, 
                                     channels=self.channels,
                                     callback=self.callbackGrabacion)
        self.stream.start()
        print("--> Grabación iniciada...")

    def detenerGrabacion(self):
        """Detiene el microfono y concatena los bloques capturados"""
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
                
                #Precalcular representaciones costosas como el espectrograma
                self.precalcularGraficas()
            
            print("--> Grabación detenida.")

    def precalcularGraficas(self):
        """Calcula el stft original de una sola vez para no recalcularlo"""
        if self.audioData is None or len(self.audioData) == 0:
            return
            
        self.n_fft = 2048
        self.hop_length = 512
        
        #Calculamos el Stft original
        self.stftOriginal = librosa.stft(self.audioData.flatten(), n_fft=self.n_fft, hop_length=self.hop_length)
        #La version censurada iniciara como una copia del original
        self.stftCensurado = self.stftOriginal.copy()

    def normalizarAudio(self):
        """Normaliza la senal de audio dividiendola por su amplitud maxima"""
        if self.audioData is not None and len(self.audioData) > 0:
            amplitudMaxima = np.max(np.abs(self.audioData))
            if amplitudMaxima > 0:
                self.audioData = self.audioData / amplitudMaxima
                
                #Al normalizar la onda tambien debemos actualizar los calculos base
                self.audioCensurado = self.audioData.copy()
                self.precalcularGraficas()
            print("--> Audio normalizado.")

    def aplicarCensuraAudio(self, palabrasASilenciar):
        """Silencia la onda y modifica el espectrograma precalculado segun tiempos"""
        self.audioCensurado = self.audioData.copy()
        self.stftCensurado = self.stftOriginal.copy()
        
        for p in palabrasASilenciar:
            #Modificar la forma de onda
            inicioMuestra = int(p["start"] * self.sampleRate)
            finMuestra = int(p["end"] * self.sampleRate)
            self.audioCensurado[inicioMuestra:finMuestra] = 0
            
            #Modificar el espectrograma precalculado Stft
            #Cada columna del Stft representa la cantidad de muestras del salto
            inicioCol = int(inicioMuestra / self.hop_length)
            finCol = int(finMuestra / self.hop_length)
            
            #Volver las frecuencias de ese bloque casi cero para evitar errores
            self.stftCensurado[:, inicioCol:finCol] = 1e-10
            
        print(f"--> Audio censurado: se silenciaron {len(palabrasASilenciar)} palabra(s).")

    def reproducirAudioOriginal(self):
        """Reproduce la senal original de audio"""
        if self.audioData is not None and len(self.audioData) > 0:
            print("--> Reproduciendo audio original...")
            sd.play(self.audioData, self.sampleRate)
        else:
            print("--> No hay audio original para reproducir.")

    def reproducirAudioCensurado(self):
        """Reproduce la senal de audio censurada silenciada"""
        if self.audioCensurado is not None and len(self.audioCensurado) > 0:
            print("--> Reproduciendo audio censurado...")
            sd.play(self.audioCensurado, self.sampleRate)
        else:
            print("--> No hay audio censurado para reproducir.")

    def graficarAudio(self, ax, tipo="onda", censurado=False):
        """Usa librosa para graficar usando los datos precalculados en los ejes provistos"""
        if self.audioData is None or len(self.audioData) == 0:
            return
            
        if tipo == "onda":
            audio = self.audioCensurado if censurado else self.audioData
            librosa.display.waveshow(audio.flatten(), sr=self.sampleRate, ax=ax, color='red' if censurado else 'blue')
            ax.set_ylabel("Amplitud")
            
        elif tipo == "espectrograma":
            stft = self.stftCensurado if censurado else self.stftOriginal
            magnitud = np.abs(stft)
            espectrogramaDb = librosa.amplitude_to_db(magnitud, ref=np.max)
            
            librosa.display.specshow(espectrogramaDb, 
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
