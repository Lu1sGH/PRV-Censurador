import sounddevice as sd  # Captura de audio del microfono
import soundfile as sf  # Lectura y escritura de archivos de audio
import numpy as np  # Manejo de arreglos matematicos
import queue  # Estructura de datos para audio
import librosa  # Libreria de procesamiento de audio
import librosa.display  # Visualizacion de audio
import cv2  # Captura de video
import threading  # Hilos para grabar video y audio simultaneamente
import time  # Control de tiempo en video
import torch  # Procesamiento tensorial en GPU


class AudioCtrl:
    def __init__(self, sampleRate=16000, channels=1):
        """Inicializa el controlador de audio y establece los arreglos base"""
        # Utilizamos 16000 Hz debido a que Whisper funciona mejor
        self.sampleRate = sampleRate
        self.channels = channels
        self.q = queue.Queue()
        self.stream = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Arreglos para el audio original y el censurado
        self.audioData = np.array([], dtype='float32')
        self.audioCensurado = np.array([], dtype='float32')

        # Parametros de STFT
        self.n_fft = 2048
        self.hop_length = 512
        self.stftOriginal = None
        self.stftCensurado = None

        # Variables para video
        self.videoFrames = []
        self.videoCaptura = None
        self.grabandoVideo = False
        self.threadVideo = None
        self.fpsVideo = 30.0

    def callbackGrabacion(self, inData, frames, time_info, status):
        """Almacena en la cola los bloques de audio recibidos del microfono"""
        if status:
            print(status)
        self.q.put(inData.copy())

    def iniciarGrabacion(self):
        """Inicia el flujo de entrada desde el microfono y la camara web"""
        self.audioData = np.array([], dtype='float32')  # Reiniciar datos
        self.audioCensurado = np.array([], dtype='float32')
        self.q = queue.Queue()

        # Iniciar video
        self.videoFrames = []
        self.videoCaptura = cv2.VideoCapture(0)
        fps = self.videoCaptura.get(cv2.CAP_PROP_FPS)
        self.fpsVideo = fps if fps > 0 else 30.0
        self.grabandoVideo = True
        self.threadVideo = threading.Thread(target=self.loopVideo)
        self.threadVideo.start()

        # Iniciar audio
        self.stream = sd.InputStream(
            samplerate=self.sampleRate,
            channels=self.channels,
            callback=self.callbackGrabacion
        )
        self.stream.start()
        self.tiempoInicioGrabacion = time.perf_counter()
        print("--> Grabación de audio y video iniciada...")

    def loopVideo(self):
        """Bucle en un hilo separado para capturar cuadros de video"""
        while self.grabandoVideo and self.videoCaptura.isOpened():
            ret, frame = self.videoCaptura.read()
            if ret:
                # Convertimos el frame de BGR a RGB para la interfaz Qt
                frameRgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.videoFrames.append(frameRgb)
            # Pequeña pausa para no saturar el hilo
            time.sleep(0.001)

    def detenerGrabacion(self):
        """Detiene el microfono, la camara y concatena los bloques capturados"""
        self.tiempoFinGrabacion = time.perf_counter()

        # Detener video
        self.grabandoVideo = False
        if self.threadVideo is not None:
            self.threadVideo.join()
        if self.videoCaptura is not None:
            self.videoCaptura.release()
            self.videoCaptura = None

        # Calcular los FPS reales en base al tiempo transcurrido
        duracionReal = self.tiempoFinGrabacion - self.tiempoInicioGrabacion
        if duracionReal > 0 and len(self.videoFrames) > 0:
            self.fpsVideo = len(self.videoFrames) / duracionReal

        # Detener audio
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

                # Precalcular representaciones costosas como el espectrograma
                self.precalcularGraficas()

            print(f"--> Grabación detenida. Total de frames de video: {len(self.videoFrames)}")

    def precalcularGraficas(self):
        """Calcula el stft original de una sola vez para no recalcularlo"""
        if self.audioData is None or len(self.audioData) == 0:
            return

        self.n_fft = 2048
        self.hop_length = 512

        # Calculamos el Stft original
        self.stftOriginal = librosa.stft(
            self.audioData.flatten(),
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        # La version censurada iniciara como una copia del original
        self.stftCensurado = self.stftOriginal.copy()

    def normalizarAudio(self):
        """Normaliza la senal de audio dividiendola por su amplitud maxima"""
        if self.audioData is not None and len(self.audioData) > 0:
            amplitudMaxima = np.max(np.abs(self.audioData))
            if amplitudMaxima > 0:
                self.audioData = self.audioData / amplitudMaxima

                # Al normalizar la onda tambien debemos actualizar los calculos base
                self.audioCensurado = self.audioData.copy()
                self.precalcularGraficas()
            print("--> Audio normalizado.")

    def calcularModuladorAcustico(self, palabrasConTiempo, beta_rms=0.4, beta_spec=0.6):
        """
        Calcula en GPU el modulador prosodico gamma(w_i) a partir de la forma de onda
        normalizada y los bins de alta frecuencia [1.5 kHz, 4.0 kHz] de self.stftOriginal.
        """
        if self.audioData is None or len(self.audioData) == 0 or not palabrasConTiempo:
            return palabrasConTiempo

        if self.stftOriginal is None:
            self.precalcularGraficas()

        onda_1d = torch.from_numpy(self.audioData.flatten()).to(self.device, dtype=torch.float32)
        mag_stft = torch.from_numpy(np.abs(self.stftOriginal)).to(self.device, dtype=torch.float32)

        total_muestras = onda_1d.shape[0]
        total_columnas = mag_stft.shape[1]
        eps = 1e-6

        rms_global = torch.sqrt(torch.mean(onda_1d ** 2)) + eps

        # Resolucion df = 16000 / 2048 = 7.8125 Hz/bin -> [1500 Hz, 4000 Hz] = bins [192, 512]
        bin_corte_inf = int(1500.0 * self.n_fft / self.sampleRate)
        bin_corte_sup = min(mag_stft.shape[0], int(4000.0 * self.n_fft / self.sampleRate))

        energia_agudos_global = torch.sum(mag_stft[bin_corte_inf:bin_corte_sup, :])
        energia_total_global = torch.sum(mag_stft[:bin_corte_sup, :]) + eps
        centroide_global = float((energia_agudos_global / energia_total_global).item())

        for p in palabrasConTiempo:
            t_ini = max(0.0, float(p["start"]))
            t_fin = max(t_ini + 0.02, float(p["end"]))

            idx_onda_ini = min(total_muestras - 1, int(np.floor(t_ini * self.sampleRate)))
            idx_onda_fin = min(total_muestras, max(idx_onda_ini + 1, int(np.ceil(t_fin * self.sampleRate))))

            idx_spec_ini = min(total_columnas - 1, int(np.floor(idx_onda_ini / self.hop_length)))
            idx_spec_fin = min(total_columnas, max(idx_spec_ini + 1, int(np.ceil(idx_onda_fin / self.hop_length))))

            sub_onda = onda_1d[idx_onda_ini:idx_onda_fin]
            rms_palabra = float((torch.sqrt(torch.mean(sub_onda ** 2)) / rms_global).item())

            sub_spec = mag_stft[:, idx_spec_ini:idx_spec_fin]
            e_agudos_w = torch.sum(sub_spec[bin_corte_inf:bin_corte_sup, :])
            e_total_w = torch.sum(sub_spec[:bin_corte_sup, :]) + eps
            c_spec_w = float((e_agudos_w / e_total_w).item())

            delta_espectral = c_spec_w - centroide_global

            gamma_i = (
                1.0
                + beta_rms * max(0.0, rms_palabra - 1.0)
                + beta_spec * max(0.0, delta_espectral)
            )

            p["rms_norm"] = round(rms_palabra, 4)
            p["delta_spec"] = round(delta_espectral, 4)
            p["gamma"] = round(float(gamma_i), 4)

        return palabrasConTiempo

    def aplicarCensuraAudio(self, palabrasASilenciar):
        """Silencia la onda con envolvente inversa de Hann y modifica el espectrograma precalculado segun tiempos"""
        self.audioCensurado = self.audioData.copy()
        self.stftCensurado = self.stftOriginal.copy()

        totalMuestras = len(self.audioCensurado)
        totalCols = self.stftCensurado.shape[1]
        rampaMuestras = int(0.010 * self.sampleRate)  # 10 ms de rampa suave

        for p in palabrasASilenciar:
            # Modificar la forma de onda
            inicioMuestra = max(0, int(p["start"] * self.sampleRate))
            finMuestra = min(totalMuestras, int(p["end"] * self.sampleRate))
            longitud = finMuestra - inicioMuestra

            if longitud > 0:
                envolvente = np.zeros((longitud, 1), dtype=np.float32)
                r_eff = min(rampaMuestras, longitud // 2)
                if r_eff > 0:
                    m = np.arange(r_eff, dtype=np.float32)
                    envolvente[:r_eff, 0] = 0.5 * (1.0 + np.cos(np.pi * m / r_eff))
                    envolvente[-r_eff:, 0] = 0.5 * (1.0 - np.cos(np.pi * m / r_eff))

                if self.audioCensurado.ndim == 1:
                    self.audioCensurado[inicioMuestra:finMuestra] *= envolvente.flatten()
                else:
                    self.audioCensurado[inicioMuestra:finMuestra] *= envolvente

            # Modificar el espectrograma precalculado Stft
            # Cada columna del Stft representa la cantidad de muestras del salto
            inicioCol = max(0, int(inicioMuestra / self.hop_length))
            finCol = min(totalCols, int(np.ceil(finMuestra / self.hop_length)))

            # Volver las frecuencias de ese bloque casi cero para evitar errores    
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
            librosa.display.waveshow(
                audio.flatten(),
                sr=self.sampleRate,
                ax=ax,
                color='red' if censurado else 'blue'
            )
            ax.set_ylabel("Amplitud")

        elif tipo == "espectrograma":
            stft = self.stftCensurado if censurado else self.stftOriginal
            magnitud = np.abs(stft)
            espectrogramaDb = librosa.amplitude_to_db(magnitud, ref=np.max)

            librosa.display.specshow(
                espectrogramaDb,
                sr=self.sampleRate,
                hop_length=self.hop_length,
                x_axis="time",
                y_axis="log",
                ax=ax,
                cmap='inferno' if censurado else 'viridis'
            )
            ax.set_ylabel("Frec. (Hz)")

        titulo = "Censurado" if censurado else "Original"
        ax.set_title(f"Forma de onda {titulo}" if tipo == "onda" else f"Espectrograma {titulo}")
        ax.set_xlabel("Tiempo (s)")