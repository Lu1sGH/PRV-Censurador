import sounddevice as sd #Captura de audio del microfono
import soundfile as sf #Lectura y escritura de archivos de audio
import numpy as np #Manejo de arreglos matematicos
import queue #Estructura de datos para audio
import librosa #Libreria de procesamiento de audio
import librosa.display #Visualizacion de audio
import cv2 #Captura de video
import threading #Hilos para grabar video y audio simultaneamente
import time #Control de tiempo en video
import torch #Procesamiento tensorial en tarjeta grafica
import torchaudio #Remuestreo de senales de audio en tensores


class AudioCtrl:
    def __init__(self, sampleRate=16000, channels=1):
        """Inicializa el controlador de audio y video y establece los arreglos base"""
        #Utilizamos 16000 Hz debido a que Whisper trabaja directamente a esta frecuencia
        self.sampleRate = sampleRate
        self.channels = channels
        self.q = queue.Queue()
        self.stream = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        #Arreglos para el audio original y el censurado
        self.audioData = np.array([], dtype='float32')
        self.audioCensurado = np.array([], dtype='float32')

        #Parametros de transformada de corto tiempo
        self.n_fft = 2048
        self.hop_length = 512
        self.stftOriginal = None
        self.stftCensurado = None

        #Variables para video
        self.videoFrames = []
        self.videoCaptura = None
        self.grabandoVideo = False
        self.threadVideo = None
        self.fpsVideo = 30.0

    def callbackGrabacion(self, inData, frames, timeInfo, status):
        """Almacena en la cola los bloques de audio recibidos del microfono"""
        if status:
            print(status)
        self.q.put(inData.copy())

    def iniciarGrabacion(self):
        """Inicia la captura simultanea desde el microfono y la camara web"""
        self.audioData = np.array([], dtype='float32') #Reiniciar datos
        self.audioCensurado = np.array([], dtype='float32')
        self.stftOriginal = None
        self.stftCensurado = None
        self.q = queue.Queue()

        #Iniciar video
        self.videoFrames = []
        self.videoCaptura = cv2.VideoCapture(0)
        fps = self.videoCaptura.get(cv2.CAP_PROP_FPS)
        self.fpsVideo = fps if fps > 0 else 30.0
        self.grabandoVideo = True
        self.threadVideo = threading.Thread(target=self.loopVideo)
        self.threadVideo.start()

        #Iniciar audio
        self.stream = sd.InputStream(
            samplerate=self.sampleRate,
            channels=self.channels,
            callback=self.callbackGrabacion
        )
        self.stream.start()
        self.tiempoInicioGrabacion = time.perf_counter()
        print("--> Grabación de audio y video iniciada...")

    def loopVideo(self):
        """Ejecuta un bucle en un hilo separado para capturar cuadros de video"""
        while self.grabandoVideo and self.videoCaptura.isOpened():
            ret, frame = self.videoCaptura.read()
            if ret:
                #Convertimos el cuadro de color a formato compatible con la interfaz
                frameRgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.videoFrames.append(frameRgb)
            #Pequena pausa para no saturar el hilo
            time.sleep(0.001)

    def detenerGrabacion(self):
        """Detiene el microfono y la camara y une los bloques crudos capturados"""
        self.tiempoFinGrabacion = time.perf_counter()

        #Detener video
        self.grabandoVideo = False
        if self.threadVideo is not None:
            self.threadVideo.join()
        if self.videoCaptura is not None:
            self.videoCaptura.release()
            self.videoCaptura = None

        #Calcular los cuadros por segundo reales en base al tiempo transcurrido
        duracionReal = self.tiempoFinGrabacion - self.tiempoInicioGrabacion
        if duracionReal > 0 and len(self.videoFrames) > 0:
            self.fpsVideo = len(self.videoFrames) / duracionReal

        #Detener audio
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

            bloques = []
            while not self.q.empty():
                bloques.append(self.q.get())

            if bloques:
                self.audioData = np.concatenate(bloques, axis=0).astype(np.float32)
                self.audioCensurado = self.audioData.copy()

            print(f"--> Grabación detenida. Total de cuadros de video: {len(self.videoFrames)}")

    def precalcularGraficas(self):
        """Calcula el espectrograma original de una sola vez para no recalcularlo"""
        if self.audioData is None or len(self.audioData) == 0:
            return

        self.n_fft = 2048
        self.hop_length = 512

        #Calculamos el Stft original
        self.stftOriginal = librosa.stft(
            self.audioData.flatten(),
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        #La version censurada iniciara como una copia del original
        self.stftCensurado = self.stftOriginal.copy()

    def preprocesarAudio(self):
        """Acondiciona el audio remuestreando a 16kHz si se requiere, normalizando amplitud y calculando su espectrograma"""
        if self.audioData is None or len(self.audioData) == 0:
            return

        #Asegurar frecuencia de muestreo de 16000 Hz para Whisper
        if self.sampleRate != 16000:
            onda1d = self.audioData.flatten().astype(np.float32)
            tensorAudio = torch.from_numpy(onda1d).unsqueeze(0).to(self.device)
            remuestreador = torchaudio.transforms.Resample(
                orig_freq=self.sampleRate,
                new_freq=16000
            ).to(self.device)
            audio16k = remuestreador(tensorAudio).squeeze(0).cpu().numpy()
            self.audioData = audio16k.reshape(-1, 1).astype(np.float32)
            self.sampleRate = 16000

        #Normalizar amplitud y precalcular espectrograma base
        amplitudMaxima = np.max(np.abs(self.audioData))
        if amplitudMaxima > 0:
            self.audioData = self.audioData / amplitudMaxima

        self.audioCensurado = self.audioData.copy()
        self.precalcularGraficas()
        print("--> Audio preprocesado y normalizado.")

    def normalizarAudio(self):
        """Alias hacia preprocesarAudio para mantener compatibilidad"""
        self.preprocesarAudio()

    def calcularModuladorAcustico(self, palabrasConTiempo, betaRms=0.4, betaSpec=0.6):
        """Calcula el modulador prosodico gamma a partir del audio preprocesado y su espectrograma"""
        if self.audioData is None or len(self.audioData) == 0 or not palabrasConTiempo:
            return palabrasConTiempo

        if self.stftOriginal is None:
            self.precalcularGraficas()

        onda1d = torch.from_numpy(self.audioData.flatten()).to(self.device, dtype=torch.float32)
        magStft = torch.from_numpy(np.abs(self.stftOriginal)).to(self.device, dtype=torch.float32)

        totalMuestras = onda1d.shape[0]
        totalColumnas = magStft.shape[1]
        eps = 1e-6

        rmsGlobal = torch.sqrt(torch.mean(onda1d ** 2)) + eps

        #Rango de alta frecuencia entre 1500 Hz y 4000 Hz
        binCorteInf = int(1500.0 * self.n_fft / self.sampleRate)
        binCorteSup = min(magStft.shape[0], int(4000.0 * self.n_fft / self.sampleRate))

        energiaAgudosGlobal = torch.sum(magStft[binCorteInf:binCorteSup, :])
        energiaTotalGlobal = torch.sum(magStft[:binCorteSup, :]) + eps
        centroideGlobal = float((energiaAgudosGlobal / energiaTotalGlobal).item())

        for p in palabrasConTiempo:
            tIni = max(0.0, float(p["start"]))
            tFin = max(tIni + 0.02, float(p["end"]))

            idxOndaIni = min(totalMuestras - 1, int(np.floor(tIni * self.sampleRate)))
            idxOndaFin = min(totalMuestras, max(idxOndaIni + 1, int(np.ceil(tFin * self.sampleRate))))

            idxSpecIni = min(totalColumnas - 1, int(np.floor(idxOndaIni / self.hop_length)))
            idxSpecFin = min(totalColumnas, max(idxSpecIni + 1, int(np.ceil(idxOndaFin / self.hop_length))))

            subOnda = onda1d[idxOndaIni:idxOndaFin]
            rmsPalabra = float((torch.sqrt(torch.mean(subOnda ** 2)) / rmsGlobal).item())

            subSpec = magStft[:, idxSpecIni:idxSpecFin]
            eAgudosW = torch.sum(subSpec[binCorteInf:binCorteSup, :])
            eTotalW = torch.sum(subSpec[:binCorteSup, :]) + eps
            cSpecW = float((eAgudosW / eTotalW).item())

            deltaEspectral = cSpecW - centroideGlobal

            gammaI = (
                1.0
                + betaRms * max(0.0, rmsPalabra - 1.0)
                + betaSpec * max(0.0, deltaEspectral)
            )

            p["rms_norm"] = round(rmsPalabra, 4)
            p["delta_spec"] = round(deltaEspectral, 4)
            p["gamma"] = round(float(gammaI), 4)

        return palabrasConTiempo

    def aplicarCensuraAudio(self, palabrasASilenciar, tipoCensura="silencio"):
        """Genera el audio censurado aplicando silencio, pitidos o ruido en los intervalos detectados"""
        if self.audioData is None or len(self.audioData) == 0:
            return

        self.audioCensurado = self.audioData.copy()
        if self.stftOriginal is None:
            self.precalcularGraficas()
        self.stftCensurado = self.stftOriginal.copy()

        totalMuestras = len(self.audioCensurado)
        totalCols = self.stftCensurado.shape[1]
        rampaMuestras = int(0.010 * self.sampleRate) #10 ms de rampa suave

        for p in palabrasASilenciar:
            inicioMuestra = max(0, int(p["start"] * self.sampleRate))
            finMuestra = min(totalMuestras, int(p["end"] * self.sampleRate))
            longitud = finMuestra - inicioMuestra

            if longitud <= 0:
                continue

            rEff = min(rampaMuestras, longitud // 2)

            if tipoCensura == "silencio":
                envolvente = np.zeros((longitud, 1), dtype=np.float32)
                if rEff > 0:
                    m = np.arange(rEff, dtype=np.float32)
                    envolvente[:rEff, 0] = 0.5 * (1.0 + np.cos(np.pi * m / rEff))
                    envolvente[-rEff:, 0] = 0.5 * (1.0 - np.cos(np.pi * m / rEff))

                if self.audioCensurado.ndim == 1:
                    self.audioCensurado[inicioMuestra:finMuestra] *= envolvente.flatten()
                else:
                    self.audioCensurado[inicioMuestra:finMuestra] *= envolvente

                inicioCol = max(0, int(inicioMuestra / self.hop_length))
                finCol = min(totalCols, int(np.ceil(finMuestra / self.hop_length)))
                self.stftCensurado[:, inicioCol:finCol] = 1e-10
            else:
                if tipoCensura == "agudo":
                    t = np.arange(longitud, dtype=np.float32) / self.sampleRate
                    onda = 0.5 * np.sin(2 * np.pi * 1000.0 * t)
                    segmento = onda.reshape(-1, self.channels).astype(np.float32)
                elif tipoCensura == "grave":
                    t = np.arange(longitud, dtype=np.float32) / self.sampleRate
                    onda = 0.5 * np.sin(2 * np.pi * 250.0 * t)
                    segmento = onda.reshape(-1, self.channels).astype(np.float32)
                elif tipoCensura == "ruido":
                    onda = np.random.uniform(-0.4, 0.4, size=(longitud, self.channels))
                    segmento = onda.astype(np.float32)
                else:
                    segmento = np.zeros((longitud, self.channels), dtype=np.float32)

                #Aplicar rampa suave de entrada y salida al sonido sintetizado
                if rEff > 0:
                    m = np.arange(rEff, dtype=np.float32)
                    ventana = np.ones((longitud, 1), dtype=np.float32)
                    ventana[:rEff, 0] = 0.5 * (1.0 - np.cos(np.pi * m / rEff))
                    ventana[-rEff:, 0] = 0.5 * (1.0 + np.cos(np.pi * m / rEff))
                    segmento *= ventana

                if self.audioCensurado.ndim == 1:
                    self.audioCensurado[inicioMuestra:finMuestra] = segmento.flatten()
                else:
                    self.audioCensurado[inicioMuestra:finMuestra] = segmento

        #Si se inyecto pitido o ruido recalculamos el espectrograma censurado
        if tipoCensura != "silencio" and len(palabrasASilenciar) > 0:
            self.stftCensurado = librosa.stft(
                self.audioCensurado.flatten(),
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )

        print(f"--> Audio censurado ({tipoCensura}): se procesaron {len(palabrasASilenciar)} palabra(s).")

    def reproducirAudioOriginal(self):
        """Reproduce la senal original de audio"""
        if self.audioData is not None and len(self.audioData) > 0:
            print("--> Reproduciendo audio original...")
            sd.play(self.audioData, self.sampleRate)
        else:
            print("--> No hay audio original para reproducir.")

    def reproducirAudioCensurado(self):
        """Reproduce la senal de audio censurada"""
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
            if stft is None:
                return
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
