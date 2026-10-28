import sys #Manejo del sistema
import time #Medicion de tiempos
import numpy as np #Manejo de arreglos matematicos
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QComboBox, QGroupBox, QSpacerItem, QSizePolicy, QGridLayout, QStackedWidget #Elementos graficos principales
from PyQt5.QtGui import QIcon, QFont, QImage, QPixmap #Iconos, fuentes e imagenes
from PyQt5.QtCore import Qt, QSize, QTimer #Constantes y temporizadores Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas #Lienzo para graficas
from matplotlib.figure import Figure #Figura grafica
from AudioCtrl import AudioCtrl #Controlador de audio y video
from Censurador import Censurador #Procesador de texto, discurso de odio y palabras altisonantes
from Whisper import WhisperTranscriptor #Transcriptor de voz

#Podríamos usar esto para censurar cualquier tipo de discurso que no nos convenga, no solo de odio o palabras altisonantes :/

class InterfazAudio(QMainWindow):
    def __init__(self):
        """Inicializa la interfaz principal y sus componentes visuales y logicos"""
        super().__init__()
        self.audioCtrl = AudioCtrl()
        self.censurador = Censurador('MX')
        self.whisperTranscriptor = WhisperTranscriptor('small')

        self.setWindowTitle("Censurador")
        self.setGeometry(100, 100, 750, 750)
        self.setStyleSheet("QMainWindow { background-color: #f5f5f5; }")

        widgetCentral = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        self.setCentralWidget(widgetCentral)

        #Focos indicadores rojo azul verde
        headerLayout = QHBoxLayout()
        headerLayout.setAlignment(Qt.AlignLeft)

        focosGrid = QGridLayout()
        focosGrid.setSpacing(10)

        #Foco rojo
        self.focoRojo = QLabel()
        self.focoRojo.setFixedSize(16, 16)
        labelRojo = QLabel("Grabando")
        labelRojo.setFont(QFont("Arial", 9))
        labelRojo.setAlignment(Qt.AlignCenter)
        focosGrid.addWidget(self.focoRojo, 0, 0, alignment=Qt.AlignCenter)
        focosGrid.addWidget(labelRojo, 1, 0, alignment=Qt.AlignTop | Qt.AlignHCenter)

        #Foco azul
        self.focoAzul = QLabel()
        self.focoAzul.setFixedSize(16, 16)
        labelAzul = QLabel("Procesando\ny censurando")
        labelAzul.setAlignment(Qt.AlignCenter)
        labelAzul.setFont(QFont("Arial", 9))
        focosGrid.addWidget(self.focoAzul, 0, 1, alignment=Qt.AlignCenter)
        focosGrid.addWidget(labelAzul, 1, 1, alignment=Qt.AlignTop | Qt.AlignHCenter)

        #Foco verde
        self.focoVerde = QLabel()
        self.focoVerde.setFixedSize(16, 16)
        labelVerde = QLabel("Listo")
        labelVerde.setAlignment(Qt.AlignCenter)
        labelVerde.setFont(QFont("Arial", 9))
        focosGrid.addWidget(self.focoVerde, 0, 2, alignment=Qt.AlignCenter)
        focosGrid.addWidget(labelVerde, 1, 2, alignment=Qt.AlignTop | Qt.AlignHCenter)

        headerLayout.addLayout(focosGrid)
        headerLayout.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        self.comboGrafico = QComboBox(self)
        self.comboGrafico.addItem("Forma de Onda")
        self.comboGrafico.addItem("Espectrograma")
        self.comboGrafico.addItem("Video")
        self.comboGrafico.setFont(QFont("Arial", 10))
        self.comboGrafico.currentTextChanged.connect(self.cambiarModoGrafico)
        headerLayout.addWidget(self.comboGrafico)

        self.comboCensura = QComboBox(self)
        self.comboCensura.addItems([
            "Censura: Silencio",
            "Censura: Pitido agudo",
            "Censura: Pitido grave",
            "Censura: Ruido"
        ])
        self.comboCensura.setFont(QFont("Arial", 10))
        self.comboCensura.currentTextChanged.connect(self.cambiarModoCensura)
        headerLayout.addWidget(self.comboCensura)

        self.comboModelo = QComboBox(self)
        self.comboModelo.addItems(["Modelo Whisper: small", "Modelo Whisper: base", "Modelo Whisper: tiny"])
        self.comboModelo.setFont(QFont("Arial", 10))
        self.comboModelo.currentTextChanged.connect(self.cambiarModeloWhisper)
        headerLayout.addWidget(self.comboModelo)

        layout.addLayout(headerLayout)

        #Panel intercambiable entre graficos y video
        self.stackedWidget = QStackedWidget()

        #Pagina 0 graficos Matplotlib
        self.figura = Figure(figsize=(6, 4))
        self.canvas = FigureCanvas(self.figura)
        self.axOriginal = self.figura.add_subplot(211)
        self.axOriginal.set_title("Original")
        self.axCensurado = self.figura.add_subplot(212)
        self.axCensurado.set_title("Censurado")
        self.figura.tight_layout()
        self.stackedWidget.addWidget(self.canvas)

        #Pagina 1 video
        self.labelVideo = QLabel("Captura de Video")
        self.labelVideo.setAlignment(Qt.AlignCenter)
        self.labelVideo.setStyleSheet("QLabel { background-color: #000; color: #fff; font-size: 16px; }")
        self.stackedWidget.addWidget(self.labelVideo)

        layout.addWidget(self.stackedWidget)

        #Configuracion del temporizador de reproduccion de video
        self.timerVideo = QTimer()
        self.timerVideo.timeout.connect(self.actualizarFrameVideo)
        self.frameActual = 0

        self.modoGrafico = "onda"
        self.tipoCensura = "silencio"
        self.ultimasPalabrasASilenciar = []

        #Estilo analogico metalico para botones
        estiloBotonAnalogico = """
            QPushButton {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #f0f0f0, stop: 1 #a0a0a0);
                border: 2px solid #555;
                border-radius: 4px;
                border-bottom: 4px solid #333;
                color: #222;
                font-weight: bold;
                font-size: 13px;
                padding: 10px;
            }
            QPushButton:pressed {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #a0a0a0, stop: 1 #f0f0f0);
                border-bottom: 2px solid #333;
                margin-top: 2px;
            }
        """

        #Botones de accion
        botonesLayout = QHBoxLayout()

        self.btnGrabar = QPushButton(" GRABAR", self)
        self.btnGrabar.setIcon(QIcon('iconos/mic.svg'))
        self.btnGrabar.setIconSize(QSize(20, 20))
        self.btnGrabar.setStyleSheet(estiloBotonAnalogico)
        self.btnGrabar.clicked.connect(self.toggleGrabacion)
        botonesLayout.addWidget(self.btnGrabar)

        self.btnReproducirOrig = QPushButton(" REPRODUCIR (ORIGINAL)", self)
        self.btnReproducirOrig.setIcon(QIcon('iconos/play.svg'))
        self.btnReproducirOrig.setIconSize(QSize(20, 20))
        self.btnReproducirOrig.setStyleSheet(estiloBotonAnalogico)
        self.btnReproducirOrig.clicked.connect(self.reproducirOriginal)
        botonesLayout.addWidget(self.btnReproducirOrig)

        self.btnReproducirCens = QPushButton(" REPRODUCIR (CENSURADO)", self)
        self.btnReproducirCens.setIcon(QIcon('iconos/play_censor.svg'))
        self.btnReproducirCens.setIconSize(QSize(20, 20))
        self.btnReproducirCens.setStyleSheet(estiloBotonAnalogico)
        self.btnReproducirCens.clicked.connect(self.reproducirCensurado)
        botonesLayout.addWidget(self.btnReproducirCens)

        layout.addLayout(botonesLayout)

        #Recuadros de texto
        textosLayout = QHBoxLayout()

        grupoOriginal = QGroupBox("Mensaje original")
        grupoOriginal.setFont(QFont("Arial", 10))
        layoutOrig = QVBoxLayout()
        self.labelTranscripcion = QLabel("...")
        self.labelTranscripcion.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.labelTranscripcion.setWordWrap(True)
        self.labelTranscripcion.setFont(QFont("Arial", 11))
        layoutOrig.addWidget(self.labelTranscripcion)
        grupoOriginal.setLayout(layoutOrig)
        textosLayout.addWidget(grupoOriginal)

        grupoCensurado = QGroupBox("Mensaje censurado")
        grupoCensurado.setFont(QFont("Arial", 10))
        layoutCens = QVBoxLayout()
        self.labelCensurado = QLabel("...")
        self.labelCensurado.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.labelCensurado.setWordWrap(True)
        self.labelCensurado.setFont(QFont("Arial", 11))
        layoutCens.addWidget(self.labelCensurado)
        grupoCensurado.setLayout(layoutCens)
        textosLayout.addWidget(grupoCensurado)

        layout.addLayout(textosLayout)

        widgetCentral.setLayout(layout)
        self.grabando = False

        #Inicializar todos apagados menos el verde
        self.actualizarFocos("verde")

    def actualizarFocos(self, focoActivo):
        """Apaga todos los focos y enciende solo el solicitado"""
        estiloApagado = "QLabel { background-color: #bbbbbb; border-radius: 8px; }"

        #Apagar todos primero
        self.focoRojo.setStyleSheet(estiloApagado)
        self.focoAzul.setStyleSheet(estiloApagado)
        self.focoVerde.setStyleSheet(estiloApagado)

        #Encender el correspondiente
        if focoActivo == "rojo":
            self.focoRojo.setStyleSheet("QLabel { background-color: #ff3333; border-radius: 8px; }")
        elif focoActivo == "azul":
            self.focoAzul.setStyleSheet("QLabel { background-color: #3385ff; border-radius: 8px; }")
        elif focoActivo == "verde":
            self.focoVerde.setStyleSheet("QLabel { background-color: #00cc66; border-radius: 8px; }")

    def toggleGrabacion(self):
        """Ejecuta en orden la arquitectura de grabacion, preprocesamiento, Whisper, censurador y audio censurado"""
        if not self.grabando:
            #1. Grabacion del audio y video (inicio)
            self.grabando = True
            self.btnGrabar.setText(" DETENER")
            self.actualizarFocos("rojo")

            self.labelTranscripcion.setText("...")
            self.labelCensurado.setText("...")
            self.ultimasPalabrasASilenciar = []

            #Limpiar graficos y video
            self.axOriginal.clear()
            self.axOriginal.set_title("Original")
            self.axCensurado.clear()
            self.axCensurado.set_title("Censurado")
            self.canvas.draw()
            self.labelVideo.clear()
            self.labelVideo.setText("Capturando video...")
            self.timerVideo.stop()

            self.tInicioGrabacion = time.perf_counter()
            print(f"\n[{time.strftime('%H:%M:%S')}] --- INICIO ---")
            self.audioCtrl.iniciarGrabacion()
        else:
            #1. Grabacion del audio y video (fin)
            self.grabando = False
            self.btnGrabar.setText(" GRABAR")
            self.actualizarFocos("azul")
            QApplication.processEvents()

            tFinGrabacion = time.perf_counter()
            print(f"[{time.strftime('%H:%M:%S')}] --- Grabación detenida. Duración de la captura: {tFinGrabacion - self.tInicioGrabacion:.3f} s ---")
            self.audioCtrl.detenerGrabacion()

            #2. Preprocesamiento del audio
            tInicioPre = time.perf_counter()
            self.audioCtrl.preprocesarAudio()
            tFinPre = time.perf_counter()
            print(f"[{time.strftime('%H:%M:%S')}] --- Tiempo de preprocesamiento del audio: {tFinPre - tInicioPre:.3f} s ---")

            #Primera graficacion del audio original preprocesado
            tInicioGraf = time.perf_counter()
            self.dibujarGraficos()
            QApplication.processEvents()
            tFinGraf = time.perf_counter()
            print(f"[{time.strftime('%H:%M:%S')}] --- Tiempo de primera graficación: {tFinGraf - tInicioGraf:.3f} s ---")

            #3. Whisper
            tInicioWhisper = time.perf_counter()
            textoOriginal, palabrasConTiempo = self.whisperTranscriptor.transcribir(
                self.audioCtrl.audioData,
                self.audioCtrl.sampleRate
            )
            tFinWhisper = time.perf_counter()
            print(f"[{time.strftime('%H:%M:%S')}] --- Tiempo de inferencia de Whisper: {tFinWhisper - tInicioWhisper:.3f} s ---")

            if textoOriginal:
                tInicioCensura = time.perf_counter()

                #Asociar modulador acustico del audio preprocesado a cada palabra transcrita
                palabrasConTiempo = self.audioCtrl.calcularModuladorAcustico(palabrasConTiempo)

                #4. Censurador (Normalizar texto -> Discurso de odio -> Palabras altisonantes)
                textoCensurado, palabrasASilenciar = self.censurador.procesarCensura(
                    textoOriginal,
                    palabrasConTiempo
                )

                self.labelTranscripcion.setText(textoOriginal)
                self.labelCensurado.setText(textoCensurado)

                #5. Audio censurado
                self.ultimasPalabrasASilenciar = palabrasASilenciar
                self.audioCtrl.aplicarCensuraAudio(self.ultimasPalabrasASilenciar, self.tipoCensura)

                tFinCensura = time.perf_counter()
                print(f"[{time.strftime('%H:%M:%S')}] --- Tiempo de censura y modificación del audio: {tFinCensura - tInicioCensura:.3f} s ---")

                #Segunda graficacion con el audio censurado
                tInicioGraf2 = time.perf_counter()
                self.dibujarGraficos()
                if self.modoGrafico == "video" and len(self.audioCtrl.videoFrames) > 0:
                    self.mostrarFrame(0)
                tFinGraf2 = time.perf_counter()
                print(f"[{time.strftime('%H:%M:%S')}] --- Tiempo de segunda graficación: {tFinGraf2 - tInicioGraf2:.3f} s ---")

                self.actualizarFocos("verde")
            else:
                self.labelTranscripcion.setText("(No se detectó voz)")
                self.labelCensurado.setText("-")
                if self.modoGrafico == "video" and len(self.audioCtrl.videoFrames) > 0:
                    self.mostrarFrame(0)
                self.actualizarFocos("verde")

            #6. Listo para reproduccion de audio y video
            tFinTotal = time.perf_counter()
            print(f"[{time.strftime('%H:%M:%S')}] --- FIN. Tiempo total de espera del usuario: {tFinTotal - tFinGrabacion:.3f} s ---\n")

    def cambiarModeloWhisper(self, textoSeleccionado):
        """Cambia el modelo de Whisper y lo recarga en memoria"""
        nombre = textoSeleccionado.replace("Modelo Whisper: ", "").strip()
        print(f"\n--> Cambiando modelo de Whisper a '{nombre}'...")

        #Indicador visual de procesamiento
        self.btnGrabar.setEnabled(False)
        self.actualizarFocos("rojo")
        QApplication.processEvents()

        #Cargamos el nuevo modelo
        self.whisperTranscriptor = WhisperTranscriptor(nombre)

        #Restauramos la interfaz
        self.btnGrabar.setEnabled(True)
        self.actualizarFocos("verde")

    def cambiarModoCensura(self, textoSeleccionado):
        """Cambia el efecto de censura aplicado y actualiza el audio si ya existe"""
        if "Silencio" in textoSeleccionado:
            self.tipoCensura = "silencio"
        elif "Pitido agudo" in textoSeleccionado:
            self.tipoCensura = "agudo"
        elif "Pitido grave" in textoSeleccionado:
            self.tipoCensura = "grave"
        elif "Ruido" in textoSeleccionado:
            self.tipoCensura = "ruido"

        if len(self.audioCtrl.audioData) > 0:
            self.audioCtrl.aplicarCensuraAudio(self.ultimasPalabrasASilenciar, self.tipoCensura)
            if self.modoGrafico != "video":
                self.dibujarGraficos()

    def cambiarModoGrafico(self, textoSeleccionado):
        """Cambia entre la vista de forma de onda, espectrograma o video"""
        if textoSeleccionado == "Forma de Onda":
            self.modoGrafico = "onda"
            self.stackedWidget.setCurrentIndex(0)
            self.dibujarGraficos()
        elif textoSeleccionado == "Espectrograma":
            self.modoGrafico = "espectrograma"
            self.stackedWidget.setCurrentIndex(0)
            self.dibujarGraficos()
        elif textoSeleccionado == "Video":
            self.modoGrafico = "video"
            self.stackedWidget.setCurrentIndex(1)
            #Mostramos el cuadro actual o inicial si hay video disponible
            if len(self.audioCtrl.videoFrames) > 0 and not self.timerVideo.isActive():
                self.mostrarFrame(0)
            elif len(self.audioCtrl.videoFrames) == 0:
                self.labelVideo.clear()
                self.labelVideo.setText("No hay video capturado")

    def dibujarGraficos(self):
        """Delega la tarea de graficar a la logica del controlador de audio"""
        if self.modoGrafico == "video":
            return

        self.axOriginal.clear()
        self.axCensurado.clear()

        #Le pedimos al controlador de audio que use librosa para dibujar
        self.audioCtrl.graficarAudio(self.axOriginal, tipo=self.modoGrafico, censurado=False)
        self.audioCtrl.graficarAudio(self.axCensurado, tipo=self.modoGrafico, censurado=True)

        self.figura.tight_layout()
        self.canvas.draw()

    def reproducirOriginal(self):
        """Reproduce simultaneamente el audio original y el video capturado"""
        if self.grabando:
            return
        self.actualizarFocos("verde")
        QApplication.processEvents()

        self.audioCtrl.reproducirAudioOriginal()

        #Iniciamos la reproduccion sincronizada del video si existen cuadros
        if len(self.audioCtrl.videoFrames) > 0:
            self.iniciarReproduccionVideo()

    def reproducirCensurado(self):
        """Reproduce simultaneamente el audio censurado y el video capturado"""
        if self.grabando:
            return
        self.actualizarFocos("verde")
        QApplication.processEvents()

        self.audioCtrl.reproducirAudioCensurado()

        #Iniciamos la reproduccion sincronizada del video si existen cuadros
        if len(self.audioCtrl.videoFrames) > 0:
            self.iniciarReproduccionVideo()

    def iniciarReproduccionVideo(self):
        """Prepara el temporizador y comienza a reproducir los cuadros de video basandose en el tiempo real"""
        self.timerVideo.stop()
        self.inicioReproduccion = time.perf_counter()
        self.frameAnteriorMostrado = -1

        if self.audioCtrl.fpsVideo > 0:
            #Refrescar cada 15 ms y buscar el cuadro exacto que corresponde
            self.timerVideo.start(15)

    def actualizarFrameVideo(self):
        """Calcula el cuadro exacto segun el tiempo transcurrido para forzar sincronia total"""
        tiempoTranscurrido = time.perf_counter() - self.inicioReproduccion
        indiceEsperado = int(tiempoTranscurrido * self.audioCtrl.fpsVideo)

        if indiceEsperado < len(self.audioCtrl.videoFrames):
            #Solo actualizar si cambio el indice y estamos en la vista de video
            if self.modoGrafico == "video" and self.frameAnteriorMostrado != indiceEsperado:
                self.mostrarFrame(indiceEsperado)
                self.frameAnteriorMostrado = indiceEsperado
        else:
            self.timerVideo.stop()

    def mostrarFrame(self, indice):
        """Convierte y dibuja un cuadro de video en la etiqueta de la interfaz"""
        frame = self.audioCtrl.videoFrames[indice]
        h, w, ch = frame.shape
        bytesPerLine = ch * w
        qImg = QImage(frame.data, w, h, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)
        pixmap = pixmap.scaled(self.labelVideo.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.labelVideo.setPixmap(pixmap)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ventana = InterfazAudio()
    ventana.show()
    sys.exit(app.exec_())

    sys.exit(app.exec_())