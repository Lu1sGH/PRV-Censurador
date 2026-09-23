import sys #Manejo del sistema
import time #Medicion de tiempos
import numpy as np #Manejo de arreglos matematicos
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QComboBox, QGroupBox, QSpacerItem, QSizePolicy, QGridLayout #Elementos graficos principales
from PyQt5.QtGui import QIcon, QFont #Iconos y fuentes
from PyQt5.QtCore import Qt, QSize #Constantes Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas #Lienzo para graficas
from matplotlib.figure import Figure #Figura grafica
from AudioCtrl import AudioCtrl #Controlador de audio
from Censurador import Censurador #Procesador de texto
from Whisper import WhisperTranscriptor #Transcriptor de voz

class InterfazAudio(QMainWindow):
    def __init__(self):
        """Inicializa la interfaz principal y sus componentes visuales y logicos"""
        super().__init__()
        self.audioCtrl = AudioCtrl()
        self.censurador = Censurador('MX')
        self.whisperTranscriptor = WhisperTranscriptor('small')
        
        self.setWindowTitle("Aplicación de Audio - Censurador")
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
        self.comboGrafico.setFont(QFont("Arial", 10))
        self.comboGrafico.currentTextChanged.connect(self.cambiarModoGrafico)
        headerLayout.addWidget(self.comboGrafico)
        
        layout.addLayout(headerLayout)

        #Graficos Matplotlib
        self.figura = Figure(figsize=(6, 4))
        self.canvas = FigureCanvas(self.figura)
        self.axOriginal = self.figura.add_subplot(211)
        self.axOriginal.set_title("Original")
        self.axCensurado = self.figura.add_subplot(212)
        self.axCensurado.set_title("Censurado")
        self.figura.tight_layout()
        layout.addWidget(self.canvas)

        self.modoGrafico = "onda"

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
        
        #Recuadros de texto (QGroupBox)
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
        """Maneja el inicio y detencion de la grabacion procesando el audio en cascada"""
        if not self.grabando:
            self.grabando = True
            self.btnGrabar.setText(" DETENER")
            self.actualizarFocos("rojo")
            
            self.labelTranscripcion.setText("...")
            self.labelCensurado.setText("...")
            
            self.axOriginal.clear()
            self.axOriginal.set_title("Original")
            self.axCensurado.clear()
            self.axCensurado.set_title("Censurado")
            self.canvas.draw()
            
            self.t_inicio_grabacion = time.perf_counter()
            print(f"\n[{time.strftime('%H:%M:%S')}] --- INICIO ---")
            self.audioCtrl.iniciarGrabacion()
        else:
            self.grabando = False
            self.btnGrabar.setText(" GRABAR")
            self.actualizarFocos("azul")
            QApplication.processEvents()
            
            t_fin_grabacion = time.perf_counter()
            print(f"[{time.strftime('%H:%M:%S')}] --- Grabación detenida. Duración de la captura: {t_fin_grabacion - self.t_inicio_grabacion:.3f} s ---")
            
            self.audioCtrl.detenerGrabacion()
            
            #Tiempo de preprocesamiento
            t_inicio_pre = time.perf_counter()
            #Normalizamos inmediatamente para precalcular las matrices finales
            self.audioCtrl.normalizarAudio()
            t_fin_pre = time.perf_counter()
            print(f"[{time.strftime('%H:%M:%S')}] --- Tiempo de preprocesamiento del audio: {t_fin_pre - t_inicio_pre:.3f} s ---")
            
            #Tiempo de primera graficacion
            t_inicio_graf = time.perf_counter()
            self.dibujarGraficos()
            QApplication.processEvents()
            t_fin_graf = time.perf_counter()
            print(f"[{time.strftime('%H:%M:%S')}] --- Tiempo de primera graficación: {t_fin_graf - t_inicio_graf:.3f} s ---")
            
            #Tiempo de inferencia de Whisper
            t_inicio_whisper = time.perf_counter()
            textoOriginal, palabrasConTiempo = self.whisperTranscriptor.transcribir(self.audioCtrl.audioData, self.audioCtrl.sampleRate)
            t_fin_whisper = time.perf_counter()
            print(f"[{time.strftime('%H:%M:%S')}] --- Tiempo de inferencia de Whisper: {t_fin_whisper - t_inicio_whisper:.3f} s ---")
            
            if textoOriginal:
                #Tiempo de censura
                t_inicio_censura = time.perf_counter()
                
                #Texto completo censurado para la etiqueta
                _, textoCensurado = self.censurador.core(textoOriginal)
                self.labelTranscripcion.setText(textoOriginal)
                self.labelCensurado.setText(textoCensurado)
                
                #Revisar que palabras exactas se censuran para obtener sus tiempos
                palabrasASilenciar = []
                for p in palabrasConTiempo:
                    #Evaluamos la palabra aislada
                    _, cens = self.censurador.core(p["word"])
                    #Si el censurador puso un asterisco en la palabra significa que es mala
                    if "*" in cens:
                        palabrasASilenciar.append(p)
                
                #Aplicamos la censura al arreglo de audio y matriz
                self.audioCtrl.aplicarCensuraAudio(palabrasASilenciar)
                
                t_fin_censura = time.perf_counter()
                print(f"[{time.strftime('%H:%M:%S')}] --- Tiempo de censura (NLP y modificación del audio): {t_fin_censura - t_inicio_censura:.3f} s ---")
                
                #Tiempo de segunda graficacion
                t_inicio_graf2 = time.perf_counter()
                self.dibujarGraficos()
                t_fin_graf2 = time.perf_counter()
                print(f"[{time.strftime('%H:%M:%S')}] --- Tiempo de segunda graficación: {t_fin_graf2 - t_inicio_graf2:.3f} s ---")
                
                self.actualizarFocos("verde")
            else:
                self.labelTranscripcion.setText("(No se detectó voz)")
                self.labelCensurado.setText("-")
                #Ya esta graficado el original de todos modos
                self.actualizarFocos("verde")
                
            #Fin del proceso
            t_fin_total = time.perf_counter()
            print(f"[{time.strftime('%H:%M:%S')}] --- FIN. Tiempo total de espera del usuario: {t_fin_total - t_fin_grabacion:.3f} s ---\n")

    def cambiarModoGrafico(self, textoSeleccionado):
        """Cambia entre la vista de forma de onda y espectrograma segun el combobox"""
        if textoSeleccionado == "Forma de Onda":
            self.modoGrafico = "onda"
        else:
            self.modoGrafico = "espectrograma"
        
        #Redibujar si hay datos
        self.dibujarGraficos()

    def dibujarGraficos(self):
        """Delega la tarea de graficar a la logica del controlador de audio"""
        self.axOriginal.clear()
        self.axCensurado.clear()
        
        #Le pedimos al controlador de audio que use librosa para dibujar
        self.audioCtrl.graficarAudio(self.axOriginal, tipo=self.modoGrafico, censurado=False)
        self.audioCtrl.graficarAudio(self.axCensurado, tipo=self.modoGrafico, censurado=True)
            
        self.figura.tight_layout()
        self.canvas.draw()

    def reproducirOriginal(self):
        """Reproduce el audio original por los altavoces"""
        if self.grabando:
            return
        self.actualizarFocos("verde")
        QApplication.processEvents()
        self.audioCtrl.reproducirAudioOriginal()
        self.actualizarFocos("verde")

    def reproducirCensurado(self):
        """Reproduce el audio censurado por los altavoces"""
        if self.grabando:
            return
        self.actualizarFocos("verde")
        QApplication.processEvents()
        self.audioCtrl.reproducirAudioCensurado()
        self.actualizarFocos("verde")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ventana = InterfazAudio()
    ventana.show()
    sys.exit(app.exec_())
