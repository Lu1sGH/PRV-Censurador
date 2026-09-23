import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QComboBox, QGroupBox, QSpacerItem, QSizePolicy, QGridLayout
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import Qt, QSize
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from AudioCtrl import AudioCtrl
from Censurador import Censurador
from Whisper import WhisperTranscriptor

class InterfazAudio(QMainWindow):
    def __init__(self):
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

        # 1. FOCOS INDICADORES (ROJO, AZUL, VERDE)
        headerLayout = QHBoxLayout()
        headerLayout.setAlignment(Qt.AlignLeft)
        
        focosGrid = QGridLayout()
        focosGrid.setSpacing(10)
        
        # Foco Rojo
        self.focoRojo = QLabel()
        self.focoRojo.setFixedSize(16, 16)
        labelRojo = QLabel("Grabando")
        labelRojo.setFont(QFont("Arial", 9))
        labelRojo.setAlignment(Qt.AlignCenter)
        focosGrid.addWidget(self.focoRojo, 0, 0, alignment=Qt.AlignCenter)
        focosGrid.addWidget(labelRojo, 1, 0, alignment=Qt.AlignTop | Qt.AlignHCenter)
        
        # Foco Azul
        self.focoAzul = QLabel()
        self.focoAzul.setFixedSize(16, 16)
        labelAzul = QLabel("Procesando\ny censurando")
        labelAzul.setAlignment(Qt.AlignCenter)
        labelAzul.setFont(QFont("Arial", 9))
        focosGrid.addWidget(self.focoAzul, 0, 1, alignment=Qt.AlignCenter)
        focosGrid.addWidget(labelAzul, 1, 1, alignment=Qt.AlignTop | Qt.AlignHCenter)
        
        # Foco Verde
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

        # 2. GRÁFICOS MATPLOTLIB
        self.figura = Figure(figsize=(6, 4))
        self.canvas = FigureCanvas(self.figura)
        self.axOriginal = self.figura.add_subplot(211)
        self.axOriginal.set_title("Original")
        self.axCensurado = self.figura.add_subplot(212)
        self.axCensurado.set_title("Censurado")
        self.figura.tight_layout()
        layout.addWidget(self.canvas)

        self.modoGrafico = "onda"

        # ESTILO ANALÓGICO METÁLICO PARA BOTONES
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

        # 3. BOTONES DE ACCIÓN
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
        
        # 4. RECUADROS DE TEXTO (QGroupBox)
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
        
        # Inicializar todos apagados menos el verde
        self.actualizarFocos("verde")

    def actualizarFocos(self, focoActivo):
        """Apaga todos los focos y enciende solo el solicitado ('rojo', 'azul', 'verde' o 'ninguno')."""
        estiloApagado = "QLabel { background-color: #bbbbbb; border-radius: 8px; }"
        
        # Apagar todos primero
        self.focoRojo.setStyleSheet(estiloApagado)
        self.focoAzul.setStyleSheet(estiloApagado)
        self.focoVerde.setStyleSheet(estiloApagado)
        
        # Encender el correspondiente (sin bordes tan toscos, solo color vivo)
        if focoActivo == "rojo":
            self.focoRojo.setStyleSheet("QLabel { background-color: #ff3333; border-radius: 8px; }")
        elif focoActivo == "azul":
            self.focoAzul.setStyleSheet("QLabel { background-color: #3385ff; border-radius: 8px; }")
        elif focoActivo == "verde":
            self.focoVerde.setStyleSheet("QLabel { background-color: #00cc66; border-radius: 8px; }")

    def toggleGrabacion(self):
        """Lógica para grabar y detener el audio con AudioCtrl."""
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
            
            self.audioCtrl.iniciarGrabacion()
        else:
            self.grabando = False
            self.btnGrabar.setText(" GRABAR")
            self.actualizarFocos("azul")
            QApplication.processEvents()
            
            self.audioCtrl.detenerGrabacion()
            
            #Normalizamos inmediatamente para precalcular las matrices finales
            self.audioCtrl.normalizarAudio()
            
            # 1. GRAFICAR INMEDIATAMENTE AL DETENER (con el audio original precalculado)
            self.dibujarGraficos()
            QApplication.processEvents()
            
            #Transcribir el audio a texto y obtener las palabras con sus marcas de tiempo
            textoOriginal, palabrasConTiempo = self.whisperTranscriptor.transcribir(self.audioCtrl.audioData, self.audioCtrl.sampleRate)
            
            if textoOriginal:
                #Texto completo censurado (para la etiqueta)
                _, textoCensurado = self.censurador.core(textoOriginal)
                self.labelTranscripcion.setText(textoOriginal)
                self.labelCensurado.setText(textoCensurado)
                
                #Revisar qué palabras exactas se censuran para obtener sus tiempos
                palabras_a_silenciar = []
                for p in palabrasConTiempo:
                    #Evaluamos la palabra aislada
                    _, cens = self.censurador.core(p["word"])
                    #Si el censurador puso un '*' en la palabra, significa que es mala
                    if "*" in cens:
                        palabras_a_silenciar.append(p)
                
                #Aplicamos la censura al arreglo de audio (silenciamos)
                self.audioCtrl.aplicarCensuraAudio(palabras_a_silenciar)
                
                # 2. VOLVER A GRAFICAR UNA VEZ CENSURADO
                self.dibujarGraficos()
                self.actualizarFocos("verde")
            else:
                self.labelTranscripcion.setText("(No se detectó voz)")
                self.labelCensurado.setText("-")
                #Ya está graficado el original de todos modos
                self.actualizarFocos("verde")

    def cambiarModoGrafico(self, textoSeleccionado):
        """Cambia entre la vista de forma de onda y espectrograma según el combobox."""
        if textoSeleccionado == "Forma de Onda":
            self.modoGrafico = "onda"
        else:
            self.modoGrafico = "espectrograma"
        
        #Redibujar si hay datos
        self.dibujarGraficos()

    def dibujarGraficos(self):
        """Delega la tarea de graficar a la lógica de AudioCtrl.py."""
        self.axOriginal.clear()
        self.axCensurado.clear()
        
        # Le pedimos al controlador de audio que use librosa para dibujar en nuestros ejes (ax)
        self.audioCtrl.graficarAudio(self.axOriginal, tipo=self.modoGrafico, censurado=False)
        self.audioCtrl.graficarAudio(self.axCensurado, tipo=self.modoGrafico, censurado=True)
            
        self.figura.tight_layout()
        self.canvas.draw()

    def reproducirOriginal(self):
        """Reproduce el audio original."""
        if self.grabando:
            return
        self.actualizarFocos("verde")
        QApplication.processEvents()
        self.audioCtrl.reproducirAudioOriginal()
        self.actualizarFocos("verde")

    def reproducirCensurado(self):
        """Reproduce el audio censurado."""
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
