import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QLabel
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from AudioPro import AudioPro
from Censurador import Censurador

class InterfazAudio(QMainWindow):
    def __init__(self):
        super().__init__()
        self.audioCtrl = AudioPro()
        self.censurador = Censurador('MX')
        
        self.setWindowTitle("Aplicación de Audio - Censurador")
        self.setGeometry(100, 100, 700, 700)

        widgetCentral = QWidget()
        layout = QVBoxLayout()
        self.setCentralWidget(widgetCentral)

        self.labelEstado = QLabel("Estado: Listo", self)
        self.labelEstado.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.labelEstado)

        #Canvas de Matplotlib con 2 subgráficos
        self.figura = Figure(figsize=(6, 5))
        self.canvas = FigureCanvas(self.figura)
        
        #Arriba: Original
        self.axOriginal = self.figura.add_subplot(211)
        self.axOriginal.set_title("Forma de onda Original")
        
        #Abajo: Censurada
        self.axCensurado = self.figura.add_subplot(212)
        self.axCensurado.set_title("Forma de onda Censurada")
        
        self.figura.tight_layout()
        layout.addWidget(self.canvas)

        #Botón para iniciar/detener grabación
        self.btnGrabar = QPushButton("Iniciar grabación", self)
        self.btnGrabar.clicked.connect(self.toggleGrabacion)
        layout.addWidget(self.btnGrabar)

        #Layout horizontal para los dos botones de reproducción
        botonesLayout = QHBoxLayout()
        
        self.btnReproducirOrig = QPushButton("Reproducir Original", self)
        self.btnReproducirOrig.clicked.connect(self.reproducirOriginal)
        botonesLayout.addWidget(self.btnReproducirOrig)
        
        self.btnReproducirCens = QPushButton("Reproducir Censurado", self)
        self.btnReproducirCens.clicked.connect(self.reproducirCensurado)
        botonesLayout.addWidget(self.btnReproducirCens)
        
        layout.addLayout(botonesLayout)
        
        self.labelTranscripcion = QLabel("Texto Original: -", self)
        self.labelTranscripcion.setAlignment(Qt.AlignCenter)
        self.labelTranscripcion.setWordWrap(True)
        layout.addWidget(self.labelTranscripcion)
        
        self.labelCensurado = QLabel("Texto Censurado: -", self)
        self.labelCensurado.setAlignment(Qt.AlignCenter)
        self.labelCensurado.setWordWrap(True)
        self.labelCensurado.setStyleSheet("color: red; font-weight: bold;")
        layout.addWidget(self.labelCensurado)

        widgetCentral.setLayout(layout)
        self.grabando = False

    def toggleGrabacion(self):
        """Lógica para grabar y detener el audio con AudioPro."""
        if not self.grabando:
            self.grabando = True
            self.btnGrabar.setText("Detener grabación")
            self.labelEstado.setText("Estado: Grabando...")
            
            self.labelTranscripcion.setText("Texto Original: Escuchando...")
            self.labelCensurado.setText("Texto Censurado: ...")
            
            self.axOriginal.clear()
            self.axOriginal.set_title("Grabando...")
            self.axCensurado.clear()
            self.axCensurado.set_title("Esperando procesamiento...")
            self.canvas.draw()
            
            self.audioCtrl.iniciarGrabacion()
        else:
            self.grabando = False
            self.btnGrabar.setText("Iniciar grabación")
            self.labelEstado.setText("Estado: Procesando audio con Whisper (puede tardar unos segundos)...")
            QApplication.processEvents()
            
            self.audioCtrl.detenerGrabacion()
            
            #Transcribir el audio a texto y obtener las palabras con sus marcas de tiempo
            textoOriginal, palabrasConTiempo = self.audioCtrl.transcribirAudio()
            
            if textoOriginal:
                #Texto completo censurado (para la etiqueta)
                _, textoCensurado = self.censurador.core(textoOriginal)
                self.labelTranscripcion.setText(f"Texto Original: {textoOriginal}")
                self.labelCensurado.setText(f"Texto Censurado: {textoCensurado}")
                
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
                
                #Dibujamos ambas gráficas
                self.dibujarOndas()
                self.labelEstado.setText("Estado: Grabación, transcripción y censura finalizadas")
            else:
                self.labelTranscripcion.setText("Texto Original: (No se detectó voz)")
                self.labelCensurado.setText("Texto Censurado: -")
                self.dibujarOndas()
                self.labelEstado.setText("Estado: Finalizado (Sin voz)")

    def dibujarOndas(self):
        """Dibuja ambas formas de onda en los subgráficos."""
        audioOrig = self.audioCtrl.audioData.flatten() if self.audioCtrl.audioData is not None else []
        audioCens = self.audioCtrl.audioCensurado.flatten() if self.audioCtrl.audioCensurado is not None else []
        
        self.axOriginal.clear()
        self.axCensurado.clear()
        
        if len(audioOrig) > 0:
            tiempo = np.arange(len(audioOrig)) / self.audioCtrl.sampleRate
            self.axOriginal.plot(tiempo, audioOrig, color='blue')
            self.axOriginal.set_title("Forma de onda Original")
            self.axOriginal.set_ylabel("Amplitud")
            
        if len(audioCens) > 0:
            tiempoCens = np.arange(len(audioCens)) / self.audioCtrl.sampleRate
            self.axCensurado.plot(tiempoCens, audioCens, color='red')
            self.axCensurado.set_title("Forma de onda Censurada (Silencios aplicados)")
            self.axCensurado.set_ylabel("Amplitud")
            self.axCensurado.set_xlabel("Tiempo (s)")
            
        self.figura.tight_layout()
        self.canvas.draw()

    def reproducirOriginal(self):
        """Reproduce el audio original."""
        if self.grabando:
            return
        self.labelEstado.setText("Estado: Reproduciendo Original...")
        self.audioCtrl.reproducirAudioOriginal()
        self.labelEstado.setText("Estado: Listo")

    def reproducirCensurado(self):
        """Reproduce el audio censurado."""
        if self.grabando:
            return
        self.labelEstado.setText("Estado: Reproduciendo Censurado...")
        self.audioCtrl.reproducirAudioCensurado()
        self.labelEstado.setText("Estado: Listo")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ventana = InterfazAudio()
    ventana.show()
    sys.exit(app.exec_())
