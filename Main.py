import sys #Manejo del sistema
import torch #Importar torch antes de PyQt5 evita error en Windows
from PyQt5.QtWidgets import QApplication #Aplicacion grafica
from Interfaz import InterfazAudio #Ventana principal

if __name__ == "__main__":
    #Iniciar la aplicacion de PyQt
    app = QApplication(sys.argv)
    ventana = InterfazAudio()
    ventana.show()
    sys.exit(app.exec_())