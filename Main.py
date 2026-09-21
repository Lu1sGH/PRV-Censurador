import sys
import torch # <- IMPORTANTE: Importar torch antes de PyQt5 evita el error de c10.dll en Windows
from PyQt5.QtWidgets import QApplication
from Interfaz import InterfazAudio

if __name__ == "__main__":
    #Iniciar la aplicación de PyQt
    app = QApplication(sys.argv)
    ventana = InterfazAudio()
    
    ventana.show()
    sys.exit(app.exec_())