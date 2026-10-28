import sys  # Manejo del sistema
import torch  # Importar torch antes de PyQt5 evita error en Windows
from PyQt5.QtWidgets import QApplication  # Aplicacion grafica
from Interfaz import InterfazAudio  # Ventana principal

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Iniciar la aplicacion de PyQt
    app = QApplication(sys.argv)
    ventana = InterfazAudio()
    ventana.show()
    sys.exit(app.exec_())