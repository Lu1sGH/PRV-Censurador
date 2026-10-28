import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from spanlp.palabrota import Palabrota  # Modulo para censurar
from spanlp.domain.countries import Country  # Dominios de paises
from spanlp.domain.strategies import Preprocessing, TextToLower, RemoveAccents  # Preprocesamiento de texto
from spanlp.domain.strategies import LevenshteinDistance  # Metrica de distancia


class ClasificadorBinarioFoE(nn.Module):
    """
    Clasificador binario con regularizacion de potencial Student-t (Field of Experts)
    entrenado sobre la columna de decision binaria unica Y in {0, 1}.
    """
    def __init__(self, dim_embedding=512, device="cuda"):
        super().__init__()
        self.dim_embedding = dim_embedding
        self.device = device

        # Vector de proyeccion unico w in R^{d x 1}, sesgo escalar b in R^{1 x 1} y parametro alpha > 0
        self.filtro_w = nn.Parameter(torch.empty((dim_embedding, 1), device=device, dtype=torch.float32))
        self.sesgo_b = nn.Parameter(torch.zeros((1, 1), device=device, dtype=torch.float32))
        self.alpha = nn.Parameter(torch.ones((1, 1), device=device, dtype=torch.float32))

        nn.init.orthogonal_(self.filtro_w)

    def forward(self, v):
        """Devuelve el vector de logits binarios z = (V @ w + b) de dimension (batch,)"""
        v_norm = F.normalize(v.float(), p=2, dim=-1)
        return (torch.matmul(v_norm, self.filtro_w) + self.sesgo_b).squeeze(-1)

    def energia_potencial_student_t(self, z):
        """Calcula el potencial E_FoE(z) = alpha * log(1 + 0.5 * z^2)"""
        alpha_pos = F.softplus(self.alpha.squeeze())
        return torch.mean(alpha_pos * torch.log1p(0.5 * (z ** 2)))


class Censurador():
    def __init__(self, pais=None, ruta_pesos_foe="pesos_foe_binario.pt"):
        """Inicializa el censurador de groserias con un pais opcional y el clasificador binario FoE"""
        cosine = LevenshteinDistance(1)  # Usa distancia de Levenshtein para palabras parecidas
        paises = {'MX': Country.MEXICO, 'CO': Country.COLOMBIA, 'AR': Country.ARGENTINA}  # Interfaz para elegir el pais

        # Asigna todos por defecto si no hay pais
        if pais:
            self.palabrota = Palabrota(censor_char="*", countries=[paises[pais]], distance_metric=cosine)
        else:
            self.palabrota = Palabrota(censor_char="*", distance_metric=cosine)

        # Configuracion en GPU para la clasificacion binaria unica de Discurso de Odio (D.D.O.)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dim_embedding = 512
        self.ruta_pesos_foe = ruta_pesos_foe
        self.umbral_atribucion_odio = 0.18

        self.modelo_foe = ClasificadorBinarioFoE(
            dim_embedding=self.dim_embedding,
            device=self.device
        )

        self._inicializar_o_cargar_foe()

    def _extraer_embeddings_gpu(self, lista_textos):
        """
        Proyecta un lote de cadenas de texto en un tensor V in R^{B x d} en GPU
        mediante Hashing Sub-lexico de n-gramas de caracteres y tokens completos.
        """
        batch_size = len(lista_textos)
        matriz_v = torch.zeros((batch_size, self.dim_embedding), device=self.device, dtype=torch.float32)

        for fila_idx, texto in enumerate(lista_textos):
            limpio = self.preprocesarTexto(texto)
            tokens = re.findall(r'\w+', limpio)
            if not tokens:
                continue

            indices = []
            pesos = []
            for tok in tokens:
                palabra_pad = f"<{tok}>"
                for n in (3, 4):
                    for i in range(max(1, len(palabra_pad) - n + 1)):
                        sub = palabra_pad[i:i + n]
                        h = hash(sub)
                        indices.append(abs(h) % self.dim_embedding)
                        pesos.append(1.0 if (h & 1) == 0 else -1.0)

                h_w = hash(tok)
                indices.append(abs(h_w) % self.dim_embedding)
                pesos.append(2.0 if (h_w & 1) == 0 else -2.0)

            if indices:
                t_idx = torch.tensor(indices, device=self.device, dtype=torch.long)
                t_pesos = torch.tensor(pesos, device=self.device, dtype=torch.float32)
                matriz_v[fila_idx].index_add_(0, t_idx, t_pesos)

        return F.normalize(matriz_v, p=2, dim=-1)

    def _inicializar_o_cargar_foe(self):
        """
        Carga en VRAM los pesos binarios entrenados desde 'pesos_foe_binario.pt' o inicializa
        el hiperplano separador w in R^{d x 1} para ejecucion inmediata.
        """
        if os.path.exists(self.ruta_pesos_foe):
            try:
                estado = torch.load(self.ruta_pesos_foe, map_location=self.device, weights_only=True)
                self.modelo_foe.load_state_dict(estado)
                self.modelo_foe.eval()
                print(f"--> Pesos FoE binarios cargados en {self.device.upper()} desde '{self.ruta_pesos_foe}'.")
                return
            except Exception as e:
                print(f"--> Aviso al cargar '{self.ruta_pesos_foe}': {e}. Usando calibracion binaria base.")

        anclas_ddo = [
            "esa gente inmigrantes indigenas mujeres homosexuales minoria raza religion extranjeros pobres "
            "son inferiores plaga parasitos subhumanos escoria asquerosos animales enfermos no valen nada "
            "deberian morir matarlos eliminarlos exterminarlos prohibirles derechos echarlos quemarlos golpearlos"
        ]

        with torch.no_grad():
            v_bin = self._extraer_embeddings_gpu(anclas_ddo).squeeze(0)
            self.modelo_foe.filtro_w[:, 0] = 4.5 * v_bin
            self.modelo_foe.sesgo_b.fill_(-0.35)

        self.modelo_foe.eval()

    def entrenar_foe_offline(self, lista_textos_292k, vector_binario_y, epocas=20, batch_size=4096, lr=1e-2):
        """
        Entrena el clasificador binario unico sobre las 292,000 entradas de texto y la columna
        de decision binaria vector_binario_y in {0, 1}^N en menos de 3 minutos en GPU.
        """
        print(f"--> Iniciando entrenamiento FoE Binario en {self.device.upper()} ({len(lista_textos_292k)} muestras)...")
        self.modelo_foe.train()

        n_muestras = len(lista_textos_292k)
        x_emb = torch.zeros((n_muestras, self.dim_embedding), dtype=torch.float16, device=self.device)
        lote_extraccion = 2048

        with torch.no_grad():
            for inicio in range(0, n_muestras, lote_extraccion):
                fin = min(n_muestras, inicio + lote_extraccion)
                x_emb[inicio:fin] = self._extraer_embeddings_gpu(lista_textos_292k[inicio:fin]).half()

        y_tensor = torch.as_tensor(vector_binario_y, dtype=torch.float32, device=self.device).view(-1)
        optimizador = torch.optim.AdamW(self.modelo_foe.parameters(), lr=lr, weight_decay=1e-4)
        criterio_bce = nn.BCEWithLogitsLoss()

        for epoca in range(epocas):
            perm = torch.randperm(n_muestras, device=self.device)
            for i in range(0, n_muestras, batch_size):
                idx = perm[i:i + batch_size]
                x_b = x_emb[idx].float()
                y_b = y_tensor[idx]

                optimizador.zero_grad()
                logits_z = self.modelo_foe(x_b)
                loss_bce = criterio_bce(logits_z, y_b)
                loss_foe = self.modelo_foe.energia_potencial_student_t(logits_z)
                loss_total = loss_bce + 0.001 * loss_foe

                loss_total.backward()
                optimizador.step()

        self.modelo_foe.eval()
        torch.save(self.modelo_foe.state_dict(), self.ruta_pesos_foe)
        print(f"--> Entrenamiento binario finalizado. Pesos guardados en '{self.ruta_pesos_foe}'.")

    def censurarTexto(self, texto):
        """Censura el texto proporcionado reemplazando malas palabras con asteriscos"""
        return self.palabrota.censor(texto)

    def preprocesarTexto(self, texto):
        """Genera una secuencia de preprocesamiento para minusculas y quitar acentos"""
        estrategias = [TextToLower(), RemoveAccents()]
        # Aplica la secuencia de preprocesamiento   
        preprocesado = Preprocessing().clean(data=texto, clean_strategies=estrategias)
        return preprocesado

    def core(self, texto):
        """Realiza el proceso de normalizacion y censura del texto"""
        # Preprocesa el texto
        preprocesado = self.preprocesarTexto(texto)
        # Censura el texto
        censurado = self.censurarTexto(preprocesado)
        return preprocesado, censurado

    def evaluar_multimodal_foe(self, textoOriginal, palabrasConTiempo):
        """
        Ejecuta la censura en dos niveles usando una unica clasificacion binaria para D.D.O.:
          1. Nivel 1 (Lexico): Distancia de Levenshtein sobre cada palabra.
          2. Nivel 2 (Multimodal Binario): Oclusion en lote en GPU evaluando el logit binario z
             modulado por el factor acustico gamma(w_i) del espectrograma y la onda.
        """
        if not palabrasConTiempo:
            _, textoCensurado = self.core(textoOriginal)
            return textoCensurado, []

        palabrasASilenciar = []
        indices_censurados_lex = set()
        tokens_salida = []

        # --- NIVEL 1: FILTRO LÉXICO (LEVENSHTEIN) ---
        for i, p in enumerate(palabrasConTiempo):
            _, cens = self.core(p["word"])
            if "*" in cens:
                palabrasASilenciar.append(p)
                indices_censurados_lex.add(i)
                tokens_salida.append("*" * len(p["word"]))
            else:
                tokens_salida.append(p["word"])

        # --- NIVEL 2: CLASIFICACIÓN BINARIA MULTIMODAL (ESPECTROGRAMA + OCLUSIÓN FoE) ---
        k = len(palabrasConTiempo)
        palabras_limpias = [p["word"] for p in palabrasConTiempo]
        oracion_completa = " ".join(palabras_limpias)

        # Fila 0: Oracion completa; Fila i + 1: Oracion sin la palabra i-esima
        lote_textos_ocluidos = [oracion_completa]
        for i in range(k):
            variante = " ".join(palabras_limpias[:i] + palabras_limpias[i + 1:])
            lote_textos_ocluidos.append(variante if variante.strip() else " ")

        with torch.no_grad():
            vectores_v = self._extraer_embeddings_gpu(lote_textos_ocluidos)  # (k + 1, d)
            logits_binarios = self.modelo_foe(vectores_v)  # Vector 1D de dimension (k + 1,)
            logit_oracion_base = float(logits_binarios[0].item())

            for i in range(k):
                if i in indices_censurados_lex:
                    continue

                p_info = palabrasConTiempo[i]
                gamma_i = float(p_info.get("gamma", 1.0))
                prob_asr = float(p_info.get("probability", 1.0))

                # Reduccion del logit binario al ocluir la palabra i
                delta_logit = logit_oracion_base - float(logits_binarios[i + 1].item())

                # Probabilidad binaria acoplada con el modulador acustico del espectrograma
                ganancia_efectiva = gamma_i * max(0.5, prob_asr)
                prob_ddo_binaria = float(torch.sigmoid(torch.tensor(ganancia_efectiva * logit_oracion_base)).item())
                importancia_palabra = delta_logit * ganancia_efectiva

                if prob_ddo_binaria >= 0.5 and importancia_palabra > self.umbral_atribucion_odio:
                    palabrasASilenciar.append(p_info)
                    tokens_salida[i] = f"[D.D.O.: {'*' * max(3, len(p_info['word']))}]"
                    print(
                        f"   [FoE Binario D.D.O.] Palabra: '{p_info['word']}' | "
                        f"Prob. D.D.O.: {prob_ddo_binaria:.2f} | "
                        f"Gamma Acústico: {gamma_i:.2f} | Atribución: {importancia_palabra:.3f}"
                    )

        texto_final_censurado = " ".join(tokens_salida)
        return texto_final_censurado, palabrasASilenciar