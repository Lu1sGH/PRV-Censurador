import os #Manejo de rutas y archivos del sistema
import re #Expresiones regulares para limpiar signos adheridos
import torch #Tensores y calculo en tarjeta grafica
import torch.nn as nn #Modulos de redes neuronales
import torch.nn.functional as F #Funciones matematicas y de activacion
from transformers import AutoTokenizer, AutoModel #Tokenizador y codificador contextual preentrenado
from spanlp.palabrota import Palabrota #Modulo para censurar palabras altisonantes
from spanlp.domain.countries import Country #Dominios de paises
from spanlp.domain.strategies import Preprocessing, TextToLower, RemoveAccents #Estrategias de normalizacion de texto
from spanlp.domain.strategies import LevenshteinDistance #Metrica de distancia entre cadenas


class TransformerMLPFoE(nn.Module):
    def __init__(self, nombre_modelo="dccuchile/distilbert-base-spanish-uncased", hidden_dim=128):
        """Inicializa el codificador contextual compacto con perceptron multicapa y regularizacion Student-t"""
        super().__init__()
        self.encoder = AutoModel.from_pretrained(nombre_modelo)
        embedDim = self.encoder.config.hidden_size

        #Congelar capas inferiores y dejar entrenables solo las ultimas dos capas
        for param in self.encoder.parameters():
            param.requires_grad = False

        if hasattr(self.encoder, "transformer") and hasattr(self.encoder.transformer, "layer"):
            for capa in self.encoder.transformer.layer[-2:]:
                for param in capa.parameters():
                    param.requires_grad = True
        elif hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layer"):
            for capa in self.encoder.encoder.layer[-2:]:
                for param in capa.parameters():
                    param.requires_grad = True

        #Clasificador multicapa no lineal hacia una neurona de salida
        self.mlp = nn.Sequential(
            nn.Linear(embedDim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, 1)
        )

        #Parametro de potencial Student-t positivo
        self.alpha = nn.Parameter(torch.ones((1, 1), dtype=torch.float32))

    def mean_pooling_enmascarado(self, last_hidden_state, attention_mask):
        """Calcula el promedio contextual ponderado por la mascara de atencion y normaliza el vector"""
        maskExpanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sumaEmbeddings = torch.sum(last_hidden_state * maskExpanded, dim=1)
        sumaMask = maskExpanded.sum(dim=1).clamp(min=1e-9)
        u = sumaEmbeddings / sumaMask
        return F.normalize(u, p=2, dim=-1)

    def forward(self, input_ids, attention_mask):
        """Ejecuta el paso hacia adelante devolviendo el vector de salidas lineales binarias"""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        uNorm = self.mean_pooling_enmascarado(outputs.last_hidden_state, attention_mask)
        logits = self.mlp(uNorm)
        return logits.squeeze(-1)

    def energia_potencial_student_t(self, z):
        """Calcula la energia de potencial Student-t sobre las salidas lineales"""
        alphaPos = F.softplus(self.alpha.squeeze())
        return torch.mean(alphaPos * torch.log1p(0.5 * (z.float() ** 2)))


class Censurador():
    def __init__(
        self,
        pais=None,
        ruta_pesos_foe="pesos_foe_binario.pt",
        modelo_transformer="dccuchile/distilbert-base-spanish-uncased"
    ):
        """Inicializa el normalizador de texto, el detector de discurso de odio y el filtro de palabras altisonantes"""
        cosine = LevenshteinDistance(1) #Usa distancia de Levenshtein para palabras parecidas
        paises = {'MX': Country.MEXICO, 'CO': Country.COLOMBIA, 'AR': Country.ARGENTINA} #Interfaz para elegir el pais

        #Asigna todos por defecto si no hay pais
        if pais:
            self.palabrota = Palabrota(censor_char="*", countries=[paises[pais]], distance_metric=cosine)
        else:
            self.palabrota = Palabrota(censor_char="*", distance_metric=cosine)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ruta_pesos_foe = ruta_pesos_foe
        self.nombre_transformer = modelo_transformer
        self.max_length = 64
        self.umbral_atribucion_odio = 0.15

        print(f"--> Cargando tokenizador y arquitectura Transformer+MLP '{modelo_transformer}' en {self.device.upper()}...")
        self.tokenizer = AutoTokenizer.from_pretrained(modelo_transformer)
        self.modelo_foe = TransformerMLPFoE(nombre_modelo=modelo_transformer, hidden_dim=128).to(self.device)

        self._inicializar_o_cargar_foe()

    def _inicializar_o_cargar_foe(self):
        """Carga los pesos entrenados desde disco o calibra el cabezal base si no existen"""
        rutaEfectiva = self.ruta_pesos_foe
        if not os.path.exists(rutaEfectiva):
            rutaAlternativa = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.ruta_pesos_foe)
            if os.path.exists(rutaAlternativa):
                rutaEfectiva = rutaAlternativa

        if os.path.exists(rutaEfectiva):
            try:
                estado = torch.load(rutaEfectiva, map_location=self.device, weights_only=True)
                self.modelo_foe.load_state_dict(estado, strict=False)
                self.modelo_foe.eval()
                print(f"--> Pesos Transformer+MLP FoE cargados exitosamente desde '{rutaEfectiva}'.")
                return
            except Exception as e:
                print(f"--> Aviso al cargar '{rutaEfectiva}': {e}. Calibrando cabezal base.")

        #Calibracion rapida de respaldo mediante descenso breve sobre prototipos contrastivos
        prototiposPositivos = [
            "esa gente son una plaga inferior y deberian eliminarlos a todos",
            "los inmigrantes e indigenas son escoria subhumana que no merece derechos",
            "hay que matar y exterminar a esa minoria porque son basura asquerosa"
        ]
        prototiposNegativos = [
            "hola buenos dias estamos probando el sistema de grabacion de audio",
            "la reunion de trabajo sera manana por la tarde en la oficina",
            "el clima en la ciudad es agradable y la conversacion es completamente normal"
        ]

        textosInit = prototiposPositivos + prototiposNegativos
        etiquetasInit = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], device=self.device)

        self.modelo_foe.mlp.train()
        optInit = torch.optim.AdamW(self.modelo_foe.mlp.parameters(), lr=1e-2)
        bceInit = nn.BCEWithLogitsLoss()

        enc = self.tokenizer(
            textosInit,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)

        for _ in range(25):
            optInit.zero_grad()
            z = self.modelo_foe(enc["input_ids"], enc["attention_mask"])
            loss = bceInit(z, etiquetasInit)
            loss.backward()
            optInit.step()

        self.modelo_foe.eval()

    def entrenar_foe_offline(self, lista_textos_292k, vector_binario_y, epocas=2, batch_size=128, lr=3e-4):
        """Entrena las ultimas dos capas del codificador y el cabezal multicapa usando precision mixta"""
        nMuestras = len(lista_textos_292k)
        print(f"--> [1/2] Normalizando y tokenizando {nMuestras:,} textos...")
        textosNormalizados = [self.normalizarTexto(t) for t in lista_textos_292k]
        tokensCorpus = self.tokenizer(
            textosNormalizados,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        inputIdsAll = tokensCorpus["input_ids"]
        attentionMaskAll = tokensCorpus["attention_mask"]
        yAll = torch.as_tensor(vector_binario_y, dtype=torch.float32).view(-1)

        #Compensacion de desbalance de clases en la columna binaria
        numPos = torch.sum(yAll == 1.0).clamp(min=1.0)
        numNeg = torch.sum(yAll == 0.0).clamp(min=1.0)
        pesoPos = (numNeg / numPos).to(self.device)
        criterioBce = nn.BCEWithLogitsLoss(pos_weight=pesoPos)

        #Tasas de aprendizaje diferenciadas
        parametrosOptim = [
            {"params": [p for _, p in self.modelo_foe.encoder.named_parameters() if p.requires_grad], "lr": 5e-5},
            {"params": self.modelo_foe.mlp.parameters(), "lr": lr},
            {"params": [self.modelo_foe.alpha], "lr": 1e-3}
        ]
        optimizador = torch.optim.AdamW(parametrosOptim, weight_decay=1e-2)
        usarAmp = self.device == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=usarAmp)

        self.modelo_foe.train()
        print(f"--> [2/2] Entrenando Transformer + MLP en {self.device.upper()} (Lote={batch_size})...")

        for epoca in range(epocas):
            perm = torch.randperm(nMuestras)
            lossAcumulada = 0.0
            pasos = 0

            for i in range(0, nMuestras, batch_size):
                idx = perm[i:i + batch_size]
                bIds = inputIdsAll[idx].to(self.device, non_blocking=True)
                bMask = attentionMaskAll[idx].to(self.device, non_blocking=True)
                bY = yAll[idx].to(self.device, non_blocking=True)

                optimizador.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=usarAmp):
                    logitsZ = self.modelo_foe(bIds, bMask)
                    lossBce = criterioBce(logitsZ, bY)
                    lossFoe = self.modelo_foe.energia_potencial_student_t(logitsZ)
                    lossTotal = lossBce + 0.001 * lossFoe

                scaler.scale(lossTotal).backward()
                scaler.step(optimizador)
                scaler.update()

                lossAcumulada += float(lossTotal.item())
                pasos += 1

            print(f"    Epoca {epoca + 1}/{epocas} finalizada | Perdida media: {lossAcumulada / max(1, pasos):.4f}")

        self.modelo_foe.eval()
        torch.save(self.modelo_foe.state_dict(), self.ruta_pesos_foe)
        print(f"--> Pesos de Transformer + MLP guardados en '{self.ruta_pesos_foe}'.")

    def preprocesarTexto(self, texto):
        """Convierte el texto a minusculas y elimina acentos usando estrategias de limpieza"""
        estrategias = [TextToLower(), RemoveAccents()]
        return Preprocessing().clean(data=texto, clean_strategies=estrategias)

    def normalizarTexto(self, texto):
        """Paso 1 del censurador que convierte a minusculas, quita acentos y limpia signos de puntuacion"""
        limpio = self.preprocesarTexto(texto)
        limpio = re.sub(r"[^\w\s]", "", limpio)
        return re.sub(r"\s+", " ", limpio).strip()

    def censurarTexto(self, texto):
        """Censura palabras altisonantes reemplazandolas con asteriscos"""
        return self.palabrota.censor(texto)

    def core(self, texto):
        """Realiza normalizacion y censura lexica directa sobre una cadena de texto"""
        preprocesado = self.normalizarTexto(texto)
        censurado = self.censurarTexto(preprocesado)
        return preprocesado, censurado

    def detectarCensurarOdio(self, palabrasNorm, palabrasConTiempo, tokensSalida, palabrasASilenciar):
        """Paso 2 del censurador que detecta y censura discurso de odio sobre el texto normalizado"""
        indicesCensuradosOdio = set()
        k = len(palabrasNorm)
        if k == 0:
            return indicesCensuradosOdio

        oracionNormalizada = " ".join([w for w in palabrasNorm if w])
        if not oracionNormalizada:
            return indicesCensuradosOdio

        #Fila 0 contiene la oracion normalizada completa y las demas omiten una palabra a la vez
        loteTextosOcluidos = [oracionNormalizada]
        for i in range(k):
            variante = " ".join([w for j, w in enumerate(palabrasNorm) if j != i and w])
            loteTextosOcluidos.append(variante if variante.strip() else " ")

        usarAmp = self.device == "cuda"
        with torch.no_grad():
            loteTokens = self.tokenizer(
                loteTextosOcluidos,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)

            with torch.amp.autocast("cuda", enabled=usarAmp):
                logitsBinarios = self.modelo_foe(
                    loteTokens["input_ids"],
                    loteTokens["attention_mask"]
                ).float()

            logitOracionBase = float(logitsBinarios[0].item())

            for i in range(k):
                if not palabrasNorm[i]:
                    continue

                pInfo = palabrasConTiempo[i]
                gammaI = float(pInfo.get("gamma", 1.0))
                probAsr = float(pInfo.get("probability", 1.0))

                deltaLogit = logitOracionBase - float(logitsBinarios[i + 1].item())
                gananciaEfectiva = gammaI * max(0.5, probAsr)
                probDdoBinaria = float(torch.sigmoid(torch.tensor(gananciaEfectiva * logitOracionBase)).item())
                importanciaPalabra = deltaLogit * gananciaEfectiva

                if probDdoBinaria >= 0.5 and importanciaPalabra > self.umbral_atribucion_odio:
                    indicesCensuradosOdio.add(i)
                    palabrasASilenciar.append(pInfo)
                    tokensSalida[i] = f"[D.D.O.: {'*' * max(3, len(palabrasNorm[i]))}]"
                    print(
                        f"   [Transformer+MLP D.D.O.] Palabra: '{palabrasNorm[i]}' | "
                        f"Prob. D.D.O.: {probDdoBinaria:.2f} | "
                        f"Gamma Acústico: {gammaI:.2f} | Atribución: {importanciaPalabra:.3f}"
                    )

        return indicesCensuradosOdio

    def detectarCensurarAltisonantes(self, palabrasNorm, palabrasConTiempo, indicesYaCensurados, tokensSalida, palabrasASilenciar):
        """Paso 3 del censurador que detecta y censura palabras altisonantes en los terminos restantes"""
        for i, palabraNorm in enumerate(palabrasNorm):
            if i in indicesYaCensurados or not palabraNorm:
                continue

            cens = self.censurarTexto(palabraNorm)
            if "*" in cens:
                palabrasASilenciar.append(palabrasConTiempo[i])
                tokensSalida[i] = "*" * len(palabraNorm)

    def procesarCensura(self, textoOriginal, palabrasConTiempo):
        """Ejecuta secuencialmente normalizacion de texto, censura de discurso de odio y censura de altisonantes"""
        #1. Normalizar texto
        textoNormalizado = self.normalizarTexto(textoOriginal)
        if not palabrasConTiempo:
            textoCensurado = self.censurarTexto(textoNormalizado)
            return textoCensurado, []

        palabrasNorm = [self.normalizarTexto(p["word"]) for p in palabrasConTiempo]
        tokensSalida = [w if w else p["word"] for w, p in zip(palabrasNorm, palabrasConTiempo)]
        palabrasASilenciar = []

        #2. Detectar y censurar discurso de odio
        indicesCensuradosOdio = self.detectarCensurarOdio(
            palabrasNorm,
            palabrasConTiempo,
            tokensSalida,
            palabrasASilenciar
        )

        #3. Detectar y censurar palabras altisonantes
        self.detectarCensurarAltisonantes(
            palabrasNorm,
            palabrasConTiempo,
            indicesCensuradosOdio,
            tokensSalida,
            palabrasASilenciar
        )

        #Ordenar los tramos de audio a censurar cronologicamente
        palabrasASilenciar.sort(key=lambda item: float(item["start"]))
        textoFinalCensurado = " ".join(tokensSalida)
        return textoFinalCensurado, palabrasASilenciar

    def evaluar_multimodal_foe(self, textoOriginal, palabrasConTiempo):
        """Alias compatible que invoca la secuencia completa de censura"""
        return self.procesarCensura(textoOriginal, palabrasConTiempo)
