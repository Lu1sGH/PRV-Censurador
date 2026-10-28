import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from spanlp.palabrota import Palabrota  # Modulo para censurar
from spanlp.domain.countries import Country  # Dominios de paises
from spanlp.domain.strategies import Preprocessing, TextToLower, RemoveAccents  # Preprocesamiento de texto
from spanlp.domain.strategies import LevenshteinDistance  # Metrica de distancia


class TransformerMLPFoE(nn.Module):
    """
    Arquitectura de Codificador Transformer Compacto + Perceptron Multicapa (MLP) no lineal
    con regularizacion de potencial Student-t (Field of Experts) para clasificacion binaria.
    """
    def __init__(self, nombre_modelo="dccuchile/distilbert-base-spanish-uncased", hidden_dim=128):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(nombre_modelo)
        embed_dim = self.encoder.config.hidden_size

        # Congelar capas inferiores y dejar entrenables solo las ultimas 2 capas del Transformer
        # para evitar sobreajuste y cumplir la restriccion de < 25 min en 6 GB de VRAM
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

        # Clasificador MLP no lineal: d_enc -> hidden_dim (128) -> 1 neurona de salida
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, 1)
        )

        # Parametro de potencial Student-t alpha > 0 del Field of Experts
        self.alpha = nn.Parameter(torch.ones((1, 1), dtype=torch.float32))

    def mean_pooling_enmascarado(self, last_hidden_state, attention_mask):
        """Calcula el promedio contextual ponderado por la mascara de atencion y normaliza en L2"""
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        suma_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
        suma_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        u = suma_embeddings / suma_mask
        return F.normalize(u, p=2, dim=-1)

    def forward(self, input_ids, attention_mask):
        """Paso hacia adelante devolviendo el vector de logits binarios z de dimension (batch,)"""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        u_norm = self.mean_pooling_enmascarado(outputs.last_hidden_state, attention_mask)
        logits = self.mlp(u_norm)
        return logits.squeeze(-1)

    def energia_potencial_student_t(self, z):
        """Calcula el potencial E_FoE(z) = alpha * log(1 + 0.5 * z^2)"""
        alpha_pos = F.softplus(self.alpha.squeeze())
        return torch.mean(alpha_pos * torch.log1p(0.5 * (z.float() ** 2)))


class Censurador():
    def __init__(
        self,
        pais=None,
        ruta_pesos_foe="pesos_foe_binario.pt",
        modelo_transformer="dccuchile/distilbert-base-spanish-uncased"
    ):
        """Inicializa el censurador de groserias con un pais opcional y el modelo Transformer + MLP}"""
        cosine = LevenshteinDistance(1)  # Usa distancia de Levenshtein para palabras parecidas
        paises = {'MX': Country.MEXICO, 'CO': Country.COLOMBIA, 'AR': Country.ARGENTINA}  # Interfaz para elegir el pais

        # Asigna todos por defecto si no hay pais
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
        """Carga los pesos entrenados del Transformer+MLP desde disco o calibra el cabezal para uso inmediato"""
        if os.path.exists(self.ruta_pesos_foe):
            try:
                estado = torch.load(self.ruta_pesos_foe, map_location=self.device, weights_only=True)
                self.modelo_foe.load_state_dict(estado, strict=False)
                self.modelo_foe.eval()
                print(f"--> Pesos Transformer+MLP FoE cargados exitosamente desde '{self.ruta_pesos_foe}'.")
                return
            except Exception as e:
                print(f"--> Aviso al cargar '{self.ruta_pesos_foe}': {e}. Calibrando cabezal base.")

        # Calibracion rapida de respaldo mediante descenso breve sobre prototipos contrastivos
        prototipos_positivos = [
            "esa gente son una plaga inferior y deberian eliminarlos a todos",
            "los inmigrantes e indigenas son escoria subhumana que no merece derechos",
            "hay que matar y exterminar a esa minoria porque son basura asquerosa"
        ]
        prototipos_negativos = [
            "hola buenos dias estamos probando el sistema de grabacion de audio",
            "la reunion de trabajo sera manana por la tarde en la oficina",
            "el clima en la ciudad es agradable y la conversacion es completamente normal"
        ]

        textos_init = prototipos_positivos + prototipos_negativos
        etiquetas_init = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], device=self.device)

        self.modelo_foe.mlp.train()
        opt_init = torch.optim.AdamW(self.modelo_foe.mlp.parameters(), lr=1e-2)
        bce_init = nn.BCEWithLogitsLoss()

        enc = self.tokenizer(
            textos_init,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)

        for _ in range(25):
            opt_init.zero_grad()
            z = self.modelo_foe(enc["input_ids"], enc["attention_mask"])
            loss = bce_init(z, etiquetas_init)
            loss.backward()
            opt_init.step()

        self.modelo_foe.eval()

    def entrenar_foe_offline(self, lista_textos_292k, vector_binario_y, epocas=2, batch_size=128, lr=3e-4):
        """
        Entrena las ultimas 2 capas del Transformer y el cabezal MLP no lineal usando Precision Mixta (FP16)
        sobre las 292,000 entradas y la columna binaria unica en ~8 a 11 minutos en 6 GB de VRAM.
        """
        n_muestras = len(lista_textos_292k)
        print(f"--> [1/2] Tokenizando {n_muestras:,} textos con tokenizador Rust paralelo...")
        tokens_corpus = self.tokenizer(
            lista_textos_292k,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids_all = tokens_corpus["input_ids"]
        attention_mask_all = tokens_corpus["attention_mask"]
        y_all = torch.as_tensor(vector_binario_y, dtype=torch.float32).view(-1)

        # Compensacion de desbalance de clases en la columna binaria
        num_pos = torch.sum(y_all == 1.0).clamp(min=1.0)
        num_neg = torch.sum(y_all == 0.0).clamp(min=1.0)
        peso_pos = (num_neg / num_pos).to(self.device)
        criterio_bce = nn.BCEWithLogitsLoss(pos_weight=peso_pos)

        # Tasas de aprendizaje diferenciadas: menor para las 2 capas del Transformer, mayor para el MLP
        parametros_optim = [
            {"params": [p for n, p in self.modelo_foe.encoder.named_parameters() if p.requires_grad], "lr": 5e-5},
            {"params": self.modelo_foe.mlp.parameters(), "lr": lr},
            {"params": [self.modelo_foe.alpha], "lr": 1e-3}
        ]
        optimizador = torch.optim.AdamW(parametros_optim, weight_decay=1e-2)
        usar_amp = self.device == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=usar_amp)

        self.modelo_foe.train()
        print(f"--> [2/2] Entrenando Transformer + MLP en {self.device.upper()} (Batch={batch_size}, AMP={usar_amp})...")

        for epoca in range(epocas):
            perm = torch.randperm(n_muestras)
            loss_acumulada = 0.0
            pasos = 0

            for i in range(0, n_muestras, batch_size):
                idx = perm[i:i + batch_size]
                b_ids = input_ids_all[idx].to(self.device, non_blocking=True)
                b_mask = attention_mask_all[idx].to(self.device, non_blocking=True)
                b_y = y_all[idx].to(self.device, non_blocking=True)

                optimizador.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=usar_amp):
                    logits_z = self.modelo_foe(b_ids, b_mask)
                    loss_bce = criterio_bce(logits_z, b_y)
                    loss_foe = self.modelo_foe.energia_potencial_student_t(logits_z)
                    loss_total = loss_bce + 0.001 * loss_foe

                scaler.scale(loss_total).backward()
                scaler.step(optimizador)
                scaler.update()

                loss_acumulada += float(loss_total.item())
                pasos += 1

            print(f"    Epoca {epoca + 1}/{epocas} finalizada | Perdida media: {loss_acumulada / max(1, pasos):.4f}")

        self.modelo_foe.eval()
        torch.save(self.modelo_foe.state_dict(), self.ruta_pesos_foe)
        print(f"--> Pesos de Transformer + MLP guardados en '{self.ruta_pesos_foe}'.")

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
        Ejecuta la censura en dos niveles usando el clasificador binario Transformer + MLP:
          1. Nivel 1 (Lexico): Distancia de Levenshtein sobre cada palabra.
          2. Nivel 2 (Multimodal Binario): Oclusion en lote en GPU con Transformer + MLP en FP16
             modulada por el factor acustico gamma(w_i) del espectrograma y la forma de onda.
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

        # --- NIVEL 2: CLASIFICACIÓN BINARIA MULTIMODAL (TRANSFORMER + MLP + OCLUSIÓN) ---
        k = len(palabrasConTiempo)
        palabras_limpias = [p["word"] for p in palabrasConTiempo]
        oracion_completa = " ".join(palabras_limpias)

        # Fila 0: Oracion completa; Fila i + 1: Oracion omitiendo la palabra i-esima
        lote_textos_ocluidos = [oracion_completa]
        for i in range(k):
            variante = " ".join(palabras_limpias[:i] + palabras_limpias[i + 1:])
            lote_textos_ocluidos.append(variante if variante.strip() else " ")

        usar_amp = self.device == "cuda"
        with torch.no_grad():
            lote_tokens = self.tokenizer(
                lote_textos_ocluidos,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)

            with torch.amp.autocast("cuda", enabled=usar_amp):
                logits_binarios = self.modelo_foe(
                    lote_tokens["input_ids"],
                    lote_tokens["attention_mask"]
                ).float()

            logit_oracion_base = float(logits_binarios[0].item())

            for i in range(k):
                if i in indices_censurados_lex:
                    continue

                p_info = palabrasConTiempo[i]
                gamma_i = float(p_info.get("gamma", 1.0))
                prob_asr = float(p_info.get("probability", 1.0))

                delta_logit = logit_oracion_base - float(logits_binarios[i + 1].item())
                ganancia_efectiva = gamma_i * max(0.5, prob_asr)
                prob_ddo_binaria = float(torch.sigmoid(torch.tensor(ganancia_efectiva * logit_oracion_base)).item())
                importancia_palabra = delta_logit * ganancia_efectiva

                if prob_ddo_binaria >= 0.5 and importancia_palabra > self.umbral_atribucion_odio:
                    palabrasASilenciar.append(p_info)
                    tokens_salida[i] = f"[D.D.O.: {'*' * max(3, len(p_info['word']))}]"
                    print(
                        f"   [Transformer+MLP D.D.O.] Palabra: '{p_info['word']}' | "
                        f"Prob. D.D.O.: {prob_ddo_binaria:.2f} | "
                        f"Gamma Acústico: {gamma_i:.2f} | Atribución: {importancia_palabra:.3f}"
                    )

        texto_final_censurado = " ".join(tokens_salida)
        return texto_final_censurado, palabrasASilenciar