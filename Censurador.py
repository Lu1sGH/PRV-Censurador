from spanlp.palabrota import Palabrota
from spanlp.domain.countries import Country
from spanlp.domain.strategies import Preprocessing, TextToLower, RemoveAccents
from spanlp.domain.strategies import LevenshteinDistance

class Censurador():
    def __init__(self, pais=None):
        cosine = LevenshteinDistance(1) #Usa Distancia Levenshtein para palabras parecidas, ej. "tonoto" y "tonto"
        paises = {'MX': Country.MEXICO, 'CO': Country.COLOMBIA, 'AR': Country.ARGENTINA} #Una interfaz más fácil para elegir el país (principales)

        #Si no se asigna un país, se consideran todos por defectos
        if pais:
            self.palabrota = Palabrota(censor_char="*", countries=[paises[pais]], distance_metric=cosine)
        else:
            self.palabrota = Palabrota(censor_char="*", distance_metric=cosine)

    def censurar(self, texto):
        """Censura el texto proporcionado."""
        return self.palabrota.censor(texto)

    def preprocesar(self, texto):
        """Genera un pipeline de preprocesamiento."""
        estrategias = [TextToLower(), RemoveAccents()]
    
        #Aplica el pipeline
        preprocesado = Preprocessing().clean(data=texto, clean_strategies=estrategias)
    
        return preprocesado

    def core(self, texto):
        """Realiza el proceso de normalización y censura del texto."""
        #Preprocesa
        preprocesado = self.preprocesar(texto)
    
        #Censura
        censurado = self.censurar(preprocesado)
    
        return preprocesado, censurado