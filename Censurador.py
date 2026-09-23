from spanlp.palabrota import Palabrota #Modulo para censurar
from spanlp.domain.countries import Country #Dominios de paises
from spanlp.domain.strategies import Preprocessing, TextToLower, RemoveAccents #Preprocesamiento de texto
from spanlp.domain.strategies import LevenshteinDistance #Metrica de distancia

class Censurador():
    def __init__(self, pais=None):
        """Inicializa el censurador de groserias con un pais opcional"""
        cosine = LevenshteinDistance(1) #Usa distancia de Levenshtein para palabras parecidas. Ej. Tonto y Tonoto
        paises = {'MX': Country.MEXICO, 'CO': Country.COLOMBIA, 'AR': Country.ARGENTINA} #Interfaz para elegir el pais

        #Asigna todos por defecto si no hay pais
        if pais:
            self.palabrota = Palabrota(censor_char="*", countries=[paises[pais]], distance_metric=cosine)
        else:
            self.palabrota = Palabrota(censor_char="*", distance_metric=cosine)

    def censurarTexto(self, texto):
        """Censura el texto proporcionado reemplazando malas palabras con asteriscos"""
        return self.palabrota.censor(texto)

    def preprocesarTexto(self, texto):
        """Genera una secuencia de preprocesamiento para minusculas y quitar acentos"""
        estrategias = [TextToLower(), RemoveAccents()]
    
        #Aplica la secuencia de preprocesamiento
        preprocesado = Preprocessing().clean(data=texto, clean_strategies=estrategias)
    
        return preprocesado

    def core(self, texto):
        """Realiza el proceso de normalizacion y censura del texto"""
        #Preprocesa el texto
        preprocesado = self.preprocesarTexto(texto)
    
        #Censura el texto
        censurado = self.censurarTexto(preprocesado)
    
        return preprocesado, censurado