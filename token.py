import pandas as pd
import gensim
import spacy
import re
# python -m spacy download es
spacy.load('es')
from spacy.lang.es import Spanish
import unicodedata
from gensim.utils import simple_preprocess
import nltk
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
parser = Spanish()
nltk.download('stopwords')
stop_word = set(nltk.corpus.stopwords.words('spanish'))
import warnings
warnings.filterwarnings('ignore')
import os

PATH = os.path.dirname(os.path.abspath(__file__))

data_or = pd.read_excel(PATH+'/data/consolidado_2018_2019_clean.xlsx')

data = data_or.cuento
def without_spaces(value) -> str:
    """
    eliminar todos los espacios y saltos de linea

    :param value: cadena de texto
    :return: cadena transformada
    """
    if value:
        value = str(value)
        return re.sub(r'\s+', '', value)
    else:
        return value


def sent_to_words(sentences):
    for sentence in sentences:
        yield(simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations



data_words = list(sent_to_words(data))

def eliminar_acentos(token):
    """
    Función para eliminar acentos a cada mensaje
    """
    return ''.join(c for c in unicodedata.normalize('NFD', token)
    if unicodedata.category(c) != 'Mn')

terminos_stops_word = [ 'seria', 'aquello', 'hacen', 'puedan', 'pueda', 'hacen', 'ninguno', 'mismo', 'falecnias', 'ahora'
            'hacia', 'siendo', 'llena', 'toda', 'esta', 'tambien', 'estdio', 'tener', 'ahora', 'aunque', 'mientras', 'do',
						'podria', 'trave', 'asi', 'dar', 'ser', 'tan', 'si', 'ma', 'nan', 'toda', 'dure', 'permita',
						'demas', 'menos', 'cada', 'creo', 'halla', 'tal', 'voy', 'seguir',
						'mientra', 'llevar', 'puz', 'situacion','dema', 'despu', 'estamo', 'parte',
						'mejor', 'cualquier', 'pu', 'puedo', 'alguna', 'va', 'cosa', 'ello', 'poder', 'adema',
						'me', 'bien', 'etc', 'tenemo', 'cuenta', 'tipo', 'nuevo','caso', 'podamo', 'den', 'po', 'ano',
						'luego', 'vamos', 'misma', 'meno', 'estan', 'aun', 'posible', 'puede','manera', 'medida',
						'algun', 'ma', 'necesito', 'ninguna', 'covid', 'dia', 'nosotro', 'tema', 'ningun',
						'nueva', 'somo', 'parar', 'urraaaaaa', 'piiiiiiii', 'piiiii']

def remove_stopwords(texts):
    lista = [[eliminar_acentos(without_spaces(word)) for word in doc] for doc in texts]
    lista =[[without_spaces(word)  for word in doc if word not in stop_word] for doc in lista]
    return [[word for word in doc if word not in terminos_stops_word] for doc in lista]

data_words_nostops = remove_stopwords(data_words)

# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words_nostops, min_count=2, threshold=20) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words_nostops], threshold=20)

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example


def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

data_words_bigrams = make_bigrams(data_words_nostops)

data_words_trigrams = make_trigrams(data_words_nostops)

print(data_words_bigrams[:1])

def lematizar(word):
    """
    función para lematizar tokens
    """

    lista=list()
    for item in word:
        lista.append(WordNetLemmatizer().lemmatize(item))

    return lista

data_lemma_bigram = [lematizar(token) for token in data_words_bigrams]
data_lemma_trigram = [lematizar(token) for token in data_words_trigrams]
data_lemma = [lematizar(token) for token in data_words_nostops]


# Create Corpus
texts = data_lemma
data_or['token'] = data_lemma
data_or['token_bigram'] = data_lemma_bigram
data_or['token_trigram'] = data_lemma_trigram

data_or.to_excel(PATH+'/data/data_consolidado_tokens.xlsx', index=False)