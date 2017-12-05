#Document Similarity - Python - Jose Ahirton Lopes (FCamara)

"""
__author__ = "Ahirton Lopes"
__copyright__ = "Copyright 2017, FCamara/Duratex"
__credits__ = ["Ahirton Lopes"]
__license__ = "None"
__version__ = "1.0"
__maintainer__ = "Ahirton Lopes"
__email__ = "ahirtonlopes@gmail.com"
__status__ = "Beta"
"""

'''
Rotina para comparação de quatro frases para com frase alvo. A saída deve ser um vetor para quatro medidas de 0 a 1, sendo os maiores valores
represnetativos das frases consideradas mais semelhantes
'''

import gensim

#Documentos a serem analisados junto a "frase alvo"

raw_documents = ["Uma grande parte do vapor gerado na caldeira, usado para secagem das fibras no secador, e perdida na atmosfera. A ideia e armazenar esse vapor, passar ele por uma serpentina de resfriamento e retornar o condensado para o balao da caldeira, economizando agua desmineralizada.",
                 "Criar um sistema de capitacao nas chamines das caldeiras. O sistema resfria o vapor gerado pelo processo, transformando novamente em agua. Essa agua passa por uma estacao de tratamento e depois e reutilizada no processo produtivo.",
                 "Atualmente a fabrica joga na atmosfera em torno de 10toneladas de agua na forma de vapor. Essa agua vem do processo de secagem da fibraencolada do MDF. Uma adaptacao no processo propiciaria ao processo maior sustentabilidadee economia de agua. Uma ideia simples que tem sua base no reaproveitamentodesse vapor. A inovacao consiste na construcao de uma tubulacao que direcioneesse vapor para um condensador, a agua depois de condensada sera entao injetadanovamente no processo pelo abrandador da caldeira. No processo de condensacaodesse vapor, obteremos agua aquecida, essa agua sera entao injetada na caldeirapara repor as percas do processo. Sendo assim, aumentaremos a eficiencia doprocesso, gastando menos energia na caldeira (uma vez que a agua entrapre-aquecida), alem de diminuir drasticamente o consumo de agua."]

#print("Number of documents:",len(raw_documents))

#Transformação em minúsculas e tokenização

from nltk.tokenize import word_tokenize
gen_docs = [[w.lower() for w in word_tokenize(text, language='portuguese')] 
            for text in raw_documents]
#print(gen_docs)
            
#Criação de dicionário a partir das palavras tokenizadas

dictionary = gensim.corpora.Dictionary(gen_docs)
#print(dictionary[5])
#print("Number of words in dictionary:",len(dictionary))
for i in range(len(dictionary)):
    #print(i, dictionary[i])

#A partir do dicionário é utilizada uma técnica de BoW (Bag of Words) - Cada frase 
#é representada pela coleção de suas palavras não mais sendo considerada gramática ou ordem
    
    corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
#print(corpus)
    
#Aplica-se TF-IDF (frequência do termo–inverso da frequência nos documentos) para o modelo gerado pelo BoW
#TF - O peso de um termo que ocorre em um documento é diretamente proporcional à sua frequência. 
#IDF - A especificidade de um termo pode ser quantificada por uma função inversa do número de documentos em que ele ocorre.

tf_idf = gensim.models.TfidfModel(corpus)
#print(tf_idf)
s = 0
for i in corpus:
    s += len(i)
#print(s)

#Medida de similaridade via gensim

sims = gensim.similarities.Similarity('/Users/FCAMARA0834/Desktop/Projetos',tf_idf[corpus],
                                      num_features=len(dictionary))
#print(sims)
#print(type(sims))

#Frase alvo

query_doc = [w.lower() for w in word_tokenize("Utilizar o vapor dagua gerado pelas ciclones da prensa para condensar e retornar a linha de producao")]
#print(query_doc)
query_doc_bow = dictionary.doc2bow(query_doc)
#print(query_doc_bow)
query_doc_tf_idf = tf_idf[query_doc_bow]
#print(query_doc_tf_idf)

print sims[query_doc_tf_idf]
