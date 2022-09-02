import re
import pandas as pd 
import numpy as np
from time import time  # Para cronometrar las operaciones
from collections import defaultdict  # Para la frecuencia de palabras
import spacy
from typing import NoReturn
import gensim
from tensorboard import summary
import xlrd 
import nltk

import logging  # Configurar los registros para monitorear gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)



#Cargamos el dataset
df_temp = pd.read_excel('dataset.xlsx', sheet_name='Hoja1') # col_names = True

#Sustituimos los valores de las estrellas por -1, 0 y 1 dependiendo del sentimientos 
df_temp.loc[df_temp.Estrellas < 3,'Estrellas']= -1
df_temp.loc[df_temp.Estrellas == 3,'Estrellas']=  0
df_temp.loc[df_temp.Estrellas > 3,'Estrellas']= 1

#Ahora eliminamos columnas
df = df_temp.drop(['Identificador', 'Usuario', 'Producto'], axis=1)
df.shape
df.head()   
df.isnull().sum() #Para ver cuantos datos faltantes hay de cada una de las columnas


clase, frecuencia = np.unique(list( df_temp["Estrellas"] ),return_counts = True) 


#Se inicia la limpieza del texto
nlp = spacy.load('es_core_news_md', disable=['ner', 'parser']) # Se deshabilita NER para mayor velocidad


def cleaning(doc):
    # Lemmatizes and removes stopwords
    # doc needs to be a spacy Doc object
    txt = [token.lemma_ for token in doc if not token.is_stop]
    # Word2Vec uses context words to learn the vector representation of a target word,
    # if a sentence is only one or two words long,
    # the benefit for the training is very small
    if len(txt) > 2:
        return ' '.join(txt)

brief_cleaning = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in df['Reseña'])

#Se crea un objeto de la clase time() para contabilizar el tiempo que llevará a acabo la limpieza
t = time()
txt = [cleaning(doc) for doc in nlp.pipe(brief_cleaning, batch_size=50, n_process=-1)]
print('Tiempo de limpieza: {} mins'.format(round((time() - t) / 60, 2)))



#Ponemos los resultados en un dataframe para eliminar valores faltantes y duplicados.
df_clean = pd.DataFrame({'resena': txt, 'polaridad': df["Estrellas"]})
df_clean = df_clean.dropna().drop_duplicates()
df_clean.shape
df_clean.head()
df_clean.columns.tolist() 

#Contamos cuantos elementos de cada clase tenemos
freq = df_clean['polaridad'].value_counts() 
print(freq)


#Ahora usamos Gensim para detectar frases comunes (bigramas)
from gensim.models.phrases import Phrases, Phraser


#Creamos una lista de listas
sent = [row.split() for row in df_clean['resena']]

#Creamos las frases relevantes de la lista de oraciones
#Phrases() toma una lista de listas de palabras como entrada
#Con Phrases entrenamos un detector de bigramas
phrases = Phrases(sent, min_count=30, progress_per=10000) 

#El objetivo de Phraser() es reducir el consumo de memoria de Phrases(), 
#descartando el estado del modelo que no es estrictamente necesario para la tarea de detección de bigramas
bigram = Phraser(phrases)

#Transformamos el corpus "base" a los bigramas detectados:
sentences = bigram[sent]


# Encontramos las palabras más frecuentes
word_freq = defaultdict(int)
for sent in sentences:
    for i in sent:
        word_freq[i] += 1
len(word_freq)
#Ordenamos las palabras más frecuentes por orden
sorted(word_freq, key=word_freq.get, reverse=True)[:10]



#Ahora entrenamos el modelo
#Implementación de Gensim Word2Vec
import multiprocessing
from gensim.models import Word2Vec

cores = multiprocessing.cpu_count()

tamano_max_vect = 28 #Definimos el número de filas de la matriz (cantidad de palabras vectorizadas)
tam_vector_w2v = 28 #Definimos el número de columnas de la matriz (dimensión del vector que representa a cada palabra)

w2v_model = Word2Vec(min_count=1,
                     window=2,
                     vector_size=tam_vector_w2v,
                     sample=6e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=20,
                     workers=cores-1)

#Construimos la tabla de vocabulario
t = time()
w2v_model.build_vocab(sentences, progress_per=10000)
print('Tiempo para construir vocabulario: {} mins'.format(round((time() - t) / 60, 2)))

#Ahora entrenamos el modelo
t = time()
w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
print('Tiempo para entrenar el modelo: {} mins'.format(round((time() - t) / 60, 2)))


#Enseguida creamos las matrices que representarán a cada una de las reseñas
oraciones_vectorizadas = []

for i in range ( len(sentences) ):
    #Convertimos las oraciones a listas de vectores
    oracion = [w2v_model.wv.get_vector(palabra) for palabra in sentences[i]]
    oracion = np.array(oracion)
    #Ahora mapeamos los valores en la matriz a un rango de 0-255
    oracion2 = ((oracion-np.amin(oracion))/(np.amax(oracion)-np.amin(oracion)))*255

    #Redimensionamos las matrices para tenerlas de una dimensión uniforme
    oracion2.resize(tamano_max_vect, tam_vector_w2v, refcheck=False)

    #Agregamos las matrices redimensionadas a una lista
    oraciones_vectorizadas.append(oracion2)



#Creamos una lista para guardar las oraciones vectorizadas redimensionadas (como si fueran imagenes)
oraciones_como_imagenes = []
oraciones_como_imagenes = [palabra.reshape(1,28, 28, 1) for palabra in oraciones_vectorizadas]

#Convertimos las listas a numpy
X = np.array( oraciones_vectorizadas ).astype(int) 
y = np.array( list( df_clean["polaridad"] ) )

#Contamos cuantos elementos de cada clase hay en el conjunto de datos
clase, frecuencia = np.unique(y,return_counts = True) 

#Realizamos la partición de los datos
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.35, random_state = 0)

#codificamos en caliente al target 
from keras.utils import to_categorical
y_train = to_categorical(y_train,3)
y_test = to_categorical(y_test,3)


from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import BatchNormalization

#Definimos el modelo de la red neuronal convolucional
#, kernel_initializer='he_uniform'
def define_model():
	model = Sequential()
	model.add(Conv2D(filters = 32 , kernel_size = (3, 3), activation='relu', input_shape=(28, 28, 1)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
    model.add(Dense(units = 400, kernel_initializer = "uniform",  activation = "relu")) #400
    model.add(Dense(units = 50, kernel_initializer = "uniform",  activation = "relu")) #50
	model.add(Dense(3, activation='softmax'))
	# compile model
	opt = SGD(learning_rate=0.01, momentum=0.9)
	model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

	return model

#Creamos el modelo, entrenamos y lo evaluamos con los datos de testing

model = define_model()
print(model.summary())

clasificador = model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
#accuracy = model.evaluate(X_test, y_test, verbose=0)
y_pred = model.predict(X_test)

y_pred = y_pred.argmax(axis=1)
y_test = y_test.argmax(axis=1)



# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm_porcentaje = confusion_matrix(y_test, y_pred, normalize='true')*100

#Obtenemos las métricas
accuracy = 100 * sum(np.diagonal(cm))/np.sum(cm)
accuracy

#51.960

#Sensibilidad
sensibilidad_1 = cm_porcentaje[0][0] 
sensibilidad_2 = cm_porcentaje[1][1] 
sensibilidad_3 = cm_porcentaje[2][2] 

sensibilidad = (sensibilidad_1 + sensibilidad_2 + sensibilidad_3)/3
sensibilidad


especificidad_1 = (cm[1][1] + cm[2][2]) / sum( sum( cm[1:3][:] ) )*100
especificidad_2 = ( (cm[0][0] + cm[2][2]) / ( sum( cm[0][:] ) + sum( cm[2][:] ) ) )*100
especificidad_3 = (cm[0][0] + cm[1][1]) / sum( sum( cm[0:2][:] ) )*100

especificidad = (especificidad_1 + especificidad_2 + especificidad_3)/3
especificidad


#Medida de la presición del algoritmo para la clase positiva
Precision_1 = ( cm[0][0] / ( cm[0][0] + cm[1][0] + cm[2][0]) )*100
Precision_2 = ( cm[1][1] / ( cm[0][1] + cm[1][1] + cm[2][1]) )*100
Precision_3 = ( cm[2][2] / ( cm[0][2] + cm[1][2] + cm[2][2]) )*100

Precision = (Precision_1 + Precision_2 + Precision_3)/3                    
Precision

#Compromiso entre la presición y al completitud
F1_Score_1 = (2 * Precision_1*sensibilidad_1) / (Precision_1 + sensibilidad_1)
F1_Score_2 = (2 * Precision_2*sensibilidad_2) / (Precision_2 + sensibilidad_2)
F1_Score_3 = (2 * Precision_3*sensibilidad_3) / (Precision_3 + sensibilidad_3)

F1_Score = (F1_Score_1 + F1_Score_2 + F1_Score_3) / 3
F1_Score