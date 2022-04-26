#!/usr/bin/env python
# coding: utf-8

# # AUTOMATIC SPEAKER RECOGNITION
# # Study on the effect of face masks on Forensic Speaker Recognition

# ## Paso 00: Instalación de las librerias necesarias

# In[1]:

# ## Paso 01: Importamos los paquetes necesarios

# In[8]:


##-- Api de google sin token y sin costo, solo audios cortos
import pyaudio as pyaudio
#Cargamos la libreria que transforma el audio a texto
import speech_recognition as sr
#Cargamos la libreria que ejecuta el audio con la finalidad de escucharlo antes de analizar
#Importamos el cliente de google
import io
import os
#from google.cloud import speech
#from google.cloud.speech import enums
#from google.cloud.speech import types
## llamamos a las librerias
import os
import pandas as pd
import librosa
import glob 
import matplotlib as mpl
import matplotlib.pyplot as plt
import librosa.display
import numpy as np


#import IPython.display as ipd
# % pylab inline
import os
import pandas as pd
import librosa
import glob 
import librosa.display
import random
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from datetime import datetime
import os
#from pydub import AudioSegment
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.datasets import dump_svmlight_file
from sklearn.preprocessing import StandardScaler
import pickle


np.set_printoptions(suppress=True)


###-Cargamos el archivo de datos de los audios
df=pd.read_excel("corpus.xlsx")
df.head()

##--dirección raiz
data_dir=('.')

##--Creamos una funcion para extraer las mediciones del audio.
def extract_features(files): 
    # Establece el nombre de la ruta a donde están los archivos de audio en mi computadora
    file_name = os.path.join(os.path.abspath(data_dir)+'//corpus//'+str(files.id))
    # Carga el archivo de audio como una serie de tiempo de coma flotante y asigna la frecuencia de muestreo predeterminada
    # Sample rate is set to 22050 by default
    # la serie de tiempo esta almacenada en [X]
    X, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
    # genera Mel-frequency cepstral coefficients (MFCCs) de la serie de tiempo
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=50).T,axis=0)
    # Genera una transformada de Fourier a corto plazo (STFT) para usar en chroma_stft
    #stft = np.abs(librosa.stft(X))
    # Calcula un cromagrama a partir de una forma de onda o espectrograma de potencia.
    #chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    # calcula un espectograma de mel-scaled 
    #mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    # Calcula el contraste espectral
    #contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    # Calcula las características del centroide tonal (tonnetz)
    #tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),sr=sample_rate).T,axis=0)
    # Agregamos también las clases de cada archivo como una etiqueta al final
    label = files.persona
    # Pedimos que nos devuelva todos los indicadores mas el target
    #return mfccs, chroma, mel, contrast, tonnetz,label
    return mfccs, label

##--Obtenemos las caracteristicas de cada audio y medimos el tiempo de demora de la consulta
startTime = datetime.now()
features_label = df.apply(extract_features, axis=1)
print(datetime.now() - startTime)

##--Revisamos los calculos de las caracteristicas del sonido cala fila representa un audio de acuerdo a las filas del df cargado
#print(features_label)

#De la lista generada construimos un array de datos, el cual ingresará a nuestra red neronal
features = []
for i in range(0, len(features_label)):
#    features.append(np.concatenate((features_label[i][0], features_label[i][1],
#               features_label[i][2], features_label[i][3],
#               features_label[i][4]), axis=0))
     features.append(features_label[i][0])

##--El largo del archivo debe ser igual al del df y de los labels
print(len(df))
print(len(features))
labels = df.persona
print(len(labels))

##-.revisamos un array de las personas unicas dentro de labels
np.unique(labels, return_counts=True)


##--separamos los valores X y los valores Y
X = np.array(features)
y = np.array(labels)

# recodificamos el valor Y a un tipo categórico
lb = LabelEncoder()
y = lb.fit_transform(y)

#revisamos los tamaños de los arrays de X e Y: (a:b); a representa total de filas y b total de columnas
print(X.shape)
print(y.shape)

##--de acuerdo al df, partimos nuestros arrays en train, validación y test
X_train = X[:38]
y_train = y[:38]

#X_test = X[36:72] 
#y_test = y[36:72]

X_test = X[38:]
y_test = y[38:]

#X = X[:75]
#y = y[:75]

scaler = StandardScaler()
scaler.fit(X_train)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)#,random_state=109, shuffle = True) # 70% training and 30% test

#X_train =preprocessing.normalize(X_train,norm='l2')
#X_test = preprocessing.normalize(X_test,norm='l2')

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X = scaler.transform(X)

#X = preprocessing.normalize(X,norm='l2')


print(X_train.shape)
print(y_train.shape)

print(X_test.shape)
print(y_test.shape)


file = 'prueba_train.svm'
dump_svmlight_file(X_train, y_train, file, zero_based = False)

file = 'prueba_test.svm'
dump_svmlight_file(X_test, y_test, file, zero_based = False)

file = 'prueba2.svm'
dump_svmlight_file(X, y, file, zero_based = False)
# # Paso 04: Modelamiento y validación

SupportVectorClassModel = SVC(C = 1, gamma=0.0078125, probability=True)
SupportVectorClassModel.fit(X_train,y_train)

filename = 'finalized_model.sav'
pickle.dump(SupportVectorClassModel, open(filename, 'wb'))

y_pred = SupportVectorClassModel.predict(X_test)
print(y_pred, y_test)
y_pred_prob = SupportVectorClassModel.predict_proba(X_test)
print(y_pred_prob)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#print("Precision:",metrics.precision_score(y_test, y_pred, average = None))
#print("Recall:",metrics.recall_score(y_test, y_pred))

