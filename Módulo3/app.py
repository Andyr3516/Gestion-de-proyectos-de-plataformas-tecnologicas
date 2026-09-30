#Crea el archivo de la APP en la interprete principal de python

#Importa librerías
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np

#A partir de aquí copié lo del profe
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats
######################################################
#Definimos la instancia
@st.cache_resource
######################################################
#Creamos la función de carga de datos
def load_data():
   #Lectura del archivo csv
   df=pd.read_csv("Titanic.csv")
    #Rellenamos nulos
   df =df.bfill()
   df =df.ffill()  
   Lista=['Survived', 'Pclass', 'Sex', 'Age']
   return df, Lista
###############################################################################
#Cargo los datos obtenidos de la función "load_data"
df, Lista = load_data()
###############################################################################
#CREACIÓN DEL DASHBOARD
#Generamos las páginas que utilizaremos en el diseño
##############################################################################
#Generamos los encabezados para la barra lateral (sidebar)
st.sidebar.title("TITANIC")
#Widget 1: Selectbox
#Menu desplegable de opciones de laa páginas seleccionadas
View= st.sidebar.selectbox(label= "Tipo de Análisis", options= ["Extracción de Características", 
                                              "Regresión Lineal", "Regresión No Lineal",
                                              "Regresión Logística"])

# CONTENIDO DE LA VISTA 1
if View == "Extracción de Características":
    #EXTRACCIÓN DE CARACTERÍSTICAS
    #Select box
    Variable_Cat= st.sidebar.selectbox(label= "Variables", options= Lista)
    #Obtenemos las frecuencias de las categorías de la variable seleccionada
    Tabla_frecuencias = df[Variable_Cat].value_counts().reset_index()
    #Ajustamos los nombre de las cabeceras de las columnas
    Tabla_frecuencias.columns = ['categorias', 'frecuencia']

    #Generamos los encabezados para el dashboard
    st.title("Extracción de Características")

    #Generamos el diseño del Layout deseado
    # Fila 1
    Contenedor_A, Contenedor_B = st.columns(2)
    with Contenedor_A: 
        st.write("Grafico de Barras")
        #GRAPH 1: BARPLOT
        #Despliegue de un bar plot, definiendo las variables "X categorias" y "Y numéricas" 
        figure1 = px.bar(data_frame=Tabla_frecuencias, x='categorias', 
                       y= 'frecuencia', title= str('Frecuencia por categoría'),color='frecuencia',
                       color_continuous_scale='Reds')
        figure1.update_xaxes(automargin=True)
        figure1.update_yaxes(automargin=True)
        figure1.update_layout(height=300)
        st.plotly_chart(figure1, use_container_width=True)

    with Contenedor_B:
        st.write("Grafico de Pastel")
        #GRAPH 2: PIEPLOT
        #Despliegue de un pie plot, definiendo las variables "X categorias" y "Y numéricas" 
        figure2 = px.pie(data_frame=Tabla_frecuencias, names='categorias', 
                  values= 'frecuencia', title= str('Frecuencia por categoría'),color='frecuencia',
                       color_discrete_sequence=px.colors.qualitative.Vivid)
        figure2.update_layout(height=300)
        st.plotly_chart(figure2, use_container_width=True)

    # Fila 2
    Contenedor_C, Contenedor_D = st.columns(2)
    with Contenedor_C:
        st.write("Grafico de anillo o dona")
        #GRAPH 3: DONUT PLOT
        #Despliegue de un line plot, definiendo las variables "X categorias" y "Y numéricas" 
        figure3 = px.pie(data_frame=Tabla_frecuencias, names='categorias', 
                  values= 'frecuencia', hole=0.4, title= str('Frecuencia por categoría'),color='frecuencia',
                       color_discrete_sequence=px.colors.qualitative.Bold)
        figure3.update_layout(height=300)
        st.plotly_chart(figure3, use_container_width=True)

    with Contenedor_D:  
        st.write("Grafico de area")
        #GRAPH 4: AREA PLOT
        #Despliegue de un area plot, definiendo las variables "X categorias" y "Y numéricas" 
        figure4 = px.area(data_frame=Tabla_frecuencias, x='categorias', 
                  y= 'frecuencia', title= str('Frecuencia por categoría'), color='frecuencia',
                       color_discrete_sequence=px.colors.qualitative.Dark24)
        figure4.update_layout(height=300)
        st.plotly_chart(figure4, use_container_width=True)
 ############################################################################
 #Otras_Paletas
#px.colors.qualitative.Vivid
#px.colors.qualitative.Bold
#px.colors.qualitative.Pastel
#px.colors.qualitative.Set1
#px.colors.qualitative.Set2
#px.colors.qualitative.Safe
#px.colors.qualitative.Dark24
 ###################################################################################
