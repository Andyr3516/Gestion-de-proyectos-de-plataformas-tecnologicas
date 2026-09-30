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

#Creamos la función de carga de datos
def load_data():
   #Lectura del archivo csv
   df=pd.read_csv("bp_limpio.csv")
   df1=pd.read_csv("base_leads_limpia.csv")
    #Rellenamos nulos
   df =df.bfill()
   df =df.ffill() 
   df1 =df1.bfill()
   df1 =df1.ffill() 

   Lista = ["Asesor", "Estado", "PDM", "SDC", "Venta", "Inter Gerente"]
   Lista2 = ["Producto", "Temperatura", "Estado", "Asesor"]
   return df, df1, Lista, Lista2
###############################################################################
#Cargo los datos obtenidos de la función "load_data"
df, df1, Lista, Lista2 = load_data()
###############################################################################
#CREACIÓN DEL DASHBOARD
#Generamos las páginas que utilizaremos en el diseño
##############################################################################
#Generamos los encabezados para la barra lateral (sidebar)
st.sidebar.title("GAC")
#Widget 1: Selectbox
#Menu desplegable de opciones de laa páginas seleccionadas
View= st.sidebar.selectbox(label= "Análisis", options= ["Bitácora de piso", 
                                              "Base leads"])
# CONTENIDO DE LA VISTA 1
if View == "Bitácora de piso":
    #EXTRACCIÓN DE CARACTERÍSTICAS
    #Select box
    Variable_Cat= st.sidebar.selectbox(label= "Variables", options= Lista)
    #Obtenemos las frecuencias de las categorías de la variable seleccionada
    Tabla_frecuencias = df[Variable_Cat].value_counts().reset_index()
    #Ajustamos los nombre de las cabeceras de las columnas
    Tabla_frecuencias.columns = ['categorias', 'frecuencia']

    #Generamos los encabezados para el dashboard
    st.title("Bitácora de piso")

   #Generamos el diseño del Layout desead
   # Fila 1
Contenedor_A, Contenedor_B = st.columns(2)
with Contenedor_A: 
      st.write("Gráfico de Área")
      #GRAPH 1: AREA PLOT
      #Obtenemos las frecuencias del estatus de los leads
      Tabla_area = df['Esatus Lead'].str.strip().str.title().value_counts().reset_index()
      Tabla_area.columns = ['estatus', 'frecuencia']
      #Quitamos los registros vacíos
      Tabla_area = Tabla_area[Tabla_area['estatus'] != 'Sin Datos']
      figure1 = px.area(data_frame=Tabla_area, x='estatus', y='frecuencia',
                        title=str('Frecuencia por estatus de lead'),
                        color_discrete_sequence=px.colors.qualitative.Vivid)
      figure1.update_xaxes(automargin=True)
      figure1.update_yaxes(automargin=True)
      figure1.update_layout(height=300)
      st.plotly_chart(figure1, use_container_width=True)

# CONTENIDO DE LA VISTA 2
if View == "Base leads":
    #EXTRACCIÓN DE CARACTERÍSTICAS
    #Select box
    Variable_Cat= st.sidebar.selectbox(label= "Variables", options= Lista2)
    #Obtenemos las frecuencias de las categorías de la variable seleccionada
    Tabla_frecuencias = df[Variable_Cat].value_counts().reset_index()
    #Ajustamos los nombre de las cabeceras de las columnas
    Tabla_frecuencias.columns = ['categorias', 'frecuencia']

    #Generamos los encabezados para el dashboard
    st.title("Producto")

   #Generamos el diseño del Layout desead
   # Fila 1
Contenedor_A1, Contenedor_B1 = st.columns(2)
with Contenedor_A1: 
      st.write("Heatmap")
      #GRAPH 1: HEATMAP
      #Obtenemos las frecuencias de Producto
      Tabla_producto = df1['Producto'].value_counts().reset_index()
      Tabla_producto.columns = ['producto', 'frecuencia']
      #Filtramos los valores que no son relevantes
      Tabla_producto = Tabla_producto[(Tabla_producto['producto'] != 'Sin especificar') & (Tabla_producto['producto'] != 'No envian producto valido')]
      #Pasamos las frecuencias a formato matriz (una fila con cada producto como columna)
      Matriz = Tabla_producto.set_index('producto')[['frecuencia']].T
      figure1 = px.imshow(Matriz, text_auto=True, aspect='auto',
                            title=str('Frecuencia por producto'),
                            color_continuous_scale='Reds')
      figure1.update_xaxes(automargin=True)
      figure1.update_yaxes(automargin=True)
      figure1.update_layout(height=300)
      st.plotly_chart(figure1, use_container_width=True)  

with Contenedor_B1:
        st.write("Grafico de Pastel")
        #GRAPH 2: PIEPLOT
        #Obtenemos las frecuencias del estatus de los leads
        Tabla_estatus = df1['Estado'].value_counts().reset_index()
        Tabla_estatus.columns = ['estatus', 'frecuencia']
        #Despliegue de un pie plot con el estatus de los leads
        figure2 = px.pie(data_frame=Tabla_estatus, names='estatus', 
                  values='frecuencia', title=str('Estatus de los Leads'), color='estatus',
                  color_discrete_sequence=px.colors.qualitative.Vivid)
        figure2.update_layout(height=300)
        st.plotly_chart(figure2, use_container_width=True)
