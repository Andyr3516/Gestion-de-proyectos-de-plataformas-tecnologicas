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
   df2=pd.read_csv("base_resumen_limpio.csv")
   df2.columns = df2.columns.str.strip()

    #Rellenamos nulos
   df =df.bfill()
   df =df.ffill() 
   df1 =df1.bfill()
   df1 =df1.ffill() 
   df2 =df2.bfill()
   df2 =df2.ffill() 

   Lista = ["Asesor", "Estado", "PDM", "SDC", "Venta", "Inter Gerente"]
   Lista2 = ["Producto", "Temperatura", "Estado", "Asesor"]
   return df, df1, df2, Lista, Lista2
###############################################################################
#Cargo los datos obtenidos de la función "load_data"
df, df1, df2,Lista, Lista2 = load_data()
###############################################################################
#CREACIÓN DEL DASHBOARD
#Generamos las páginas que utilizaremos en el diseño
##############################################################################
#Generamos los encabezados para la barra lateral (sidebar)
st.sidebar.title("GAC")
#Widget 1: Selectbox
#Menu desplegable de opciones de laa páginas seleccionadas
View= st.sidebar.selectbox(label= "Análisis", options= ["Bitácora de piso", 
                                              "Base leads", "Regresión Lineal"])
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

 ######################################################################################################
# CONTENIDO DE LA VISTA 2
if View == "Regresión Lineal":
    #REGRESIÓN LINEAL MÚLTIPLE
    #Generamos la lista de variables numéricas
    Lista_num = pd.Index(["Leads", "Efectivos", "PDM", "SDC", "Autorizadas", "Formalizado", "Ventas Real"])
    numeric_df = df2[Lista_num]        
    #Select box
    Variable_y= st.sidebar.selectbox(label= "Variable objetivo (Y)", options= Lista_num, 
                                     index=Lista_num.get_loc("Ventas Real"))
    #Multiselect box
    Variables_x= st.sidebar.multiselect(label="Variables independientes del modelo múltiple (X)", 
                                        options= Lista_num, default=["Efectivos", "PDM", "SDC"])

    #Generamos los encabezados para el dashboard
    st.title("Regresión Lineal Múltiple")  

    #Generamos el diseño del Layout deseado
    # Fila 1
    Contenedor_A, Contenedor_B = st.columns(2)
    with Contenedor_A: 
        st.write("Análisis de Correlaciones")
        #GRAPH 5: HEATMAP DE CORRELACIONES
        figure5 = px.imshow(df2[Variables_x + [Variable_y]].corr(), text_auto=True,
                            title= 'Matriz de Correlación')
        st.plotly_chart(figure5)

    with Contenedor_B:
        st.write("Correlación Lineal Múltiple")
        #Se define model como la función de regresión lineal
        from sklearn.linear_model import LinearRegression
        model_M= LinearRegression()
        #Ajustamos el modelo con las variables antes declaradas
        model_M.fit(X=df2[Variables_x], y=df2[Variable_y])
        #Predecimos los valores de la variable objetivo
        y_pred_M= model_M.predict(X=df2[Variables_x])
        #Corroboramos cual es el coeficiente de Determinación de nuestro modelo
        coef_Deter_multiple=model_M.score(X=df2[Variables_x], y=df2[Variable_y])
        #Corroboramos cual es el coeficiente de Correlación de nuestro modelo
        coef_Correl_multiple=np.sqrt(coef_Deter_multiple)
        #Mostramos el coeficiente de correlación múltiple
        st.write(coef_Correl_multiple)

        #GRAPH 6: SCATTERPLOT CON DATOS PREDECIDOS SUPERPUESTOS
        figure6 = px.scatter(data_frame=numeric_df, x=Variables_x, y=Variable_y,
                     title= 'Modelo Lineal Múltiple')
        for x in Variables_x:
            figure6.add_scatter(x=df2[x], y=y_pred_M, mode="markers", name=f"Predicho ({x})")
        st.plotly_chart(figure6)

