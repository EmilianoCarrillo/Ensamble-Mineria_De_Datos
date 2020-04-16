# PCA para el análisis, visualización de datos, y entrenamiento de algoritmos de clasificación

## Tabla de Contenidos
1. Introducción
2. Importación de Datos
3. Preprocesar Datos
4. PCA <br/>
    4.1 Aplicación del PCA <br/>
    4.2 Análisis de resultados <br/>
    4.3 Visualización del PCA 2D <br/>
5. Detección de Outliers
6. Clasificación <br/>
    6.1 Clasificar <br/>
    6.2 Predicción y Análisis <br/>
    
    
## 1. Introducción
    
PCA es un algoritmo de reducción de dimensionalidad no supervisado. En este ejemplo lo emplearemos para satisfacer 3 objetivos:

**1. Hacer un análisis** <br/>
    Analizaremos la cantidad de componentes principales que son necesarios para resumir nuestros datos de una manera que no haya tanta pérdida de información. Este objetivo se establece únicamente como fase de experimentación.
    
**2. Facilitar la visualización de nuestros datos** <br/>
Ocuparemos 2 componentes principales para poder graficar nuestros datos multidimensionales en una gráfica de dispersión. Así podremos entender mejor nuestra data y podemos detectar posibles outliers.
    
**3. Como paso de preprocesamiento para la clasificación de elementos** <br/>
Por último ocuparemos PCA para resumir nuestros datos y eficientizar el entrenamiento de nuestro clasificador sin perder efectividad de clasificación.
        

## 2. Importación de Datos
El primer paso es importar los datos que ocuparemos para el análisis. El archivo de entrada debe ser un archivo de texto plano con el formato siguiente:
```
No. Elementos
No. Atributos
No. Clases
atrib_0, atrib_1, ..., atrib_n, clase
atrib_0, atrib_1, ..., atrib_n, clase
... ... ...
atrib_0, atrib_1, ..., atrib_n, clase
```

### Preprocesar archivo
Primero preprocesamos el archivo para obtener los metadatos de No. de elementos, atributos y clases que éste contiene en el encabezado y así construir nuestro dataset.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
%matplotlib inline

nombre_archivo = "data"
```


```python
try:
    with open( nombre_archivo + ".txt", "r") as archivo:
        nElem = int(archivo.readline())
        nAtrib = int(archivo.readline())
        nClases = int(archivo.readline())
        
        atributos = []
        for i in range(0, nAtrib):
            atributos.append("atrib_" + str(i+1))
        
        atributos.append("clase")
        data = pd.read_csv(archivo, delimiter=',', header=None)
        data.columns = atributos
    
except FileNotFoundError:
    print( "ERROR: El archivo " + nombre_archivo + " no fue encontrado");
finally:
    archivo.close();

dataset = data;
```

Podemos obtener un pequeño vistazo de cómo se ve nuestro dataset hasta ahora.


```python
x_fin = dataset.drop('clase', 1)
y_fin = dataset['clase']
nComponentes = "Sin Componentes"
```


```python
dataset.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>atrib_1</th>
      <th>atrib_2</th>
      <th>atrib_3</th>
      <th>atrib_4</th>
      <th>atrib_5</th>
      <th>atrib_6</th>
      <th>atrib_7</th>
      <th>atrib_8</th>
      <th>atrib_9</th>
      <th>atrib_10</th>
      <th>atrib_11</th>
      <th>atrib_12</th>
      <th>atrib_13</th>
      <th>atrib_14</th>
      <th>atrib_15</th>
      <th>atrib_16</th>
      <th>atrib_17</th>
      <th>atrib_18</th>
      <th>atrib_19</th>
      <th>clase</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>140</td>
      <td>125</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.277778</td>
      <td>0.062963</td>
      <td>0.666667</td>
      <td>0.311111</td>
      <td>6.185185</td>
      <td>7.333334</td>
      <td>7.666666</td>
      <td>3.555556</td>
      <td>3.444444</td>
      <td>4.444445</td>
      <td>-7.888889</td>
      <td>7.777778</td>
      <td>0.545635</td>
      <td>-1.121818</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>188</td>
      <td>133</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.333333</td>
      <td>0.266667</td>
      <td>0.500000</td>
      <td>0.077778</td>
      <td>6.666666</td>
      <td>8.333334</td>
      <td>7.777778</td>
      <td>3.888889</td>
      <td>5.000000</td>
      <td>3.333333</td>
      <td>-8.333333</td>
      <td>8.444445</td>
      <td>0.538580</td>
      <td>-0.924817</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>105</td>
      <td>139</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.277778</td>
      <td>0.107407</td>
      <td>0.833333</td>
      <td>0.522222</td>
      <td>6.111111</td>
      <td>7.555555</td>
      <td>7.222222</td>
      <td>3.555556</td>
      <td>4.333334</td>
      <td>3.333333</td>
      <td>-7.666666</td>
      <td>7.555555</td>
      <td>0.532628</td>
      <td>-0.965946</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>34</td>
      <td>137</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.500000</td>
      <td>0.166667</td>
      <td>1.111111</td>
      <td>0.474074</td>
      <td>5.851852</td>
      <td>7.777778</td>
      <td>6.444445</td>
      <td>3.333333</td>
      <td>5.777778</td>
      <td>1.777778</td>
      <td>-7.555555</td>
      <td>7.777778</td>
      <td>0.573633</td>
      <td>-0.744272</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>39</td>
      <td>111</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.722222</td>
      <td>0.374074</td>
      <td>0.888889</td>
      <td>0.429629</td>
      <td>6.037037</td>
      <td>7.000000</td>
      <td>7.666666</td>
      <td>3.444444</td>
      <td>2.888889</td>
      <td>4.888889</td>
      <td>-7.777778</td>
      <td>7.888889</td>
      <td>0.562919</td>
      <td>-1.175773</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## 3. Preprocesar datos
Como PCA se soporta de la desviación estándar de los datos para calcular la nueva proyección de nuestros datos, una variable con una desviación estándar alta tendrá un peso mayor para el cálculo de la proyección que una variable con una desviación estándar baja. Si normalizamos los datos, todas las variables tendrán la misma desviación estándar, por lo tanto, el cálculo no estará cargado. 

Además, como no tenemos conocimiento del dominio del conjunto de datos de ejemplo, no sabemos si las unidades de medida de sus variables son distintas. Otra razón por la cual normalizar nuestros datos.

PCA se considera como un algoritmo no supervisado, esto quiere decir que se apoya únicamente del set de datos sin las clases asignadas. Por esto, el primer paso de preprocesamiento será dividir nuestro set en dos: el set con los atributos y el set de las clases de asignación. Paso continuo sería estandarizar los datos sin la columna de las clases.


```python
x = dataset.drop('clase', 1)
y = dataset['clase']

x_estandarizada = StandardScaler().fit_transform(x)

try:
    atributos.remove('clase')
except:
    print('')
    
x_fin = pd.DataFrame(data = x_estandarizada, columns = atributos)
x_fin.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>atrib_1</th>
      <th>atrib_2</th>
      <th>atrib_3</th>
      <th>atrib_4</th>
      <th>atrib_5</th>
      <th>atrib_6</th>
      <th>atrib_7</th>
      <th>atrib_8</th>
      <th>atrib_9</th>
      <th>atrib_10</th>
      <th>atrib_11</th>
      <th>atrib_12</th>
      <th>atrib_13</th>
      <th>atrib_14</th>
      <th>atrib_15</th>
      <th>atrib_16</th>
      <th>atrib_17</th>
      <th>atrib_18</th>
      <th>atrib_19</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.058049</td>
      <td>-0.285629</td>
      <td>0.0</td>
      <td>-0.338097</td>
      <td>-0.199868</td>
      <td>-0.621918</td>
      <td>-0.110554</td>
      <td>-0.463761</td>
      <td>-0.090389</td>
      <td>-0.784836</td>
      <td>-0.693751</td>
      <td>-0.796105</td>
      <td>-0.848481</td>
      <td>1.511535</td>
      <td>-0.712136</td>
      <td>-0.070207</td>
      <td>-0.849913</td>
      <td>0.757935</td>
      <td>-0.001032</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.598297</td>
      <td>-0.153891</td>
      <td>0.0</td>
      <td>-0.338097</td>
      <td>-0.199868</td>
      <td>-0.598559</td>
      <td>-0.103935</td>
      <td>-0.509463</td>
      <td>-0.093776</td>
      <td>-0.769504</td>
      <td>-0.659068</td>
      <td>-0.793056</td>
      <td>-0.837138</td>
      <td>1.680258</td>
      <td>-0.772656</td>
      <td>-0.106443</td>
      <td>-0.831181</td>
      <td>0.724362</td>
      <td>0.119387</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.536634</td>
      <td>-0.055087</td>
      <td>0.0</td>
      <td>-0.338097</td>
      <td>-0.199868</td>
      <td>-0.621918</td>
      <td>-0.109110</td>
      <td>-0.418060</td>
      <td>-0.087325</td>
      <td>-0.787194</td>
      <td>-0.686044</td>
      <td>-0.808304</td>
      <td>-0.848481</td>
      <td>1.607948</td>
      <td>-0.772656</td>
      <td>-0.052089</td>
      <td>-0.856157</td>
      <td>0.696035</td>
      <td>0.094247</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.507478</td>
      <td>-0.088021</td>
      <td>0.0</td>
      <td>-0.338097</td>
      <td>-0.199868</td>
      <td>-0.528480</td>
      <td>-0.107184</td>
      <td>-0.341891</td>
      <td>-0.088024</td>
      <td>-0.795450</td>
      <td>-0.678336</td>
      <td>-0.829651</td>
      <td>-0.856043</td>
      <td>1.764620</td>
      <td>-0.857382</td>
      <td>-0.043030</td>
      <td>-0.849913</td>
      <td>0.891177</td>
      <td>0.229748</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.439109</td>
      <td>-0.516170</td>
      <td>0.0</td>
      <td>-0.338097</td>
      <td>-0.199868</td>
      <td>-0.435042</td>
      <td>-0.100444</td>
      <td>-0.402826</td>
      <td>-0.088669</td>
      <td>-0.789553</td>
      <td>-0.705312</td>
      <td>-0.796105</td>
      <td>-0.852262</td>
      <td>1.451277</td>
      <td>-0.687929</td>
      <td>-0.061148</td>
      <td>-0.846791</td>
      <td>0.840188</td>
      <td>-0.034012</td>
    </tr>
  </tbody>
</table>
</div>



## 4. PCA

### 4.1. Aplicar PCA
A continuación aplicaremos el PCA con tantos componentes principales como especifique el usuario.
Se calculan 2 por defecto para que a continuación podamos graficar nuestros datos.


```python
nComponentes = int(input())
pca = PCA(n_components=nComponentes)
```

    2



```python
atributos = []
for i in range(nComponentes):
    atributos.append('PC'+ str(i+1))
```


```python
x_pca = pca.fit_transform(x_estandarizada)
pca_dataframe = pd.DataFrame(data = x_pca, columns=atributos)
```

Una vez aplicado el PCA, podemos observar que las dimensiones se redujeron.


```python
x_fin = pca_dataframe
x_fin.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PC1</th>
      <th>PC2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-2.299677</td>
      <td>-0.343837</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-2.371925</td>
      <td>-0.403867</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-2.389590</td>
      <td>-0.266572</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-2.514506</td>
      <td>-0.096834</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-2.236746</td>
      <td>-0.124341</td>
    </tr>
  </tbody>
</table>
</div>



### 4.2. Análisis de resultados

A continuación calcularemos la razón de varianza para cada componente. Ésta relación es la varianza causada por cada componente princiapl. Esta nos sirve para observar que tan bien los componentes principales calculados representan a nuestros datos originales. 

Estos son las razones para los 19 componentes principales.
```
[4.13814900e-01, 1.65640288e-01, 1.06345188e-01, 6.27965215e-02,
5.78674248e-02, 4.70498598e-02, 4.20205907e-02, 3.83644353e-02,
2.77241028e-02, 1.90903343e-02, 1.22596944e-02, 4.70502749e-03,
2.29038862e-03, 3.12439197e-05, 1.49188547e-16, 1.03612940e-16,
9.12790822e-17, 7.73403277e-17, 9.78771803e-34]
```

Podemos observar que el CP1 es responsable de 41.3% de la varianza. Similarmente, el CP2 causa el 16.5% de la varianza en nuestro set de datos. Por lotanto, podemos decir que colectivamente, los dos primeros componentes principales capturan el 57.8% (41.3% + 16.5%) de la información de nuestro dataset, lo cual no es tan óptimo pero nos puede ser útil, por ejemplo, para graficar nuestros datos.

Como ejemplo, si quisieramos representar un 85% de nuestros datos, tendríamos que ocupar los 6 Componentes picipales primarios.


```python
explained_variance = pca.explained_variance_ratio_

explained_variance_total = 0
i = 1
for element in explained_variance:
    print("PC" + str(i) + ": " + "{:.3f}".format(element))
    explained_variance_total += element
    i += 1

explained_variance_total *= 100
print("Total: " + "{:.2f}%".format(explained_variance_total))
```

    PC1: 0.414
    PC2: 0.166
    Total: 57.95%


### 4.3. Visualizar PCA 2D
Nuestro dataset original contenía 19 dimensiones las cuales, a través del PCA, logramos reducir a 2. Esto lo hicimos para poder graficar nuesta data y poder recuperar patrones. 


```python
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Componente Principal 1', fontsize = 15)
ax.set_ylabel('Componente Principal  2', fontsize = 15)
ax.set_title('PCA de 2 componentes', fontsize = 20)

plt.scatter(x_pca[:,0],x_pca[:,1], c=data['clase'])

ax.grid()
```


![png](output_18_0.png)


## 5. Detección de outliers

Al graficar observamos que hay elementos en nuestro dataset que se comportan de manera extraña y están muy alejados de los demás.

Se puede ver mucha varianza en el componente 2, con algunos elementos muy alejados de la media. Éstos son posibles elementos anómalos o outliers que, sin el dominio del dataset, no podríamos evaluar con certeza.

Primero, nos apoyaremos de la técnica de graficación por bigotes para detectar outliers en los dos componentes.

En el PC 1 no vemos elementos graficados fuera de los bigotes, lo que nos indica que no hay casos anómalos.


```python
import seaborn as sns

pca_df_o = pd.concat([x_fin, data[['clase']]], axis = 1);
sns.boxplot(y = pca_df_o['PC1'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a189d7f50>




![png](output_20_1.png)


A diferencia del PC1, en el PC2 podemos observar que existen elementos muy alejados de la media y de los bordes superior e inferior (Q1 y Q3).


```python
sns.boxplot(y = pca_df_o['PC2'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a18a52990>




![png](output_22_1.png)


Para efectos de la experimentación, supondremos que sí son outliers y queremos recortarlos.
Para esto nos apoyaremos del método del Rango Inter-Cuartil (IQR) que es el mismo que emplean las gráficas de bigotes para su graficación. Este método observa la dispersión estadística de nuestros datos. Con esto podremos detectar aquellos elementos que están muy alejados de la media y sobrepasan los límites superior e inferior, y eliminarlos.


```python
def outlier_treatment(datacolumn):
    sorted(datacolumn)
    Q1,Q3 = np.percentile(datacolumn , [25,75])
    IQR = Q3 - Q1
    lower_range = Q1 - (1.5 * IQR)
    upper_range = Q3 + (1.5 * IQR)
    return lower_range,upper_range
```

Calculamos los límites inferior y superior.


```python
lowerbound,upperbound = outlier_treatment(pca_df_o.PC2)
print(lowerbound, upperbound)
```

    -1.4671932005127526 1.1785539827654923


Una vez identificados los límites podemos observar aquellos elementos que los exceden.


```python
pca_df_o[(pca_df_o.PC2 < lowerbound) | (pca_df_o.PC2 > upperbound)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PC1</th>
      <th>PC2</th>
      <th>clase</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>30</th>
      <td>7.088319</td>
      <td>-1.560816</td>
      <td>5</td>
    </tr>
    <tr>
      <th>31</th>
      <td>6.653630</td>
      <td>-1.589645</td>
      <td>5</td>
    </tr>
    <tr>
      <th>32</th>
      <td>6.754232</td>
      <td>-1.498079</td>
      <td>5</td>
    </tr>
    <tr>
      <th>34</th>
      <td>7.097352</td>
      <td>-1.683036</td>
      <td>5</td>
    </tr>
    <tr>
      <th>35</th>
      <td>7.072242</td>
      <td>-1.571110</td>
      <td>5</td>
    </tr>
    <tr>
      <th>36</th>
      <td>6.902150</td>
      <td>-1.486911</td>
      <td>5</td>
    </tr>
    <tr>
      <th>43</th>
      <td>6.851001</td>
      <td>-1.565639</td>
      <td>5</td>
    </tr>
    <tr>
      <th>44</th>
      <td>6.424831</td>
      <td>-1.556289</td>
      <td>5</td>
    </tr>
    <tr>
      <th>45</th>
      <td>7.027830</td>
      <td>-1.539067</td>
      <td>5</td>
    </tr>
    <tr>
      <th>46</th>
      <td>6.724713</td>
      <td>-1.799339</td>
      <td>5</td>
    </tr>
    <tr>
      <th>47</th>
      <td>7.139813</td>
      <td>-1.586438</td>
      <td>5</td>
    </tr>
    <tr>
      <th>51</th>
      <td>6.992923</td>
      <td>-1.774118</td>
      <td>5</td>
    </tr>
    <tr>
      <th>52</th>
      <td>6.631356</td>
      <td>-1.481004</td>
      <td>5</td>
    </tr>
    <tr>
      <th>57</th>
      <td>6.927716</td>
      <td>-1.625539</td>
      <td>5</td>
    </tr>
    <tr>
      <th>66</th>
      <td>7.332543</td>
      <td>29.564381</td>
      <td>2</td>
    </tr>
    <tr>
      <th>69</th>
      <td>0.634807</td>
      <td>2.810154</td>
      <td>2</td>
    </tr>
    <tr>
      <th>70</th>
      <td>-0.381843</td>
      <td>2.170110</td>
      <td>2</td>
    </tr>
    <tr>
      <th>72</th>
      <td>3.782068</td>
      <td>6.501359</td>
      <td>2</td>
    </tr>
    <tr>
      <th>77</th>
      <td>3.894876</td>
      <td>8.263521</td>
      <td>2</td>
    </tr>
    <tr>
      <th>83</th>
      <td>1.302963</td>
      <td>8.787455</td>
      <td>2</td>
    </tr>
    <tr>
      <th>115</th>
      <td>1.803542</td>
      <td>3.317052</td>
      <td>1</td>
    </tr>
    <tr>
      <th>116</th>
      <td>2.196656</td>
      <td>1.927560</td>
      <td>1</td>
    </tr>
    <tr>
      <th>119</th>
      <td>2.646916</td>
      <td>2.595937</td>
      <td>1</td>
    </tr>
    <tr>
      <th>132</th>
      <td>-1.363235</td>
      <td>1.940031</td>
      <td>6</td>
    </tr>
    <tr>
      <th>151</th>
      <td>0.264748</td>
      <td>2.194292</td>
      <td>4</td>
    </tr>
    <tr>
      <th>162</th>
      <td>1.471837</td>
      <td>1.291218</td>
      <td>4</td>
    </tr>
    <tr>
      <th>167</th>
      <td>1.499283</td>
      <td>2.125307</td>
      <td>4</td>
    </tr>
    <tr>
      <th>170</th>
      <td>2.954066</td>
      <td>1.884618</td>
      <td>4</td>
    </tr>
    <tr>
      <th>242</th>
      <td>-2.681541</td>
      <td>1.368012</td>
      <td>3</td>
    </tr>
    <tr>
      <th>317</th>
      <td>2.420052</td>
      <td>1.702191</td>
      <td>4</td>
    </tr>
    <tr>
      <th>319</th>
      <td>2.339737</td>
      <td>1.287311</td>
      <td>4</td>
    </tr>
    <tr>
      <th>419</th>
      <td>-0.218876</td>
      <td>1.686693</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>



Una vez identificados nuestros outliers, procederemos a eliminarlos.


```python
final_df = pca_df_o

final_df.drop(
    final_df[
        (final_df.PC2 > upperbound) | (final_df.PC2 < lowerbound) 
    ].index , inplace=True
)
```


```python
final_df[(final_df.PC2 < lowerbound) | (final_df.PC2 > upperbound)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PC1</th>
      <th>PC2</th>
      <th>clase</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



Como podemos observar, una vez eliminados los outliers, graficamos la gráfica de bigotes del PC2 y no tenemos outliers.


```python
sns.boxplot(y = final_df['PC2'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a18b3f850>




![png](output_33_1.png)


Una vez que limpiamos nuestros datos, procederemos a graficar de nuevo nuestros dos primeros componentes principales para observar de mejor manera como se dispersan nuestras clases.


```python
fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('PC1', fontsize = 15)
ax.set_ylabel('PC2', fontsize = 15)
ax.set_title('PCA 2D', fontsize = 20)


targets = [0, 1, 2, 3, 4, 5, 6]
colors = ['#1abc9c', '#3498db', '#9b59b6', '#34495e', '#f1c40f', '#e74c3c', '#95a5a6']
for target, color in zip(targets,colors):
    indicesToKeep = final_df['clase'] == target
    ax.scatter(final_df.loc[indicesToKeep, 'PC1']
               , final_df.loc[indicesToKeep, 'PC2']
               , c = color
               , s = 50)
ax.legend(targets, title="Clases")
ax.grid()
```


![png](output_35_0.png)



```python
x_fin = final_df.drop('clase', 1)
y_fin = final_df['clase']

x_fin.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PC1</th>
      <th>PC2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-2.299677</td>
      <td>-0.343837</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-2.371925</td>
      <td>-0.403867</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-2.389590</td>
      <td>-0.266572</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-2.514506</td>
      <td>-0.096834</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-2.236746</td>
      <td>-0.124341</td>
    </tr>
  </tbody>
</table>
</div>



## 6. Clasificación
Tener una gran cantidad de atributos en un dataset afecta el rendimiento y la precisión de los algoritmos de clasificación. Nuestro dataset original contenía 19 atributos, los cuales, a través de la técnica de reducción de dimensionalidad de PCA, logramos reducir a 2.

En este ejemplo entrenaremos un árbol de decisión con nuestros datos reducidos con PCA. En seguida, analizaremos la precisión de éste cuando es entrenado con distintas cantidades de Componentes Principales. El objetivo es ver el número óptimo de Componentes Principales que nos permitan reducir el tiempo de entrenamiento del clasificador al resumir nuestros datos adecuadamente, y conservar un elevado porcentaje de precisión.


```python
dataframe = pd.concat([x_fin, y_fin], axis = 1);

X = dataframe.drop('clase', 1)
y = dataframe['clase']

```

El método de clasificación por árbol de decisión es un método de aprendizaje supervisado, por esto, debemos entrenarlo con el set de atributos y su clasificación inicial. Además, para probar la precisión de éste, necesitamos un set de prueba. Por ello procederemos a partir nuestro set de datos en 2 secciones: Un set para entrenar a nuestro clasificador, y uno para entrenarlo.


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

## 6.1 Clasificar
Procederemos a entrenar nuestro clasificador con el set de datos de entrenamiento y sus respectivas clases.


```python
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
```


```python
import graphviz 
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("arbol")
```




    'arbol.pdf'



A continuación se muestra el árbol de decisión generado. El objetivo es generar un árbol no tan profundo para que la toma de decisiónes sea rápida. 


```python
import pydotplus
dot_data = tree.export_graphviz(clf, out_file=None, 
                      feature_names=atributos,  
                      class_names= str([0, 1, 2, 3, 4, 5, 6]),  
                      filled=True, rounded=True,  
                      special_characters=True) 


pydot_graph = pydotplus.graph_from_dot_data(dot_data)
pydot_graph.set_size('"10,10!"')

gvz_graph = graphviz.Source(pydot_graph.to_string())
gvz_graph
```




![svg](output_45_0.svg)



## 6.2 Predicción y Análisis
Por último predeciremos la clasificación de los datos de prueba que apartamos del set original antes de clasificar y compararemos, a través de una matriz de confusión, que tanto éstos se alejan de su verdadera clasificación.


```python
y_pred = clf.predict(X_test)
```


```python
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

cm = confusion_matrix(y_test, y_pred)
print("Número de Componentes Principales ocupados:")
print(nComponentes)
try:
    print("Explained_Variance: \n" + "{:.2f}%\n".format(explained_variance_total))
except:
    print()
print("Matriz de confusión: ")
print(cm)
print('\nPrecición de clasificación: \n' + "{:.2f}%\n".format(accuracy_score(y_test, y_pred)*100))
```

    Número de Componentes Principales ocupados:
    2
    Explained_Variance: 
    57.95%
    
    Matriz de confusión: 
    [[ 4  0  1  1  0  0  2]
     [ 0  1  0  0  3  0  0]
     [ 2  0  3  0  0  0  3]
     [ 0  0  0 11  0  0  2]
     [ 0  4  0  0 21  0  0]
     [ 0  0  0  0  0  2  0]
     [ 0  2  2  0  0  0 14]]
    
    Precición de clasificación: 
    71.79%
    


Podemos obeservar que al ocupar 2 componentes principales, estamos representando el 57.95% de nuestros datos originales y la precisión de nuestro clasificador es de 70.51%, lo cual no es óptimo.

Para observar la variación en la precisión del clasificador dependiendo de la cantidad de componentes principales que ocupemos realizamos múltiples experimentos, entrenando el clasificador el resultado del PCA tras alterar la cantidad de componentes principales que deseabamos.

A continuación se presenta la gráfica que relaciona la cantidad de componentes principales ocupados para el entrenmamiento del clasificador y su respectiva calificación de precisión de clasificación.


```python
from IPython.display import Image
Image(filename='analisis.png') 
```




![png](output_50_0.png)



Como se observa en la gráfica, a partir de los 11-12 componentees principales, se tiene una casi perfecta representación de los datos principales. A su vez, es en estos valores que el clasificador tiene su máximo en cuanto a grado de precisión. Esto quiere decir que reducir nuestra data a 12 dimensiones, en este ejmplo, sería óptimo para el clasificador.
