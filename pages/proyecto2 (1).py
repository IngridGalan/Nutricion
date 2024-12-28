# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import kagglehub
import zipfile
import os

# Download latest version
path = kagglehub.dataset_download("bitanianielsen/nutrition-daily-meals-in-diseases-cases")
folder_path = path;

# Busca el archivo CSV en la carpeta
csv_path = None
for file_name in os.listdir(folder_path):
    if file_name.endswith(".csv"):
        csv_path = os.path.join(folder_path, file_name)
        break


df = pd.read_csv(csv_path)
print("Path to dataset files:", path)

df

"""# Hipotesis

## Hipotesis 1: Nivel de actividad
Los pacientes con un nivel de actividad menor o igual a moderado y que ingieren mas calorias de su nivel recomendado son más propensos a padecer múltiples enfermedades, incluyendo aquellas consideradas graves como hipertensión y enfermedades del corazón y renales.

## Hipotesis 2: Diferencia de ingesta calórica
Las comidas diseñadas para enfermedades metabólicas tienen una ingesta calórica significativamente menor que las diseñadas para enfermedades cardiovasculares.

## Hipotesis 3: Restricción proteica y daño renal crónico
Los pacientes con enfermedades renales crónicas que siguen dietas restringidas en proteínas experimentan una menor progresión del daño renal que aquellos con alimentación no controladas.

## Hipotesis 4: Consumo de azúcar y proteína en dependecia de la dieta
Las personas que prefieren una dieta vegetariana tienen un consumo de proteínas más bajo pero un consumo de azúcar más alto en comparación con quienes siguen una dieta omnívora.

##Hipotesis 5
Las personas con un mayor consumo de fibra tienden a tener un peso corporal promedio más bajo en comparación con aquellas con un consumo reducido de fibra.

Las mujeres con un consumo de fibra superior a 23 gramos al día tienen un peso corporal promedio inferior a 74 kilogramos, mientras que para los hombres, aquellos con un consumo de fibra superior a 34 gramos al día tienen un peso corporal promedio inferior a 81 kilogramos.

# Analisis Exploratorio

## Descripcion de dataset

El dataset recopila la informacion de personas que padecen enfermedades relacionadas al peso corporal. Para cada una de las personas se cuenta con la siguiente informacion:

- Edad
- Genero
- Altura, medida en cm.
- Peso, medido en kilogramos.
- Nivel de actividad: es una columna cualitativa que especifica el que tan fisicamente activa es la persona, se tienen los niveles sedentario, ligeramente activo, moderadamente activo, muy activo y extremadamente activo.
- Dieta preferida: es una columna cualitativa que especifica el tipo de comida que prefiere el paciente, se tienen las opciones omnivoro, vegano, vegetariano, pescetariano.
- Objetivo de calorias diarias: es una columna cuantitativa que especifica las calorias a quemar en el dia a dia.
- Proteina: es una columna cuantitativa que mide la cantidad de gramos de proteina que consume la persona en un dia.
- Azucar: es una columna cuantitativa que mide la cantidad de gramos de azucar que consume la persona en un dia.
- Sodio: es una columna cuantitativa que mide la cantidad de gramos de sodio que consume la persona en un dia.
- Carbohidratos: es una columna cuantitativa que mide la cantidad de gramos de carbohidratos que consume la persona en un dia.
- Fibra: es una columna cuantitativa que mide la cantidad de gramos de fibra que consume la persona en un dia.
- Grasas: es una columna cuantitativa que mide la cantidad de gramos de grasas que consume la persona en un dia.
- Calorias: es una columna cuantitativa que mide las calorias que actualemente consume la persona.

- Las columnas de sugerencia sobre Desayuno, almuerzo, cena y snack, son columnas cualitativas que expresan las comidas que han sido sugeridas a cada paciente, debido a que es una de las partes mas personalizadas no se tiene una clasificacion directa sobre las comidas, ya que para cada tiempo de comida existe una comida que es la mas recomendada pero tambien aparecen casos en los que una comida solo ha sido recomendada a un paciente.

  - Desayuno mas recomendado: "Smoothie with protein powder", 210 recomendaciones
  - Almuerzo mas recomendado: "Lentil soup with whole wheat bread", 161 recomendaciones
  - Cena mas recomendada: "Salmon with roasted vegetables", 175 recomendaciones
  - Snack mas recomendado: "Trail mix" , 296 recomendaciones

- Enfermedad: es una columna cualitativa que detalla la o las enfermedades que padece el paciente. Las posibles enfermedades son:
  - Diabetes
  - Acne
  - Perdida de peso
  - Hipertension
  - Enfermedad cardíaca
  - Enfermedad Renal
"""

print(f'Desayuno: {df["Breakfast Suggestion"].value_counts().idxmax()}, {df["Breakfast Suggestion"].value_counts().max()}')
print(f'Almuerzo: {df["Lunch Suggestion"].value_counts().idxmax()}, {df["Lunch Suggestion"].value_counts().max()}')
print(f'Snack: {df["Snack Suggestion"].value_counts().idxmax()}, {df["Snack Suggestion"].value_counts().max()}')
print(f'Cena: {df["Dinner Suggestion"].value_counts().idxmax()}, {df["Dinner Suggestion"].value_counts().max()}')

df.describe()

columns = ['Calories', 'Protein', 'Sugar', 'Sodium', 'Carbohydrates', 'Fiber', 'Fat']

# Configurar el tamaño del lienzo
plt.figure(figsize=(14, 20))

# Graficar cada columna
for i, col in enumerate(columns, 1):
    plt.subplot(len(columns), 1, i)  # Crear subgráficos
    plt.scatter(df['Weight'], df[col], alpha=0.7, edgecolors='k')
    plt.title(f'Relación entre Peso y {col}')
    plt.xlabel('Peso')
    plt.ylabel(col)
    plt.grid(True)

# Mostrar todos los gráficos
plt.tight_layout()
plt.show()

"""

## Desarrollo Hipotesis 1: Nivel de actividad
"""

#Filtrado de datos
data = df['Activity Level'].isin(['Sedentary','Lightly Active','Moderately Active'])
data_filtered = df[data]
data_filtered =data_filtered[['Activity Level','Disease']]
data_filtered

#Mapeo de niveles de actividad
map_act = {'Sedentary':0,'Lightly Active':1,'Moderately Active':2}
data_filtered['Activity Mapped'] = data_filtered['Activity Level'].map(map_act)
data_filtered['Calorie_Excess'] = df['Calories'] - df['Daily Calorie Target']
data_filtered

data_filtered['Num_Diseases'] = data_filtered['Disease'].apply(lambda x: len(x.split(',')))
stacked_data = data_filtered.groupby(['Activity Level', 'Num_Diseases']).size().unstack(fill_value=0)
stacked_data.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Distribución de Enfermedades por Nivel de Actividad')
plt.xlabel('Nivel de Actividad (Mapped)')
plt.ylabel('Número de Pacientes')
plt.legend(title='Número de Enfermedades')
plt.show()

target_diseases = ['Heart Disease', 'Kidney Disease', 'Hypertension']

for disease in target_diseases:
    data_filtered[disease] = data_filtered['Disease'].apply(lambda x: 1 if disease in x else 0)

grouped_data = data_filtered.groupby('Activity Level')[target_diseases].sum()

grouped_data.plot(kind='bar', figsize=(10, 6))

# Configuración del gráfico
plt.title('Incidencia de Enfermedades por Nivel de Actividad')
plt.xlabel('Nivel de Actividad (Mapped)')
plt.ylabel('Número de Casos')
plt.legend(title='Enfermedad')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.show()

import matplotlib.pyplot as plt

# Preparar los datos para el gráfico de dispersión
diseases = ['Heart Disease', 'Kidney Disease', 'Hypertension']
data_for_scatter = data_filtered.melt(
    id_vars=['Calorie_Excess', 'Activity Level'],  # Conservar 'Activity Mapped'
    value_vars=diseases,
    var_name='Disease',
    value_name='Has Disease'
)

# Filtrar solo los registros donde la enfermedad está presente
data_for_scatter = data_for_scatter[data_for_scatter['Has Disease'] == 1]

# Crear el gráfico de dispersión
plt.figure(figsize=(10, 6))
for disease in diseases:
    disease_data = data_for_scatter[data_for_scatter['Disease'] == disease]
    plt.scatter(disease_data['Disease'], disease_data['Calorie_Excess'], label=disease, alpha=0.7)

# Configuración del gráfico
plt.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.6)  # Línea para marcar calorías neutrales
plt.title('Diferencia Calórica vs Enfermedades')
plt.xlabel('Enfermedades')
plt.ylabel('Calorie Excess')
plt.legend(title='Enfermedades', loc='best')
plt.xticks(rotation=15)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

import plotly.express as px
import pandas as pd

# Asegurarte de que la columna 'Activity Level' sea categórica y establecer el orden deseado
data_for_scatter['Activity Level'] = pd.Categorical(
    data_for_scatter['Activity Level'],
    categories=['Sedentary', 'Lightly Active', 'Moderately Active'],
    ordered=True
)

# Crear el gráfico 3D interactivo con el nuevo orden
fig = px.scatter_3d(
    data_for_scatter,
    x='Disease',
    y='Calorie_Excess',
    z='Activity Level',  # Usar la columna categórica con el orden ajustado
    color='Disease',
    title='Relación entre Diferencia Calórica, Enfermedades y Nivel de Actividad',
    labels={'Calorie_Excess': 'Calorie Excess', 'Activity Level': 'Activity Level'},
    opacity=0.7
)

# Mostrar el gráfico
fig.show()

data_for_scatter

"""## Hipotesis 2: Diferencia de ingesta calórica"""

# Clasificación de enfermedades
metabolic_diseases = ['Diabetes']
cardiovascular_diseases = ['Heart Disease', 'Hypertension']

def classify_disease(disease_list):
    diseases = disease_list.split(',')
    if any(disease in metabolic_diseases for disease in diseases):
        return 'Metabolic'
    elif any(disease in cardiovascular_diseases for disease in diseases):
        return 'Cardiovascular'
    return 'Other'

df['Disease Type'] = df['Disease'].apply(classify_disease)

calories_data = df[['Disease Type', 'Calories']].dropna()

calories_metrics = calories_data.groupby('Disease Type')['Calories'].agg(['mean', 'std'])
calories_metrics

plt.figure(figsize=(10, 6))
sns.histplot(data=calories_data, x='Calories', hue='Disease Type', kde=True, bins=30, palette="Set2")
plt.title('Distribución de Ingesta Calórica por Tipo de Enfermedad')
plt.xlabel('Ingesta Calórica (Kcal)')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.show()

from scipy import stats

metabolic_calories = calories_data[calories_data['Disease Type'] == 'Metabolic']['Calories']
cardiovascular_calories = calories_data[calories_data['Disease Type'] == 'Cardiovascular']['Calories']

t_stat, p_value = stats.ttest_ind(metabolic_calories, cardiovascular_calories)
print(f'T-statistic: {t_stat} \nP-value: {p_value}')

import seaborn as sns
import matplotlib.pyplot as plt

#para comparar calorías
plt.figure(figsize=(8, 6))
sns.boxplot(x='Disease Type', y='Calories', data=calories_data)
plt.title('Comparación de Calorías entre Enfermedades Metabólicas y Cardiovasculares')
plt.xlabel('Tipo de Enfermedad')
plt.ylabel('Calorías')
plt.grid(True)
plt.show()

columns_of_interest = ['Calories', 'Protein', 'Sugar', 'Sodium', 'Carbohydrates', 'Fiber', 'Fat']

nutrient_data = df[columns_of_interest]
print(df[columns_of_interest].isnull().sum())
df[columns_of_interest] = df[columns_of_interest].fillna(df[columns_of_interest].mean())

correlation_matrix = nutrient_data.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Mapa de Calor de Correlaciones entre Nutrientes')
plt.show()

"""## Hipotesis 3: Restricción proteica y daño renal crónico"""

#Gráfico de distribución de la ingesta de proteínas
plt.figure(figsize=(10, 6))
sns.histplot(df['Protein'], kde=True, color='skyblue')
plt.title("Distribución de la ingesta de proteínas")
plt.xlabel("Gramos de proteínas")
plt.ylabel("Frecuencia")
plt.show()

#Grafico de comparación entre ingesta de proteínas y progresión del daño renal
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Protein', y='Disease', data=df, hue='Dietary Preference')
plt.title("Relación entre ingesta de proteínas y daño renal")
plt.xlabel("Ingesta de proteínas (g)")
plt.ylabel("Progresión del daño renal")
plt.legend(title="Dieta Controlada")
plt.show()

#Encontrar relación entre columnas
plt.figure(figsize=(12, 8))
# Selecciona solo las columnas numéricas para el cálculo de la correlación
numeric_df = df.select_dtypes(include=['number'])
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de correlación entre variables")
plt.show()

#Boxplot de progresión del daño renal según dieta controlada
plt.figure(figsize=(8, 6))
sns.boxplot(x='Dietary Preference', y='Protein', data=df)  # Reemplaza 'Protein' con la columna real que representa la progresión de la enfermedad renal si es diferente
plt.title("Progresión del daño renal según control de la dieta")
plt.xlabel("Dieta Controlada")
plt.ylabel("Progresión del daño renal")
plt.show()

"""## Hipotesis 4: Consumo de azúcar y proteína en dependecia de la dieta


"""

# Configuración general para los gráficos
sns.set(style="whitegrid")

# Paleta monocromática personalizada
mono_palette = sns.dark_palette("blue", n_colors=3, reverse=False)

# Filtrar datos para incluir solo Omnivore y Vegetarian
filtered_df = df[df['Dietary Preference'].isin(['Omnivore', 'Vegetarian'])]

# Boxplot para consumo de azúcar
plt.figure(figsize=(10, 6))
sns.boxplot(x='Dietary Preference', y='Sugar', data=filtered_df, palette=mono_palette[:2], hue='Dietary Preference', legend=False)
plt.title('Comparación de Consumo de Azúcar por Preferencia Dietética (Omnivore vs Vegetarian)', fontsize=14)
plt.xlabel('Preferencia Dietética', fontsize=12)
plt.ylabel('Consumo de Azúcar (g)', fontsize=12)
plt.show()

# Filtrar datos para incluir solo Omnivore y Vegetarian
filtered_df = df[df['Dietary Preference'].isin(['Omnivore', 'Vegetarian'])]

# Boxplot para consumo de proteínas
plt.figure(figsize=(10, 6))
sns.boxplot(x='Dietary Preference', y='Protein', data=filtered_df, palette=mono_palette[:2], hue='Dietary Preference', legend=False)
plt.title('Comparación de Consumo de Proteínas por Preferencia Dietética (Omnivore vs Vegetarian)', fontsize=14)
plt.xlabel('Preferencia Dietética', fontsize=12)
plt.ylabel('Consumo de Proteínas (g)', fontsize=12)
plt.show()

# Scatter plot para observar relación entre consumo de azúcar y proteínas (Omnivore vs Vegetarian)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Sugar', y='Protein', hue='Dietary Preference', data=filtered_df, palette=mono_palette[:2])
plt.title('Relación entre Consumo de Azúcar y Proteínas (Omnivore vs Vegetarian)', fontsize=14)
plt.xlabel('Consumo de Azúcar (g)', fontsize=12)
plt.ylabel('Consumo de Proteínas (g)', fontsize=12)
plt.legend(title='Preferencia Dietética', fontsize=10)
plt.show()

# Gráfico de barras agrupadas para el consumo promedio por dieta
mean_values = filtered_df.groupby('Dietary Preference')[['Sugar', 'Protein']].mean().reset_index()
mean_values_melted = mean_values.melt(id_vars='Dietary Preference', var_name='Nutriente', value_name='Consumo Promedio')

plt.figure(figsize=(12, 6))
sns.barplot(x='Dietary Preference', y='Consumo Promedio', hue='Nutriente', data=mean_values_melted, palette=mono_palette[:2])
plt.title('Consumo Promedio de Azúcar y Proteínas por Dieta', fontsize=14)
plt.xlabel('Preferencia Dietética', fontsize=12)
plt.ylabel('Consumo Promedio (g)', fontsize=12)
plt.legend(title='Nutriente', fontsize=10)
plt.show()

"""## Hipotesis 5



"""

promedio_por_genero = df.groupby('Gender')['Weight'].mean()

# Mostrar los resultados
print("Peso promedio por género:")
print(promedio_por_genero)

import numpy as np

# Crear la figura
plt.figure(figsize=(12, 8))

# Configurar el estilo base de seaborn
sns.set_theme(style="whitegrid")

# Crear el scatter plot
sns.scatterplot(data=df,
                x='Fiber',
                y='Weight',
                hue='Gender',
                style='Gender',
                s=80,  # Tamaño de los puntos
                alpha=0.6,  # Transparencia
                palette={'Male': '#4E79A7', 'Female': '#F28E2B'},  # Colores más amigables
                markers={'Male': 'o', 'Female': 's'})  # Círculos para hombres, cuadrados para mujeres

# Agregar líneas de referencia
plt.axvline(x=23, color='#F28E2B', linestyle='--', alpha=0.3, label='Límite fibra mujeres (23g)')
plt.axhline(y=74, color='#F28E2B', linestyle='--', alpha=0.3, label='Límite peso mujeres (74kg)')
plt.axvline(x=34, color='#4E79A7', linestyle='--', alpha=0.3, label='Límite fibra hombres (34g)')
plt.axhline(y=81, color='#4E79A7', linestyle='--', alpha=0.3, label='Límite peso hombres (81kg)')

# Agregar líneas de tendencia para cada género
for genero in df['Gender'].unique():
    datos = df[df['Gender'] == genero]
    z = np.polyfit(datos['Fiber'], datos['Weight'], 1)
    p = np.poly1d(z)
    plt.plot(datos['Fiber'], p(datos['Fiber']),
             linestyle='-', alpha=0.8,
             color='#4E79A7' if genero == 'Male' else '#F28E2B')

# Personalizar el gráfico
plt.title('Relación entre Consumo de Fibra y Peso Corporal\npor Género',
          pad=20, size=14, fontweight='bold')
plt.xlabel('Consumo de Fibra (g/día)', size=12)
plt.ylabel('Peso Corporal (kg)', size=12)

# Mejorar la leyenda
plt.legend(title='Género',
          title_fontsize=12,
          bbox_to_anchor=(1.15, 1),
          loc='upper left')

# Agregar anotación explicativa
plt.text(0.02, 0.98, 'Tendencia: A mayor consumo de fibra,\nmenor peso corporal',
         transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', alpha=0.8),
         verticalalignment='top')

plt.tight_layout()
plt.show()

# Primero creamos las categorías de consumo de fibra
df['Categoria_Fibra'] = 'Bajo consumo'
# Para mujeres
df.loc[(df['Gender'] == 'Female') & (df['Fiber'] > 23), 'Categoria_Fibra'] = 'Alto consumo'
# Para hombres
df.loc[(df['Gender'] == 'Male') & (df['Fiber'] > 34), 'Categoria_Fibra'] = 'Alto consumo'

# Crear el gráfico
plt.figure(figsize=(10,6))
sns.boxplot(data=df, x='Gender', y='Weight', hue='Categoria_Fibra')
plt.title('Distribución del Peso según Consumo de Fibra por Género')
plt.xlabel('Género')
plt.ylabel('Peso (kg)')
plt.show()

plt.figure(figsize=(10,6))
sns.violinplot(data=df, x='Gender', y='Weight', hue='Categoria_Fibra')
plt.title('Distribución del Peso según Consumo de Fibra por Género')
plt.xlabel('Género')
plt.ylabel('Peso (kg)')
plt.show()

# Calcular promedios
promedios = df.groupby(['Gender', 'Categoria_Fibra'])['Weight'].mean().unstack()

# Crear el gráfico
plt.figure(figsize=(10,6))
promedios.plot(kind='bar')
plt.title('Peso Promedio según Consumo de Fibra y Género')
plt.xlabel('Género')
plt.ylabel('Peso Promedio (kg)')
plt.xticks(rotation=0)
plt.legend(title='Categoría de Fibra')
plt.tight_layout()
plt.show()

# Opcional: Mostrar los valores numéricos
print("\nPromedios de peso por categoría:")
print(promedios.round(2))

# Estadísticas descriptivas más detalladas
stats_por_genero = df.groupby('Gender')[['Weight', 'Fiber']].agg(['mean', 'median', 'std'])
print("Estadísticas por género:")
print(stats_por_genero)

# Análisis para mujeres
mujeres = df[df['Gender'] == 'Female']
total_mujeres = len(mujeres)
mujeres_alto_fibra = mujeres[mujeres['Fiber'] > 23]
mujeres_cumplen_hipotesis = mujeres_alto_fibra[mujeres_alto_fibra['Weight'] < 74]

print("\nAnálisis para mujeres:")
print(f"Total de mujeres: {total_mujeres}")
print(f"Mujeres con alta fibra (>23g): {len(mujeres_alto_fibra)}")
print(f"Mujeres que cumplen la hipótesis: {len(mujeres_cumplen_hipotesis)}")
print(f"Porcentaje que cumple la hipótesis: {(len(mujeres_cumplen_hipotesis)/len(mujeres_alto_fibra)*100):.2f}%")

# Análisis para hombres
hombres = df[df['Gender'] == 'Male']
total_hombres = len(hombres)
hombres_alto_fibra = hombres[hombres['Fiber'] > 34]
hombres_cumplen_hipotesis = hombres_alto_fibra[hombres_alto_fibra['Weight'] < 81]

print("\nAnálisis para hombres:")
print(f"Total de hombres: {total_hombres}")
print(f"Hombres con alta fibra (>34g): {len(hombres_alto_fibra)}")
print(f"Hombres que cumplen la hipótesis: {len(hombres_cumplen_hipotesis)}")
print(f"Porcentaje que cumple la hipótesis: {(len(hombres_cumplen_hipotesis)/len(hombres_alto_fibra)*100):.2f}%")

# Correlación entre fibra y peso para cada género
df.groupby('Gender').apply(lambda x: x['Fiber'].corr(x['Weight']))

"""#Modelo de Machine Learning

Hemos propuesto utilizar un modelo de aprendizaje supervisado, ya que contamos con etiquetas explícitas o podemos derivarlas (como peso, consumo calórico, etc.)

El modelo de Decision Tree (árbol de decisión) es un algoritmo de modelo de aprendizaje supervisado.
El árbol de decisión funciona descomponiendo el conjunto de datos en subconjuntos más pequeños basándose en condiciones o reglas de decisión. Primero el algoritmo evalúa todas las características (las columnas) para determinar cuál divide mejor los datos según el objetivo. Seguidamente se divide en nodos que cada uno  representa una decisión basada en el valor de una característica. El árbol crece dividiendo datos en ramas hasta que se cumple una de las condiciones de parada: (No hay más datos que dividir, Se alcanza la profundidad máxima definida por el usuario, El nodo contiene datos homogéneos o suficientemente similares). Por último, para predecir, el modelo sigue las reglas de decisión desde la raíz hasta una hoja, donde asigna la clase o valor.

## Decision Tree
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# **1. Preprocesamiento**
# Convertir Activity Level a numérico (si aún no se ha hecho)
activity_mapping = {'Sedentary': 0, 'Lightly Active': 1, 'Moderately Active': 2}
data_filtered['Activity Mapped'] = data_filtered['Activity Level'].map(activity_mapping)

# Seleccionar características (X) y etiquetas (y)
X = data_filtered[['Calorie_Excess', 'Activity Mapped']]  # Puedes agregar más características si las tienes.
y = data_filtered[['Heart Disease', 'Kidney Disease', 'Hypertension']]

# Crear una etiqueta única para predecir cualquier enfermedad
y['Has Disease'] = y.max(axis=1)

# **2. Dividir en conjunto de entrenamiento y prueba**
X_train, X_test, y_train, y_test = train_test_split(X, y['Has Disease'], test_size=0.2, random_state=42)

# **3. Entrenar el modelo**
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# **4. Predicción y Evaluación**
y_pred = clf.predict(X_test)

# Métricas de evaluación
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Disease', 'Has Disease'], yticklabels=['No Disease', 'Has Disease'])
plt.title("Matriz de Confusión")
plt.xlabel("Predicción")
plt.ylabel("Realidad")
plt.show()

# Visualizar la Matriz de Confusión
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Calcular la Matriz de Confusión
conf_matrix = confusion_matrix(y_test, y_pred)

# Dibujar el gráfico
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["No Disease", "Has Disease"], yticklabels=["No Disease", "Has Disease"])
plt.title("Matriz de Confusión")
plt.xlabel("Predicción")
plt.ylabel("Realidad")
plt.show()