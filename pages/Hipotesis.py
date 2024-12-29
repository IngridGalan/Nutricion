import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import kagglehub
import zipfile
import os


#Download latest version
path = kagglehub.dataset_download("bitanianielsen/nutrition-daily-meals-in-diseases-cases")
folder_path = path;

#Busca el archivo CSV en la carpeta
csv_path = None
for file_name in os.listdir(folder_path):
    if file_name.endswith(".csv"):
        csv_path = os.path.join(folder_path, file_name)
        break


df = pd.read_csv(csv_path)
print("Path to dataset files:", path)

df

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