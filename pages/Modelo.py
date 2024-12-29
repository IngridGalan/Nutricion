import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar los datos (Asegúrate de que 'data_filtered' esté definido o cargado antes de esto)
# data_filtered = pd.read_csv('ruta_del_archivo.csv')  # Si estás cargando un archivo CSV, usa esto.

# Mapeo de nivel de actividad
activity_mapping = {'Sedentary': 0, 'Lightly Active': 1, 'Moderately Active': 2}
data_filtered['Activity Mapped'] = data_filtered['Activity Level'].map(activity_mapping)

# Seleccionar características (X) y etiquetas (y)
X = data_filtered[['Calorie_Excess', 'Activity Mapped']]
y = data_filtered[['Heart Disease', 'Kidney Disease', 'Hypertension']]
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

# Dibujar el gráfico
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["No Disease", "Has Disease"], yticklabels=["No Disease", "Has Disease"])
plt.title("Matriz de Confusión")
plt.xlabel("Predicción")
plt.ylabel("Realidad")
plt.show()