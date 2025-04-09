# Clasificador de Distorsiones Cognitivas

## Descripción
Este proyecto implementa un clasificador de distorsiones cognitivas basado en un modelo SVM (Support Vector Machine) entrenado para identificar 12 tipos diferentes de distorsiones cognitivas en textos en español.

## Nota
Si al correr este algoritmo se pide los nombres de los archivos no corren, solo  debe cambiar el nombre de los archivos dentro del código.

## Tipos de Distorsiones Cognitivas
El clasificador puede identificar las siguientes distorsiones:
1. Abstracción selectiva
2. Catastrofismo
3. Debería
4. Descalificación positiva
5. Etiquetar
6. Lectura de la mente
7. Maximización
8. Minimizar
9. Pensamiento adivinatorio
10. Pensamiento dicotómico
11. Razonamiento emocional
12. Sobregeneralización

## Archivos Incluidos
- `tfidf_vectorizer.pkl`: Vectorizador TF-IDF entrenado para transformar texto a características numéricas.
- `svm_model.pkl`: Modelo SVM entrenado para clasificar distorsiones cognitivas.
- `class_mapping.npy`: Mapeo de etiquetas numéricas a nombres de distorsiones cognitivas.

## Requisitos
Ver el archivo `requirements.txt` para la lista completa de dependencias.

## Instalación
```bash
pip install -r requirements.txt
```

## Uso
```python
import pickle
import numpy as np

# Cargar el modelo y vectorizador
with open('svm_model.pkl', 'rb') as file:
    modelo_svm = pickle.load(file)
    
with open('tfidf_vectorizer.pkl', 'rb') as file:
    vectorizador = pickle.load(file)
    
mapeo_clases = np.load('class_mapping.npy', allow_pickle=True).item()

# Función para clasificar mensajes
def clasificar_mensaje(mensaje):
    # Transformar el mensaje usando el vectorizador
    mensaje_vectorizado = vectorizador.transform([mensaje])
    
    # Predecir la clase
    clase_predicha = modelo_svm.predict(mensaje_vectorizado)[0]
    
    # Obtener el nombre de la clase
    nombre_clase = mapeo_clases[clase_predicha]
    
    return nombre_clase

# Ejemplo de uso
mensaje = "Nunca podré superar este problema, es demasiado difícil"
distorsion = clasificar_mensaje(mensaje)
print(f"Distorsión cognitiva detectada: {distorsion}")
```

## Licencia
Este proyecto está licenciado bajo CC BY-ND 4.0 - ver el archivo [LICENSE.md](LICENSE.md) para más detalles.

## Funding
Este algoritmo se hizo en el marco del proyecto Hatemedia (PID2020-114584GB-I00), financiado por MCIN/AEI/10.13039/501100011033.

## Contacto
Para preguntas o soporte, por favor abra un issue en este repositorio.

## Cómo citar este algoritmo
Said-Hung, E. M. (2025). Algoritmo de clasificación de distorciones cognitivas en mensajes con expresiones de odio. Hatemedia Project. https://doi.org/10.5281/zenodo.15180918
