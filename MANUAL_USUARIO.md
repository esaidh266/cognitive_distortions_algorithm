# Manual de Usuario: Clasificador de Distorsiones Cognitivas

## Introducción

Este manual describe cómo utilizar el clasificador de distorsiones cognitivas basado en un modelo SVM. El clasificador está diseñado para identificar 12 tipos diferentes de distorsiones cognitivas en textos en español.

## Requisitos Previos

Antes de utilizar el clasificador, asegúrese de tener instalado:

1. Python 3.7 o superior
2. Las dependencias listadas en el archivo `requirements.txt`

## Instalación

1. Clone o descargue este repositorio
2. Instale las dependencias:

```bash
pip install -r requirements.txt
```

## Estructura de Archivos

- `tfidf_vectorizer.pkl`: Vectorizador TF-IDF entrenado
- `svm_model.pkl`: Modelo SVM entrenado
- `class_mapping.npy`: Mapeo de etiquetas numéricas a nombres de distorsiones
- `Script_Modelo_SVM.py`: Script básico de ejemplo

## Uso Básico

### Desde la línea de comandos

1. Ejecute el script de ejemplo:

```bash
python ejemplo_uso_clasificador.py
```

Este script clasificará una serie de mensajes de ejemplo y mostrará los resultados.

### Desde su propio código

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
mensaje = "Nunca podré superar este problema"
distorsion = clasificar_mensaje(mensaje)
print(f"Distorsión cognitiva detectada: {distorsion}")
```

## Tipos de Distorsiones Cognitivas

El clasificador puede identificar las siguientes distorsiones:

1. **Abstracción selectiva**: Enfocarse en un detalle negativo e ignorar el contexto completo.
2. **Catastrofismo**: Anticipar el peor resultado posible sin considerar otras posibilidades.
3. **Debería**: Tener expectativas rígidas sobre cómo uno mismo o los demás deberían comportarse.
4. **Descalificación positiva**: Rechazar experiencias positivas insistiendo en que "no cuentan".
5. **Etiquetar**: Asignarse a uno mismo o a otros etiquetas globales negativas.
6. **Lectura de la mente**: Asumir que se conocen los pensamientos de los demás sin evidencia.
7. **Maximización**: Exagerar la importancia de los errores o defectos.
8. **Minimizar**: Reducir la importancia de los aspectos positivos.
9. **Pensamiento adivinatorio**: Predecir resultados negativos sin suficiente evidencia.
10. **Pensamiento dicotómico**: Ver las situaciones en términos absolutos (blanco o negro).
11. **Razonamiento emocional**: Asumir que los sentimientos negativos reflejan la realidad.
12. **Sobregeneralización**: Extraer una conclusión general a partir de un incidente aislado.

## Solución de Problemas

### Error al cargar los archivos

Asegúrese de que los archivos `svm_model.pkl`, `tfidf_vectorizer.pkl` y `class_mapping.npy` estén en el mismo directorio que su script.

### Resultados inesperados

El modelo está entrenado con textos específicos y puede no funcionar correctamente con:
- Textos muy cortos
- Textos con errores ortográficos significativos
- Textos en idiomas diferentes al español

## Contacto y Soporte

Para preguntas o soporte, por favor abra un issue en el repositorio del proyecto.
