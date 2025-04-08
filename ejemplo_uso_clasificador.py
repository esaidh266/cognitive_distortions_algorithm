# Ejemplo de uso del Clasificador de Distorsiones Cognitivas
# --------------------------------------------------------

import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def cargar_modelo():
    """Carga el modelo SVM, el vectorizador TF-IDF y el mapeo de clases."""
    try:
        # Cargar el modelo SVM
        with open('svm_model.pkl', 'rb') as file:
            modelo_svm = pickle.load(file)
            
        # Cargar el vectorizador TF-IDF
        with open('tfidf_vectorizer.pkl', 'rb') as file:
            vectorizador = pickle.load(file)
            
        # Cargar el mapeo de clases
        mapeo_clases = np.load('class_mapping.npy', allow_pickle=True).item()
        
        return modelo_svm, vectorizador, mapeo_clases
    
    except FileNotFoundError as e:
        print(f"Error al cargar archivos: {e}")
        return None, None, None
    except Exception as e:
        print(f"Error inesperado: {e}")
        return None, None, None

def clasificar_mensaje(mensaje, modelo_svm, vectorizador, mapeo_clases):
    """Clasifica un mensaje de texto y devuelve la distorsión cognitiva detectada."""
    # Transformar el mensaje usando el vectorizador
    mensaje_vectorizado = vectorizador.transform([mensaje])
    
    # Predecir la clase
    clase_predicha = modelo_svm.predict(mensaje_vectorizado)[0]
    
    # Obtener el nombre de la clase
    nombre_clase = mapeo_clases[clase_predicha]
    
    # Obtener probabilidades (si el modelo lo soporta)
    if hasattr(modelo_svm, 'predict_proba'):
        probabilidades = modelo_svm.predict_proba(mensaje_vectorizado)[0]
        confianza = probabilidades.max() * 100
    else:
        # Usar decision_function como alternativa
        decision_scores = modelo_svm.decision_function(mensaje_vectorizado)
        confianza = None
    
    return nombre_clase, confianza

def clasificar_multiples_mensajes(mensajes, modelo_svm, vectorizador, mapeo_clases):
    """Clasifica múltiples mensajes y devuelve un DataFrame con los resultados."""
    resultados = []
    
    for mensaje in mensajes:
        distorsion, confianza = clasificar_mensaje(mensaje, modelo_svm, vectorizador, mapeo_clases)
        resultados.append({
            'Mensaje': mensaje,
            'Distorsión Cognitiva': distorsion,
            'Confianza (%)': confianza
        })
    
    return pd.DataFrame(resultados)

def main():
    # Cargar el modelo y componentes relacionados
    modelo_svm, vectorizador, mapeo_clases = cargar_modelo()
    
    if modelo_svm is None:
        print("No se pudo cargar el modelo. Verifique que los archivos existan.")
        return
    
    # Ejemplos de mensajes con posibles distorsiones cognitivas
    mensajes_ejemplo = [
        "Nunca podré superar este problema, es demasiado difícil",
        "Si cometo un error en la presentación, será un desastre total",
        "Debería ser capaz de manejar todo sin ayuda",
        "Aunque me felicitaron por mi trabajo, fue solo suerte",
        "Soy un completo fracaso por haber reprobado ese examen",
        "Sé que todos piensan que soy incompetente",
        "Este pequeño error arruinó todo el proyecto",
        "Mi contribución al proyecto fue insignificante",
        "Si acepto este trabajo, seguro que fracasaré",
        "O hago el trabajo perfectamente o mejor no lo hago",
        "Me siento ansioso, así que debe haber un peligro real",
        "Fallé una vez, así que siempre fallaré en situaciones similares"
    ]
    
    # Clasificar los mensajes de ejemplo
    print("Clasificando mensajes de ejemplo...")
    resultados_df = clasificar_multiples_mensajes(mensajes_ejemplo, modelo_svm, vectorizador, mapeo_clases)
    
    # Mostrar resultados
    print("\nResultados de la clasificación:")
    print(resultados_df)
    
    # Guardar resultados en un archivo CSV
    resultados_df.to_csv('resultados_clasificacion.csv', index=False, encoding='utf-8')
    print("\nResultados guardados en 'resultados_clasificacion.csv'")
    
    # Visualizar distribución de distorsiones detectadas
    plt.figure(figsize=(12, 6))
    conteo_distorsiones = resultados_df['Distorsión Cognitiva'].value_counts()
    sns.barplot(x=conteo_distorsiones.index, y=conteo_distorsiones.values)
    plt.title('Distribución de Distorsiones Cognitivas Detectadas')
    plt.xlabel('Tipo de Distorsión')
    plt.ylabel('Frecuencia')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('distribucion_distorsiones.png')
    plt.show()
    
    print("\nGráfico de distribución guardado como 'distribucion_distorsiones.png'")

if __name__ == "__main__":
    main()
