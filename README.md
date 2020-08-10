-Entrenar un modelo para predecir el criterio de belleza de los usuarios de reddit
La opción más tentadora es la segunda, para esto lo más simple es hacer un modelo por entrenamiento supervisado, y crear un dataset de imágenes, con sus respectivos landmarks, y un número asignado que tenga la intención de medir la belleza de dicho dato, es conveniente esta distribución pues hay subreddits como r/rateme o r/truerateme donde los usuarios ya hacen esto, por lo que sería solo cuestión de recolectar, en la parte del modelo, hay dos opciones:
-Usar como dato la imagen cruda de la cara, y esta con cierta clasificación, una posibilidad de modelo con estos datos es por medio de redes neuronales, requeriría más datos que la otra opción
-Usar los landmarks de la cara y estos con cierta clasificación, esta solo clasificaría según facciones de la cara y no piel, cabello o un conjunto y de
1día después
La parte de scrapping del bot está lista hasta las funcionalidades anteriores, aprendí expresiones regulares
2días después
-Los landmarks no me convencen, también recopila caras con mascara, sin fondo
-Desde esta última modificación recopiló 535 caras en 15 minutos, solo contando tiempo de post a post, sin el recorrer ‘espacio vacío’, quizá pueda más sin la doble obtención de landmark(que conlleva también obtener el rectángulo de la cara), pero hasta el momento no encuentro forma de evitarlo sin sacrificar la máscara de cara
Con un dataset de aproximadamente 500 caras se entrena un modelo de regresión en google colab, (basado en el clasificador de autos), el modelo se guarda para su uso sin entrenamiento y se incluye para su uso en el bot (a futuro se tendrá que mejorar el modelo)
-Según el criterio de r/truerateme las puntuaciones predichas no varian demasiado de 5, pero los comentarios son consecuentes con ello
