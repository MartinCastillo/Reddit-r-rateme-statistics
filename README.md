<h1>Resumen</h1>
Proyecto que tiene como intención obtener estadísticas y recolectar sets de caras y las puntuaciones superficiales que le ponen los usuarios en subredes por medio de un bot.
Se explora la posibilidad de predecir el puntaje de un rostro clasificado en el contexto con la imagen de este.

<br>
Media de puntajes.

![img](https://github.com/MartinCastillo/Reddit-r-rateme-statistics/blob/master/captures/scores.png)

<br>
Obtención de rostros catalogados y con formato de forma automática.
<br>

![img](https://github.com/MartinCastillo/Reddit-r-rateme-statistics/blob/master/captures/capture_2.png)

![img](https://github.com/MartinCastillo/Reddit-r-rateme-statistics/blob/master/captures/capture_3.png)

<h1>Archivos</h1>

* ```face_score_model``` Contiene los modelos y el analisis de los datos obtenidos.<br>
* ```bot``` Contiene el bot hecho para recolectar los datos (imágenes, landmarks y clasificaciones según género y puntaje).<br>
  * -> ```Bot```<br>
    * -> ```Bot.py``` La clase del bot<br>
  * -> ```main.py``` Archivo principal para recolectar la información<br>
  * -> ```FaceAnalysis``` Contiene archivos que contienen funciones para analizar los rostros (Y alinearlos)<br>
