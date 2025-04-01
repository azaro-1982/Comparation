# *Reto2 Trabajo en equipo*

Este pequeño script ha sido creado para comparar distintas versiones de un delincuente con los hechos probados
que se conocen de un delito.

## Elaboración:

Para la elaboración de este Scrip he usado las librerías sentence-transformer la cual convierte las frases en vectores
numéricos para comparar la similitud semántica con otras frases mediante el modelo all-MiniLM-L6-v2 que es un modelo más
ligero de SBERT.
También he usado la librería Transforme

## Uso:
1. Descarga del archivo.

2. Instalación de los paquetes necesarios para que el archivo programa pueda realizar los procesos mediante el archivo
requirements.txt.

pip install -r requirements.txt

3. Cambiar las frases (si se quiere)

## *Salida*

Compara las distintas versiones con la original
1. ENTAILMENT(uno implica el otro).
2. CONTRADICCIÓN (uno contradice al otro).
3. NEUTRAL
