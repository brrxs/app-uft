<!-- id: decisiones_intro -->
### Qué cambia en esta versión

Este dashboard corre el mismo pipeline que el dashboard por APDA, pero cambia
la unidad de análisis: cada celda es una institución dentro de un área
CINE-F 13, no dentro de una APDA. Son 9 áreas en vez de 4 grupos.

La APDA sigue presente, pero sólo como navegación: se elige primero la APDA y
después el área CINE-F 13 que la compone. Así queda a la vista qué áreas
agrupa cada APDA.

**Qué se recalcula por área.** La normalización de cada variable (z-score
dentro del grupo par), los puntajes de dimensión, los ejes, el percentil, el
PCA y los arquetipos. Todo eso se rehace con las instituciones del área, no
con las de la APDA.

**Qué no se puede recalcular.** Las variables de ANID, Scimago y SciVal vienen
de sus fuentes con grano institución × APDA. No existe una versión por área.
Entonces las áreas de una misma APDA comparten el mismo valor en esas
variables, que pesan 70 de los 100 puntos del eje Investigación. Dentro de una
APDA, ese eje separa poco entre sus áreas.

**Dónde no aplica ese problema.** Dos APDAs tienen una sola área: Artes y
Humanidades, y Salud y Bienestar. Ahí el área y la APDA son el mismo conjunto
de instituciones, así que los resultados coinciden con el dashboard por APDA
salvo por la escala 0-100, que se reescala sobre 9 grupos en vez de 4.

### Por qué porcentajes por dimensión, y no un solo puntaje

Cada área se evalúa con varios porcentajes por dimensión, no con un solo
índice combinado. Es una decisión deliberada.

**El problema con un solo número:** cada área agrupa carreras distintas entre
sí — en duración, en empleabilidad esperada, en costo, en cómo se acredita.
Forzar toda esa diversidad a un solo puntaje esconde justamente las
diferencias que importan: dos instituciones pueden llegar al mismo número
combinado por caminos opuestos, uno fuerte y otro débil.

**La alternativa:** mostrar cada dimensión por separado, como porcentaje. Así
se ve en qué es fuerte una institución y en qué no, dimensión por dimensión,
en vez de un promedio que las mezcla todas.

Este archivo es editable. El texto de esta pestaña se puede cambiar en
cualquier momento sin tocar el código.
