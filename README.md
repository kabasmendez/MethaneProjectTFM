# Segmentación de Plumas de Metano en Imágenes Sentinel-2 mediante Aprendizaje Profundo

Este repositorio contiene el flujo experimental desarrollado para el Trabajo Final de Máster orientado a la **segmentación automática de plumas de metano en imágenes Sentinel-2** mediante modelos de aprendizaje profundo.

El proyecto combina técnicas de teledetección, procesamiento multiespectral, realces espectrales sensibles al metano, variables meteorológicas auxiliares y arquitecturas de segmentación semántica basadas en U-Net y Transformers.

---

## Descripción general

El metano es uno de los gases de efecto invernadero más relevantes para la mitigación climática a corto plazo. Aunque permanece menos tiempo en la atmósfera que el dióxido de carbono, tiene un potencial de calentamiento global mucho mayor en horizontes temporales cortos. Por esta razón, la detección y delimitación de plumas de metano asociadas a superemisores es una tarea clave para el monitoreo ambiental, energético y climático.

Este proyecto aborda el problema como una tarea de **segmentación semántica binaria**, donde cada píxel de un parche Sentinel-2 se clasifica como:

- Fondo.
- Pluma de metano.

El trabajo se basa en muestras derivadas de **MethaneSet / TACO Foundation**, un conjunto de datos orientado a la detección de plumas de metano a partir de imágenes Sentinel-2, imágenes de referencia, máscaras binarias y metadatos asociados.

---

## Objetivo del proyecto

El objetivo principal es evaluar si diferentes arquitecturas de aprendizaje profundo pueden segmentar automáticamente plumas de metano en imágenes Sentinel-2 utilizando:

- Bandas espectrales Sentinel-2.
- Índices y ratios sensibles a diferencias en el SWIR.
- Realces derivados de la comparación entre imagen objetivo e imagen de referencia.
- Variables meteorológicas de viento.
- Arquitecturas U-Net convolucionales y modelos con componentes Transformer.

---

## Conjunto de datos

El conjunto de datos final se construyó a partir de muestras de MethaneSet filtradas bajo criterios de calidad y consistencia experimental. Se consideraron principalmente muestras del sector oil & gas, con plumas reales, productos completos y condiciones utilizables.

La partición final utilizada en los experimentos fue:

| Conjunto | Número de muestras |
|---|---:|
| Entrenamiento | 2.463 |
| Validación | 528 |
| Test | 528 |
| **Total** | **3.519** |

Cada muestra corresponde a un parche de **200 × 200 píxeles** con resolución espacial de 20 m.

---

## Configuraciones de entrada

Durante el desarrollo del proyecto se trabajó con tres configuraciones de variables.

### ConfigA: configuración espectral básica

ConfigA corresponde a una configuración preliminar evaluada en una fase inicial del proyecto. Incluye siete canales:

1. B8A  
2. B11  
3. B12  
4. NDSWIR  
5. RatioB12B11  
6. RatioB12B8A  
7. MBMP  

Esta configuración permitió establecer una línea base espectral inicial, pero no forma parte de la comparación final del TFM.

### ConfigB: configuración espectral avanzada

ConfigB es la configuración espectral principal utilizada como referencia en los experimentos finales. Incluye nueve canales:

1. B8A  
2. B11  
3. B12  
4. NDSWIR  
5. RatioB12B11  
6. RatioB12B8A  
7. MBMP  
8. MBMPPlus  
9. DualEnhancementB12B11  

Esta configuración combina bandas crudas Sentinel-2, ratios SWIR, índices normalizados y realces target-reference diseñados para resaltar contrastes asociados a la presencia de metano.

### ConfigC: configuración espectral con viento

ConfigC extiende ConfigB incorporando tres variables meteorológicas:

10. WindSpeed10m  
11. WindDirCos10m  
12. WindDirSin10m  

Estas variables se incorporan como canales constantes por muestra. Su objetivo es aportar contexto físico relacionado con la dirección y velocidad del viento, factores que pueden influir en la forma y desplazamiento de la pluma.

---

## Modelos evaluados

Se evaluaron cuatro familias de modelos de segmentación:

| Modelo | Descripción |
|---|---|
| SimpleUNet | Arquitectura U-Net base |
| EnhancedUNet | Variante convolucional mejorada de U-Net |
| TransformerUNet | U-Net con componentes Transformer |
| TransformerPlus | Variante Transformer ampliada con mayor capacidad contextual |

Todos los modelos generan mapas de probabilidad por píxel, que posteriormente se convierten en máscaras binarias mediante un umbral de decisión.

---

## Diseño experimental

Los experimentos finales comparan:

- ConfigB frente a ConfigC.
- Modelos convolucionales frente a modelos con componentes Transformer.
- Métricas globales de segmentación.
- Comportamiento por muestra.
- Errores de falsos positivos y falsos negativos.
- Influencia del tamaño de la pluma y del fluxrate.
- Calidad visual de las predicciones.

Los principales experimentos incluidos en este repositorio son:

| Run Tag | Contenido |
|---|---|
| 101622 | Experimentos principales con SimpleUNet, EnhancedUNet y TransformerUNet |
| 101840 | Experimentos con TransformerPlus |

---

## Métricas de evaluación

El desempeño de los modelos se evaluó mediante métricas de segmentación a nivel de píxel:

- Dice.
- Intersection over Union.
- Precision.
- Recall.
- Global Dice.
- Global IoU.
- Falsos positivos.
- Falsos negativos.
- Relación de área predicha frente a área real.
- Categorías de calidad de predicción.
- Sensibilidad al umbral de decisión.

---

## Resultados principales

El mejor rendimiento global fue obtenido por **TransformerPlus con ConfigB**, con los siguientes valores aproximados:

| Métrica | Valor |
|---|---:|
| Mean Dice | 0.6368 |
| Mean IoU | 0.5082 |
| Mean Precision | 0.6658 |
| Mean Recall | 0.7001 |
| Global Dice | 0.6779 |
| Global IoU | 0.5127 |
| Mejor umbral | 0.30 |

Los resultados muestran que:

- ConfigB fue la configuración más robusta en términos generales.
- La incorporación de variables de viento en ConfigC no mejoró sistemáticamente los resultados.
- TransformerPlus obtuvo el mejor desempeño global.
- El viento puede ser físicamente relevante, pero su representación disponible en el dataset puede ser demasiado agregada para mejorar una tarea de segmentación a escala de píxel.
- Los principales errores aparecen en plumas débiles, bordes de pluma, fondos heterogéneos y casos de sobresegmentación.

---

## Estructura del repositorio

```text
MethaneProjectTFM/
├── Scripts/
│   ├── Step*.py
│   └── Scripts de preprocesamiento, entrenamiento, evaluación y visualización
│
├── Outputs/
│   ├── Experiments/
│   │   ├── 101622/
│   │   │   ├── ConfigB/
│   │   │   └── ConfigC/
│   │   └── 101840/
│   │       ├── ConfigB/
│   │       └── ConfigC/
│   │
│   └── ResultsChapter_101622_101840/
│       ├── Tables/
│       ├── Figures/
│       └── Resultados finales utilizados en el capítulo de resultados
│
├── README.md
├── .gitignore
├── requirements.txt
└── Archivos de configuración del proyecto