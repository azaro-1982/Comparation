from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# Cargo el modelo para dividir en vectores para poder compararlo
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Cargamos el modelo de inferencia lógica natural (contradicciones)
"""
Cargo SBERT (versión más ligera de BERT), 
que transforma frases en vectores para comparar su significado.

"""

nli_model = pipeline("text-classification", model="roberta-large-mnli")

# Hechos verificados de la investigación por los que se pregunta posteriormente
# a los investigados
hechos_probados = [
    "El sospechoso fue visto en la escena del crimen a las 10 PM.",
    "La víctima murió entre las 10 y las 10:30 PM.",
    "Murio de un disparo"
]
# Declaraciones del sospechoso en respuestas a tres preguntas
version = [
    "Yo estaba en mi casa viendo televisión toda la noche.",
    "No vi a nadie esa noche, estuve solo.",
    "No tengo armas"
]
# Comparación
for i, declaracion in enumerate(version):
    print(f"\nDeclaración {i+1}: {declaracion}")
    for j, hecho in enumerate(hechos_probados):
        # Similitud semántica
        emb = sbert_model.encode([declaracion, hecho], convert_to_tensor=True)
        sim = util.cos_sim(emb[0], emb[1]).item()
        # Inferencia lógica (NLI) sin usar [0]
        nli_result = nli_model(
            {
                "text": hecho,
                "text_pair": declaracion
            }
        )
        # Podemos usar una lista o un diccionario, manejamos ambos
        if isinstance(nli_result, list):
            nli_result = nli_result[0]

        print(f"Se comparan los hechos {j+1}: {hecho}")
        print(f"Similitud entre los hechos: {sim:.4f}")
        print(f"NLI Resultado: {nli_result['label']} (confianza: {nli_result['score']:.2f})")




