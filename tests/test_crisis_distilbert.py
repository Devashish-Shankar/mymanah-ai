from transformers import pipeline


MODEL_NAME = "dsuram/distilbert-mentalhealth-classifier"


print("Loading DistilBERT crisis model...")

classifier = pipeline(
    "text-classification",
    model=MODEL_NAME,
    top_k=None
)

print("Model loaded successfully.")

print("\nMODEL LABELS:")
print(classifier.model.config.id2label)


texts = [
    "I had a great day today and I'm excited about tomorrow.",

    "I've been feeling extremely sad and overwhelmed lately.",

    "Sometimes I feel like I don't want to wake up anymore.",

    "I sometimes think about ending my life, but I don't have any plan.",

    "I am thinking about killing myself and I have a plan.",

    "My friend told me that they wanted to kill themselves.",

    "Last year I struggled with suicidal thoughts, but I am doing much better now.",

    "My exam results were terrible and I feel like giving up, but I know things will improve.",

    "I want to hurt myself tonight.",

    "I am worried because my brother has been talking about suicide.",

    "If you are struggling with suicidal thoughts, please talk to someone you trust.",

    "Suicide prevention programs can help people who are struggling.",

    "I am not suicidal and I have no intention of hurting myself.",

    "I felt hopeless yesterday, but today I want to keep going and make things better.",

    "I don't want to be alive anymore and I am thinking about killing myself."
]


for i, text in enumerate(texts, start=1):

    print("\n" + "=" * 85)
    print(f"TEST {i}")
    print("=" * 85)

    print("\nTEXT:")
    print(text)

    results = classifier(text)[0]

    results = sorted(
        results,
        key=lambda x: x["score"],
        reverse=True
    )

    print("\nPREDICTIONS:")

    for result in results:
        print(
            f"{result['label']:25s}"
            f"{result['score']:.6f}"
        )