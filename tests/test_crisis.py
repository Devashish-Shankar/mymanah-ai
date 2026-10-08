from transformers import pipeline


MODEL_NAME = "Akashpaul123/modernbert-crisis-detection"


print("Loading ModernBERT crisis model...")

classifier = pipeline(
    "text-classification",
    model=MODEL_NAME,
    top_k=None
)

print("Model loaded successfully.")

print("\nMODEL LABELS:")
print(classifier.model.config.id2label)


texts = [

    # 1. Normal
    "I had a great day today and I'm excited about tomorrow.",

    # 2. Emotional distress
    "I've been feeling extremely sad and overwhelmed lately.",

    # 3. Passive death ideation
    "Sometimes I feel like I don't want to wake up anymore.",

    # 4. Active suicidal ideation without plan
    "I sometimes think about ending my life, but I don't have any plan.",

    # 5. Explicit suicidal ideation + plan
    "I am thinking about killing myself and I have a plan.",

    # 6. Third person
    "My friend told me that they wanted to kill themselves.",

    # 7. Historical
    "Last year I struggled with suicidal thoughts, but I am doing much better now.",

    # 8. Ambiguous phrase
    "My exam results were terrible and I feel like giving up, but I know things will improve.",

    # 9. Self-harm
    "I want to hurt myself tonight.",

    # 10. Concern about another person
    "I am worried because my brother has been talking about suicide.",

    # 11. Supportive statement
    "If you are struggling with suicidal thoughts, please talk to someone you trust.",

    # 12. Prevention
    "Suicide prevention programs can help people who are struggling.",

    # 13. Explicit denial
    "I am not suicidal and I have no intention of hurting myself.",

    # 14. Protective context
    "I felt hopeless yesterday, but today I want to keep going and make things better.",

    # 15. Explicit current ideation
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