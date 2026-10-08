from app.services.crisis_service import CrisisService


service = CrisisService()


texts = [
    "I had a great day today.",

    "I've been feeling very hopeless lately.",

    "Sometimes I don't want to wake up anymore.",

    "I am thinking about ending my life, but I don't have a plan.",

    "I am thinking about killing myself and I have a plan.",

    "I want to hurt myself tonight.",

    "My friend told me that they wanted to kill themselves.",

    "Last year I struggled with suicidal thoughts, but I am doing much better now.",

    "I am not suicidal and I have no intention of hurting myself.",
]


for i, text in enumerate(texts, start=1):

    print("\n" + "=" * 70)
    print(f"TEST {i}")
    print("=" * 70)

    print("TEXT:")
    print(text)

    result = service.predict(text)

    print("\nRESULT:")
    print(result)