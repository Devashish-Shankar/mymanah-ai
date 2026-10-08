from transformers import pipeline

emotion_pipeline = pipeline(
    "text-classification",
    model="SamLowe/roberta-base-go_emotions",
    top_k=None
)


texts = [
    "I am extremely happy today because I got the job.",
    "I have been feeling very sad and lonely lately.",
    "I am terrified about what might happen.",
    "I am so angry about what happened.",
    "I am nervous and worried about my upcoming exam.",
    "Today was an ordinary day."
]


for text in texts:

    results = emotion_pipeline(text)[0]

    results = sorted(
        results,
        key=lambda x: x["score"],
        reverse=True
    )

    print("\nText:", text)

    for result in results[:5]:
        print(
            f"{result['label']}: "
            f"{result['score']:.4f}"
        )