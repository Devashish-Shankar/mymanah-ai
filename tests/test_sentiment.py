from app.models.sentiment import SentimentModel

model = SentimentModel()

texts = [
    "I had an amazing day and I feel very happy.",
    "Today was okay, nothing particularly good or bad.",
    "I am extremely disappointed and exhausted."
]

for text in texts:
    result = model.predict(text)

    print("\nText:", text)
    print("Result:", result)