from .llm import generate_post, get_sentiment_score
from .propagation import propagate

def test_multiple_messages(G, opinions, prompts):
    # Try multiple prompts and compare influence spread
    records = []

    for i, prompt in enumerate(prompts, start=1):
        post = generate_post(prompt)

        # Convert generated text into sentiment signal
        sentiment = get_sentiment_score(post)

        # Run propagation
        activated_count, _, _ = propagate(G, opinions, sentiment)

        records.append({
            "Post": i,
            "Prompt": prompt,
            "Sentiment": sentiment,
            "Activated Nodes": activated_count
        })

    return records
