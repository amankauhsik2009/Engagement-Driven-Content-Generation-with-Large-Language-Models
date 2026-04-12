import numpy as np
import random
from .propagation import propagate

def search_best_message(G, opinions, resolution=21):
    # Brute-force search over possible message values
    best_score = -1
    best_message = None

    for val in np.linspace(0, 1, resolution):
        activated_count, _, _ = propagate(G, opinions, val)

        if activated_count > best_score:
            best_score = activated_count
            best_message = val

    return best_message, best_score


def lightweight_ppo(G, opinions, iterations=8, candidates_per_iter=6, step_size=0.12):
    # Start with random guess
    current_message = random.random()
    history = []

    for _ in range(iterations):
        candidates = [current_message]

        # Generate nearby candidate values
        for _ in range(candidates_per_iter - 1):
            proposal = current_message + np.random.uniform(-step_size, step_size)
            proposal = min(max(proposal, 0.0), 1.0)
            candidates.append(proposal)

        # Evaluate all candidates
        scores = [propagate(G, opinions, c)[0] for c in candidates]

        # Pick best candidate
        best_idx = int(np.argmax(scores))
        current_message = candidates[best_idx]

        history.append((current_message, scores[best_idx]))

    return current_message, history
