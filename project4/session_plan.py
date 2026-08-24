"""
Builds the fixed sequence of movie IDs a participant will see, sampled
without replacement from the whole catalog so that no movie is ever shown
twice to the same participant (avoids memory/carryover confounds between
the two conditions and between elicitation and validation, see report).
"""

import random

from .data import N_CATALOG, get_movie
from .study_config import PAIRWISE_ELICIT_N, RANKING_ELICIT_N, RANKING_SIZE, VALIDATION_N

N_NEEDED = (
    2               # pairwise practice
    + RANKING_SIZE  # ranking practice
    + 2 * PAIRWISE_ELICIT_N
    + RANKING_ELICIT_N * RANKING_SIZE
    + 2 * VALIDATION_N
)


def build_plan() -> dict:
    pool = random.sample(range(N_CATALOG), N_NEEDED)
    cursor = 0

    def take(n):
        nonlocal cursor
        chunk = pool[cursor:cursor + n]
        cursor += n
        return chunk

    pairwise_practice_movies = take(2)
    ranking_practice_movies = take(RANKING_SIZE)

    pairwise_elicitation = [{"movies": take(2)} for _ in range(PAIRWISE_ELICIT_N)]
    ranking_elicitation = [{"movies": take(RANKING_SIZE)} for _ in range(RANKING_ELICIT_N)]

    validation = [{"movies": take(2)} for _ in range(VALIDATION_N)]

    # instructed target for the practice trials, so the interface can tell
    # the participant exactly what to click
    pairwise_practice_target = pairwise_practice_movies[1]  # "click the second one"
    ranking_practice_target = ranking_practice_movies[3]    # "click this one first"

    return {
        "pairwise": {
            "practice": {"movies": pairwise_practice_movies, "target": pairwise_practice_target},
            "elicitation": pairwise_elicitation,
        },
        "ranking": {
            "practice": {"movies": ranking_practice_movies, "target": ranking_practice_target},
            "elicitation": ranking_elicitation,
        },
        "validation": validation,
    }


def movie_ids_for_spec(spec: dict) -> list[int]:
    return spec["movies"]
