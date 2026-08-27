"""Shared constants for the elicitation study (Task 3 design + Task 4 interface).

Trial budgets are chosen so the two designs receive a matched amount of
preference information: a ranking of RANKING_SIZE items decomposes (under
Plackett-Luce) into RANKING_SIZE - 1 "effective" sequential comparisons, so
RANKING_ELICIT_N rankings of 10 items give the same effective-comparison
budget as PAIRWISE_ELICIT_N direct pairwise trials:

    RANKING_ELICIT_N * (RANKING_SIZE - 1) == PAIRWISE_ELICIT_N
    2 * 9 == 18
"""

PAIRWISE_ELICIT_N = 18
RANKING_ELICIT_N = 2
RANKING_SIZE = 10
VALIDATION_N = 6

assert RANKING_ELICIT_N * (RANKING_SIZE - 1) == PAIRWISE_ELICIT_N

DESIGN_LABELS = {
    "pairwise": "Pairwise Comparisons",
    "ranking": "Ranking Lists",
}

STAGE_URL_NAME = {
    "background": "project4:background",
    "instructions": "project4:instructions",
    "trials": "project4:trials",
    "questionnaire": "project4:questionnaire",
    "validation": "project4:validation",
    "final": "project4:final",
    "done": "project4:done",
}
