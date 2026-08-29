"""Regenerate the cached simulation results used by the Task 4 report.

    python manage.py p4_simulate            # full run (a few minutes)
    python manage.py p4_simulate --quick    # smaller run, for a smoke test

Results are written to project4/data/simulation_results.json. The report
reads that file rather than recomputing, so downloading the PDF stays fast.
"""

import json
import os
import time

from django.core.management.base import BaseCommand

from project4 import data as p4data
from project4.simulation import compare_selection_strategies, power_analysis

OUT_PATH = os.path.join(os.path.dirname(p4data.__file__), "data", "simulation_results.json")


class Command(BaseCommand):
    help = "Run the offline simulation study for project 4 and cache the results."

    def add_arguments(self, parser):
        parser.add_argument("--quick", action="store_true",
                            help="Small run for testing; not for the report.")

    def handle(self, *args, **options):
        quick = options["quick"]
        X = p4data.FEATURE_MATRIX.astype("float64")

        self.stdout.write("(A) adaptive vs random selection ...")
        t0 = time.time()
        selection = compare_selection_strategies(
            X, n_participants=5 if quick else 40,
        )
        self.stdout.write(self.style.SUCCESS(f"    done in {time.time() - t0:.0f}s"))

        self.stdout.write("(B) simulation-based power analysis ...")
        t0 = time.time()
        power = power_analysis(
            X,
            pool_size=30 if quick else 400,
            n_bootstrap=100 if quick else 600,
        )
        self.stdout.write(self.style.SUCCESS(f"    done in {time.time() - t0:.0f}s"))

        payload = {
            "quick": quick,
            "n_catalog": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "selection": selection,
            "power": power,
        }
        with open(OUT_PATH, "w") as fh:
            json.dump(payload, fh, indent=2)
        self.stdout.write(self.style.SUCCESS(f"wrote {OUT_PATH}"))