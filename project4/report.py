"""
Generates the Project 4 PDF report (Tasks 1-3: feature representation,
Plackett-Luce extension of Bradley-Terry, and the full user-study design).
Built with ReportLab, following the same visual style as the Project 3 report.
"""

import io
import json
import os
from datetime import date

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, ListFlowable, ListItem, PageBreak,
)

BLUE = colors.HexColor('#1d4ed8')
LIGHT = colors.HexColor('#eff6ff')
GREY = colors.HexColor('#6b7280')
DARK = colors.HexColor('#111827')
GREEN = colors.HexColor('#15803d')
BORDER = colors.HexColor('#e5e7eb')


def _styles():
    base = getSampleStyleSheet()
    return {
        'title': ParagraphStyle('T', parent=base['Title'], fontSize=22, textColor=DARK,
                                 spaceAfter=6, alignment=TA_CENTER, fontName='Helvetica-Bold'),
        'subtitle': ParagraphStyle('Sub', parent=base['Normal'], fontSize=12, textColor=GREY,
                                    spaceAfter=4, alignment=TA_CENTER),
        'section': ParagraphStyle('Sec', parent=base['Heading1'], fontSize=14, textColor=BLUE,
                                   spaceBefore=18, spaceAfter=6, fontName='Helvetica-Bold'),
        'subsection': ParagraphStyle('Sub2', parent=base['Heading2'], fontSize=11, textColor=DARK,
                                      spaceBefore=10, spaceAfter=4, fontName='Helvetica-Bold'),
        'body': ParagraphStyle('Body', parent=base['Normal'], fontSize=10, textColor=DARK,
                                spaceAfter=6, leading=15, alignment=TA_JUSTIFY),
        'bold_label': ParagraphStyle('BL', parent=base['Normal'], fontSize=10, textColor=DARK,
                                      spaceAfter=3, fontName='Helvetica-Bold'),
        'formula': ParagraphStyle('F', parent=base['Normal'], fontSize=10, textColor=DARK,
                                   spaceAfter=6, leading=16, leftIndent=24, fontName='Courier'),
        'bullet': ParagraphStyle('Bul', parent=base['Normal'], fontSize=10, textColor=DARK,
                                  leading=14, spaceAfter=3),
    }


def _table(headers, rows, col_widths=None):
    cell_style = ParagraphStyle('Cell', fontSize=9, leading=12, textColor=DARK)
    header_style = ParagraphStyle('CellHead', fontSize=9, leading=12, textColor=colors.white,
                                   fontName='Helvetica-Bold')

    def _wrap(value, style):
        return Paragraph(value, style) if isinstance(value, str) else value

    data = (
        [[_wrap(h, header_style) for h in headers]]
        + [[_wrap(c, cell_style) for c in row] for row in rows]
    )
    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), BLUE),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT]),
        ('GRID', (0, 0), (-1, -1), 0.4, BORDER),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    return tbl


def _divider():
    return HRFlowable(width='100%', thickness=0.5, color=BORDER, spaceAfter=6, spaceBefore=6)


def _bullets(items, style):
    return ListFlowable(
        [ListItem(Paragraph(it, style), bulletColor=BLUE) for it in items],
        bulletType='bullet', leftIndent=16, spaceAfter=6,
    )


SIMULATION_PATH = os.path.join(os.path.dirname(__file__), "data", "simulation_results.json")


def _load_simulation():
    """Cached results from `python manage.py p4_simulate`.

    Returns None if the file is missing, in which case section 4 is simply
    omitted; the report must never fail to render because an optional
    offline analysis has not been run.
    """
    try:
        with open(SIMULATION_PATH) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def generate_report_pdf() -> bytes:
    from . import data as p4data
    from .study_config import PAIRWISE_ELICIT_N, RANKING_ELICIT_N, RANKING_SIZE, VALIDATION_N
    from .features import genre_vocabulary, RATING_BUCKETS, NUMERIC_RAW

    S = _styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2.5 * cm, rightMargin=2.5 * cm, topMargin=2.5 * cm, bottomMargin=2.5 * cm,
    )
    story = []

    # ── Cover ────────────────────────────────────────────────────────────
    story += [
        Spacer(1, 1.2 * cm),
        Paragraph("Project 4: Preference Elicitation", S['title']),
        Paragraph("Human-Centric Artificial Intelligence &middot; TUHH", S['subtitle']),
        Paragraph(f"IMDB 5000 Movie Dataset &middot; Report generated: {date.today().isoformat()}", S['subtitle']),
        Spacer(1, 0.5 * cm),
        _divider(),
        Spacer(1, 0.3 * cm),
        Paragraph(
            "This report documents the design of a movie-recommender preference-elicitation "
            "system and of a user study comparing two ways of eliciting preferences from a new "
            "user: pairwise comparisons and rank-ordered lists. It covers the feature "
            "representation used for movies (Task 1), the extension of the Bradley-Terry model "
            "to full rankings (Task 2), and a complete, ready-to-run experimental protocol "
            "(Task 3). The corresponding participant-facing interface (Task 4) is available "
            "from the project landing page.",
            S['body'],
        ),
        Spacer(1, 0.3 * cm),
    ]

    # ══════════════════════════════════════════════════════════════════
    # Task 1  - Feature representation
    # ══════════════════════════════════════════════════════════════════
    genres = genre_vocabulary(p4data.CATALOG_DF)
    story += [
        _divider(),
        Paragraph("1. Feature Representation (Task 1)", S['section']),
        Paragraph(
            "Each movie is represented as a feature vector x so that a linear utility "
            "U(x) = w<super>T</super>x can express a user's preferences. The IMDB 5000 dataset "
            "contains no user ratings, so x is built purely from movie metadata: attributes a "
            "person could plausibly judge from a synopsis page before watching the film. "
            f"After removing duplicate listings, the catalog contains {p4data.N_CATALOG} movies "
            f"and {p4data.FEATURE_MATRIX.shape[1]} features.",
            S['body'],
        ),
        Paragraph("Design rationale", S['subsection']),
        _bullets([
            "<b>Genres (multi-hot, %d dims):</b> the strongest and most interpretable signal of "
            "taste. Only genres occurring at least 100 times in the catalog are kept (%s); "
            "rarer tags such as <i>Film-Noir</i> or <i>Game-Show</i> occur fewer than 10 times "
            "and would add noise dimensions no elicitation session could ever pin down." % (
                len(genres), ", ".join(genres[:8]) + ", ..."),
            "<b>Content rating (one-hot, 5 dims):</b> collapsed from ~19 raw MPAA/TV values into "
            "five coarse buckets (family / PG / PG-13 / mature / unrated) &mdash; a reliable proxy "
            "for tone that survives the dataset's inconsistent rating vocabulary.",
            "<b>Continuous attributes (z-scored, %d dims):</b> IMDB score (quality), duration "
            "(pacing), release year (recency), budget, gross revenue, and vote/like counts "
            "(popularity and star power for the director and cast). Heavy-tailed count "
            "variables are log-transformed before standardization; missing values are imputed "
            "with the catalog median so a small number of missing metadata fields never "
            "removes a movie from the study." % len(NUMERIC_RAW),
        ], S['bullet']),
        Paragraph(
            "Dimensionality is intentionally kept moderate (%d total features) rather than, "
            "say, one-hot encoding actor or director identity. w must be estimated from a "
            "handful of interactions in a single sitting (Task 3), so an over-expressive "
            "feature space would make the estimation problem hopeless; the chosen features "
            "trade some expressiveness for identifiability."
            % p4data.FEATURE_MATRIX.shape[1],
            S['body'],
        ),
    ]

    # ══════════════════════════════════════════════════════════════════
    # Task 2  - Bradley-Terry -> Plackett-Luce
    # ══════════════════════════════════════════════════════════════════
    story += [
        _divider(),
        Paragraph("2. From Bradley-Terry to Rankings (Task 2)", S['section']),
        Paragraph(
            "The standard Bradley-Terry model gives the probability that item i is preferred "
            "to item j from their utilities U(x) = w<super>T</super>x:",
            S['body'],
        ),
        Paragraph("P(i &gt; j | w)  =  &sigma;(U(x<sub>i</sub>) &minus; U(x<sub>j</sub>))  "
                   "=  exp(U(x<sub>i</sub>)) / [ exp(U(x<sub>i</sub>)) + exp(U(x<sub>j</sub>)) ]",
                   S['formula']),
        Paragraph(
            "Design 2 of the interface asks a participant to rank ten movies at once, "
            "i<sub>1</sub> &gt; i<sub>2</sub> &gt; &hellip; &gt; i<sub>n</sub>, "
            "which is not a single pairwise comparison. We extend the model using "
            "<b>Luce's choice axiom</b>: a full ranking is generated as a sequence of "
            "\"pick the most preferred item among those not yet placed\" choices, each "
            "obeying the same softmax/Bradley-Terry choice rule among the remaining "
            "candidates. This gives the <b>Plackett-Luce model</b>:",
            S['body'],
        ),
        Paragraph(
            "P(i<sub>1</sub> &gt; i<sub>2</sub> &gt; &hellip; &gt; i<sub>n</sub> | w)  "
            "=  &prod;<sub>k=1..n-1</sub>  "
            "exp(U(x<sub>i<sub>k</sub></sub>))  /  "
            "&sum;<sub>l=k..n</sub> exp(U(x<sub>i<sub>l</sub></sub>))",
            S['formula'],
        ),
        Paragraph(
            "At step k, the numerator is the utility of the item actually placed next, and the "
            "denominator sums over every item still unplaced &mdash; exactly a Bradley-Terry-style "
            "softmax choice among the remaining candidates. Setting n = 2 collapses the product "
            "to its single k = 1 term, which is exactly the pairwise Bradley-Terry probability "
            "above; Plackett-Luce is therefore a strict generalization, not a different model. "
            "The log-likelihood decomposes into n &minus; 1 independent multinomial-logit terms, "
            "which is what makes it tractable to fit by gradient-based maximum likelihood "
            "(implemented in <font face='Courier'>preference_model.py</font>).",
            S['body'],
        ),
        Paragraph(
            "A useful consequence of this decomposition, used to design the study in Task 3: a "
            "ranking of n items carries the same amount of \"effective\" pairwise information as "
            "n &minus; 1 direct pairwise comparisons. A ranking of %d items is therefore "
            "informationally equivalent to %d pairwise comparisons, which is exactly how the two "
            "conditions' trial counts were matched below (%d rankings &times; %d = %d effective "
            "comparisons = %d pairwise trials)." % (
                RANKING_SIZE, RANKING_SIZE - 1, RANKING_ELICIT_N, RANKING_SIZE - 1,
                RANKING_ELICIT_N * (RANKING_SIZE - 1), PAIRWISE_ELICIT_N,
            ),
            S['body'],
        ),
        Paragraph(
            "Fitting w: because both likelihoods depend only on utility <i>differences</i>, w is "
            "unidentifiable without regularization (any scalar multiple of a maximizer is also a "
            "maximizer, and an unregularized fit on a handful of trials badly overfits a "
            f"{p4data.FEATURE_MATRIX.shape[1]}-dimensional w). We therefore fit w by maximum a "
            "posteriori estimation: minimizing the negative log-likelihood plus an L2 penalty "
            "&lambda;||w||&sup2;, equivalent to a zero-mean Gaussian prior on w. This is solved "
            "with L-BFGS using closed-form gradients of both the Bradley-Terry and Plackett-Luce "
            "log-likelihoods.",
            S['body'],
        ),
    ]

    # ══════════════════════════════════════════════════════════════════
    # Task 3  - User study design
    # ══════════════════════════════════════════════════════════════════
    story += [
        _divider(),
        Paragraph("3. User Study Design (Task 3)", S['section']),

        Paragraph("3.1 Research questions and hypotheses", S['subsection']),
        Paragraph(
            "Does the way preferences are elicited &mdash; pairwise comparisons vs. "
            "rank-ordered lists &mdash; change how accurately the resulting utility model "
            "predicts a user's future choices, and how it is experienced?",
            S['body'],
        ),
        _bullets([
            "<b>H1 (primary, accuracy):</b> under an information-matched budget (%d pairwise "
            "trials &asymp; %d rankings &asymp; %d effective comparisons, see Task 2), pairwise "
            "comparisons will yield a preference vector with equal or higher held-out "
            "predictive accuracy than rankings. Rationale: each pairwise judgment is a single, "
            "low-effort decision, whereas ranking research consistently finds that people are "
            "reliable about their top choice(s) but increasingly arbitrary further down a list, "
            "which violates the Plackett-Luce independence assumption for lower ranks and adds "
            "noise the model cannot distinguish from signal." % (
                PAIRWISE_ELICIT_N, RANKING_ELICIT_N, PAIRWISE_ELICIT_N),
            "<b>H2 (cognitive load / efficiency):</b> ranking trials will be rated as more "
            "cognitively demanding than pairwise trials, but will take less total wall-clock "
            "time to deliver the same amount of effective preference information (one ranking "
            "act yields %d effective comparisons at once)." % (RANKING_SIZE - 1),
            "<b>H3 (subjective preference, exploratory):</b> no directional prediction is made "
            "for which interface participants personally prefer overall; this is treated as an "
            "exploratory secondary outcome.",
        ], S['bullet']),

        Paragraph("3.2 Design", S['subsection']),
        Paragraph(
            "<b>Within-subjects, counterbalanced.</b> Every participant completes both designs; "
            "order is alternated deterministically by participant count so exactly half see "
            "pairwise comparisons first and half see rankings first, regardless of drop-out "
            "patterns during recruitment. A within-subject design is chosen because individual "
            "differences in movie taste (and in how consistent a person's preferences are) are "
            "large relative to the expected effect of interface design; comparing each "
            "participant against themselves removes this between-person variance and requires "
            "far fewer participants than a between-subjects design for the same power.",
            S['body'],
        ),
        Paragraph(
            "To prevent contamination between conditions, movies are sampled without replacement "
            "across the <i>entire</i> session: the pairwise condition, the ranking condition, and "
            "the shared validation block each draw from disjoint pools, so no participant ever "
            "re-encounters a movie they already judged. This isolates the held-out validation "
            "accuracy (below) as a test of generalization rather than memory.",
            S['body'],
        ),

        Paragraph("3.3 Materials and trial budget", S['subsection']),
        _table(
            ["Block", "Trials", "Purpose"],
            [
                ["Practice (per condition)", "1", "Familiarize with the UI; doubles as an instructional-manipulation attention check"],
                ["Pairwise elicitation", f"{PAIRWISE_ELICIT_N} pairs", "Fit w_pairwise"],
                ["Ranking elicitation", f"{RANKING_ELICIT_N} rankings of {RANKING_SIZE}", "Fit w_ranking"],
                ["Shared validation (once, after both)", f"{VALIDATION_N} pairs", "Ground truth to score both fitted models"],
                ["Post-condition questionnaire", "2 (one per condition)", "4-item Likert + free text"],
                ["Final comparison", "1", "Forced choice + free-text reason"],
            ],
            col_widths=[5.5 * cm, 4 * cm, 6.2 * cm],
        ),
        Spacer(1, 0.2 * cm),
        Paragraph(
            "Movies for every trial are drawn uniformly at random from the ~4,900-movie catalog, "
            "as specified in the assignment; no adaptive/informative item-selection strategy is "
            "used in this baseline design (a natural extension, noted in the assignment, would "
            "select movies that are maximally informative about w given the responses so far).",
            S['body'],
        ),

        Paragraph("3.4 Outcome measures", S['subsection']),
        _bullets([
            "<b>Objective (primary DV for H1):</b> held-out predictive accuracy of w_pairwise "
            f"and w_ranking on the shared {VALIDATION_N} validation pairs, i.e. the fraction of "
            "the participant's real choices each fitted model predicts correctly, plus the "
            "held-out log-likelihood as a more sensitive secondary measure.",
            "<b>Objective (H2):</b> completion time of the elicitation phase, per condition.",
            "<b>Subjective (1&ndash;7 Likert, per condition):</b> ease of use, cognitive load, "
            "enjoyment, and trust (\"I am confident this method captured my movie taste\"), plus "
            "an optional free-text comment.",
            "<b>Final forced choice:</b> \"Which of the two approaches did you personally prefer "
            "overall?\" with an optional free-text reason (secondary outcome for H3).",
        ], S['bullet']),

        Paragraph("3.5 Statistical analysis plan", S['subsection']),
        _bullets([
            "<b>H1:</b> Wilcoxon signed-rank test on paired per-participant validation accuracy "
            "(pairwise vs. ranking); accuracy over only 6 held-out items is coarse and "
            "non-normal, so a non-parametric paired test is preferred over a t-test. As a "
            "robustness check, a mixed-effects logistic regression is fit on trial-level "
            "correct/incorrect outcomes with design as a fixed effect, participant as a random "
            "intercept, and condition order as a covariate &mdash; order should not interact "
            "significantly with the design effect if counterbalancing worked as intended.",
            "<b>H2:</b> paired t-test (or Wilcoxon if the normality assumption is violated) on "
            "the cognitive-load Likert item and on completion time, pairwise vs. ranking.",
            "<b>H3:</b> descriptive summary of the forced-choice split (with an exploratory "
            "two-sided binomial test against 50/50) plus a thematic summary of free-text reasons.",
            "Significance threshold &alpha; = 0.05, two-sided, with Holm-Bonferroni correction "
            "across the family of pre-specified tests (H1 accuracy, H1 log-likelihood, H2 load, "
            "H2 time).",
            "<b>Exclusions:</b> sessions where a practice/attention check was answered "
            "incorrectly (exported directly as the <font face='Courier'>is_correct</font> "
            "column of the practice-phase trial rows), or where median response time falls "
            "below 300&nbsp;ms (indicating click-through rather than genuine judgments), are "
            "flagged; all analyses are run both with and without flagged sessions as a "
            "sensitivity check.",
        ], S['bullet']),

        Paragraph("3.6 Sample size", S['subsection']),
        Paragraph(
            "For a two-sided paired test at &alpha; = 0.05 and power = 0.80, detecting a medium "
            "effect size (Cohen's d<sub>z</sub> = 0.5) requires n &asymp; "
            "((z<sub>&alpha;/2</sub> + z<sub>&beta;</sub>) / d<sub>z</sub>)&sup2; = "
            "((1.96 + 0.84) / 0.5)&sup2; &asymp; 31.4, i.e. at least 32 completed sessions. "
            "Allowing for an expected 15&ndash;20% exclusion/drop-out rate typical of online "
            "studies, we recommend recruiting <b>N = 42</b> participants who complete the study "
            "in full. This figure is revised in &sect;4.2, where a simulation-based power "
            "analysis estimates the effect size directly from the outcome measure this study "
            "actually records, rather than assuming a medium effect.",
            S['body'],
        ),

        Paragraph("3.7 Recruitment", S['subsection']),
        _bullets([
            "<b>Platform:</b> Prolific (or an equivalent vetted participant pool / university "
            "subject pool), for built-in prescreening, enforced fair compensation, and "
            "duplicate-participation controls.",
            "<b>Eligibility:</b> fluent in English (interface text and movie metadata are "
            "English-only), age 18+, normal or corrected-to-normal vision, self-reported general "
            "interest in movies (single screener question), and access to a laptop/desktop "
            "&mdash; the ranking task asks participants to click ten movie cards in order, "
            "which works on a touchscreen but is comfortably faster with a mouse or trackpad; "
            "this is disclosed on the consent screen.",
            "<b>Compensation:</b> paid pro-rata for an estimated 10-minute session at or above "
            "the platform's minimum hourly-rate guidance (roughly &euro;1.50&ndash;&euro;2.00 "
            "per completed session on Prolific).",
            "<b>Anonymity and data protection:</b> no directly identifying information is "
            "collected. If deployed on Prolific, the participant's platform ID is stored only to "
            "process payment and is kept separate from substantive responses; movie choices, "
            "questionnaire answers, and timing are otherwise anonymous, satisfying GDPR "
            "data-minimization requirements for a study run within the EU.",
        ], S['bullet']),

        Paragraph("3.8 Procedure (participant-facing)", S['subsection']),
        Paragraph(
            "The steps below mirror the implemented interface exactly, so the study can be "
            "launched with no further engineering changes:",
            S['body'],
        ),
        ListFlowable([
            ListItem(Paragraph(t, S['bullet'])) for t in [
                "Landing page &rarr; \"Start the Study\".",
                "Consent screen: purpose, what is collected, voluntary participation and right "
                "to withdraw at any time without penalty, estimated duration, researcher contact "
                "details; participant must check \"I agree\" to proceed.",
                "Silent, counterbalanced random assignment to condition order.",
                "Brief background survey (age bracket, movie-watching frequency, familiarity "
                "with recommender features) &mdash; asked once, used only to describe the "
                "cohort in the write-up of results, never linked back to individual preference "
                "judgments in the analysis.",
                "Instructions for condition 1, followed by one practice trial.",
                f"Elicitation trials for condition 1 ({PAIRWISE_ELICIT_N} pairs or "
                f"{RANKING_ELICIT_N}&times;{RANKING_SIZE}-item rankings, depending on assignment).",
                "Short questionnaire for condition 1 (4 Likert items + optional comment).",
                "Instructions and practice trial for condition 2 (the other design).",
                "Elicitation trials for condition 2.",
                "Questionnaire for condition 2.",
                f"{VALIDATION_N} shared held-out validation trials (pairwise, on movies not "
                "seen before) &mdash; collected once and used afterward to silently score both "
                "conditions' fitted models.",
                "Final forced-choice comparison question with an optional free-text reason.",
                "Debrief screen: full disclosure of the study's purpose, a completion code for "
                "payment/credit, and a small \"reveal\" of which fitted model best predicted the "
                "participant's own held-out choices.",
            ]
        ], bulletType='1', leftIndent=18, spaceAfter=8),

        Paragraph("3.9 Steps to actually run the study", S['subsection']),
        ListFlowable([
            ListItem(Paragraph(t, S['bullet'])) for t in [
                "Obtain research-ethics approval / consult the TUHH data protection officer "
                "before recruiting real participants, even for pseudonymous data.",
                "Pilot the interface internally (5&ndash;10 colleagues) to check timing, "
                "instruction clarity, and bugs; adjust wording (not trial counts or measures) "
                "as needed.",
                "Freeze the design after piloting; any further change should not enter the main "
                "data collection, to preserve the confirmatory value of H1/H2.",
                "Optionally pre-register the hypotheses and analysis plan (e.g. on OSF).",
                "Deploy on Prolific with the eligibility screener and compensation from &sect;3.7 "
                "&mdash; the landing/consent/study flow implemented here requires no further "
                "engineering.",
                "Monitor recruitment and data quality live via the Django admin, which lists "
                "every participant, trial, and questionnaire response and offers a CSV export.",
                "Stop recruitment once N = 42 valid, completed sessions are reached.",
                "Export the data and run the analysis plan from &sect;3.5, including the "
                "sensitivity check that excludes flagged low-attention sessions.",
            ]
        ], bulletType='1', leftIndent=18, spaceAfter=8),
    ]

    # ══════════════════════════════════════════════════════════════════
    # Section 4  - Extension: adaptive selection + simulation-based power
    # ══════════════════════════════════════════════════════════════════
    sim = _load_simulation()
    if sim:
        sel = sim["selection"]
        fats = sorted(sim["power"].items())
        null_case, worst_case = fats[0][1], fats[-1][1]
        story += [
            PageBreak(),
            Paragraph("4. Extension: Adaptive Selection and a Simulated Power Analysis", S['section']),
            Paragraph(
                "The assignment notes that investigating more informative or adaptive item "
                "selection would be an interesting extension. This section reports that "
                "extension, together with a simulation-based re-derivation of the sample size "
                "in &sect;3.6. Both are offline analyses: the study in Task 3 uses uniformly "
                "random selection exactly as specified, and nothing here runs during a "
                "participant session. Results are regenerated with "
                "<font face='Courier'>python manage.py p4_simulate</font>.",
                S['body'],
            ),

            Paragraph("4.1 An adaptive selection rule", S['subsection']),
            Paragraph(
                "After k comparisons we hold a MAP estimate w<sub>MAP</sub> and a Laplace "
                "approximation to its posterior covariance &Sigma; = H<super>&minus;1</super>, "
                "where H = 2&lambda;I + &sum;<sub>k</sub> p<sub>k</sub>(1&minus;p<sub>k</sub>) "
                "d<sub>k</sub>d<sub>k</sub><super>T</super> is the Hessian of the regularized "
                "negative log-likelihood and d<sub>k</sub> = x<sub>winner</sub> &minus; "
                "x<sub>loser</sub>. For a candidate pair with d = x<sub>i</sub> &minus; "
                "x<sub>j</sub>, the expected information from observing its outcome is:",
                S['body'],
            ),
            Paragraph(
                "score(i, j)  =  p (1 &minus; p) &middot; d<super>T</super> &Sigma; d,     "
                "p = &sigma;(d<super>T</super> w<sub>MAP</sub>)",
                S['formula'],
            ),
            Paragraph(
                "The two factors do different jobs. p(1&minus;p) peaks at p = 0.5, so a "
                "comparison whose outcome we can already predict is worth little however novel "
                "the movies are. d<super>T</super>&Sigma;d is large when the pair probes a "
                "direction of w we remain uncertain about, which is what prevents the rule from "
                "serving near-identical movies forever: a coin-flip pair along an axis we have "
                "already pinned down scores low. This is the standard Bayesian D-optimal design "
                "criterion for a logistic model, specialized to Bradley-Terry. The cold start "
                "needs no special case: at w = 0 every pair has p = 0.5 and &Sigma; is "
                "isotropic, so the score reduces to ||d||&sup2; and the rule opens with the "
                "sharpest available feature contrast.",
                S['body'],
            ),
            Paragraph(
                "Scoring all ~12 million possible pairs per trial is unnecessary and too slow "
                "for a web request, so we greedily optimize over a random candidate subsample "
                "of a few hundred pairs, which costs a few milliseconds per trial.",
                S['body'],
            ),
            Paragraph(
                "Simulating %d synthetic participants over %d pairwise trials, adaptive "
                "selection reaches the held-out accuracy that random selection attains after "
                "all %d trials in roughly %s trials, and ends %.1f percentage points higher "
                "(%.3f vs %.3f)." % (
                    sel["n_participants"], sel["n_trials"], sel["n_trials"],
                    sel["trials_for_adaptive_to_match_random"],
                    100 * (sel["adaptive"][-1] - sel["random"][-1]),
                    sel["adaptive"][-1], sel["random"][-1],
                ),
                S['body'],
            ),
            _table(
                ["Trials completed", "Random selection", "Adaptive selection"],
                [[str(k), f"{sel['random'][k-1]:.3f}", f"{sel['adaptive'][k-1]:.3f}"]
                 for k in (1, 3, 5, 10, 14, sel["n_trials"])],
                col_widths=[5 * cm, 5.3 * cm, 5.3 * cm],
            ),
            Spacer(1, 0.2 * cm),
            Paragraph(
                "<b>How much to believe this.</b> A synthetic participant obeys Bradley-Terry "
                "exactly; real people do not. The assumption flatters adaptive selection more "
                "than it flatters random selection, because the rule deliberately seeks out "
                "near-tied comparisons &mdash; which are precisely the comparisons on which "
                "real people are least self-consistent. The gain above should therefore be read "
                "as an upper bound, and the obvious next study is whether it survives contact "
                "with human participants. That is also why the adaptive rule is kept out of the "
                "Task 3 comparison: applying it to one condition and not the other would "
                "confound selection strategy with elicitation format.",
                S['body'],
            ),

            Paragraph("4.2 Re-deriving the sample size by simulation", S['subsection']),
            Paragraph(
                "The calculation in &sect;3.6 assumes a medium effect (d<sub>z</sub> = 0.5) and "
                "yields N &asymp; 32. That number is only as good as the assumption, and the "
                "assumption was made without reference to the DV we actually measure: accuracy "
                f"over {VALIDATION_N} held-out pairs can take only {VALIDATION_N + 1} distinct "
                "values. We therefore estimated the effect size directly, by simulating whole "
                "sessions at the study's real trial budgets.",
                S['body'],
            ),
            Paragraph(
                "H1's stated rationale is that people are reliable about their top choices but "
                "increasingly arbitrary further down a list. We model exactly that: at step k of "
                "a ranking the simulated participant chooses from softmax(u / T<sub>k</sub>) "
                "with T<sub>k</sub> = 1 + &phi;k, so &phi; = 0 is an exact Plackett-Luce ranker "
                "and larger &phi; degrades the tail of the ranking while leaving the top intact. "
                "Each simulated participant is run at every &phi;, reusing the same taste "
                "vector, the same pairwise session and the same held-out set, so comparisons "
                "across &phi; are paired.",
                S['body'],
            ),
            _table(
                ["Ranking noise &phi;", "Acc. pairwise", "Acc. ranking",
                 "d<sub>z</sub> (accuracy)", "d<sub>z</sub> (log-lik.)"],
                [[f"{fat}", f"{v['mean_pairwise_accuracy']:.3f}",
                  f"{v['mean_ranking_accuracy']:.3f}",
                  f"{v['dvs']['accuracy']['effect_size_dz']:+.3f}",
                  f"{v['dvs']['log_likelihood']['effect_size_dz']:+.3f}"]
                 for fat, v in fats],
                col_widths=[3.4 * cm, 3.1 * cm, 3.1 * cm, 3 * cm, 3 * cm],
            ),
            Spacer(1, 0.2 * cm),
            _bullets([
                "<b>The procedure is calibrated.</b> At &phi; = 0 no true difference exists and "
                "the test rejects at roughly the nominal 5%, confirming the bootstrap-Wilcoxon "
                "pipeline is not itself generating false positives.",
                "<b>At &phi; = 0 the ranking condition is slightly ahead</b> "
                "(%.3f vs %.3f), which corrects a claim made in Task 2. There we matched the "
                "two budgets on <i>choice events</i> (2 &times; 9 = 18) and called them "
                "informationally equivalent; they are not. Each Plackett-Luce choice at step k "
                "is made among n &minus; k + 1 alternatives, so the first pick of a ten-item "
                "ranking eliminates nine rivals where a pairwise judgment eliminates one, and a "
                "completed ranking of ten implies all 45 pairwise relations among those items "
                "(though not as independent observations). Matching on choice events therefore "
                "hands the ranking condition slightly more information, which makes H1 a "
                "conservative test rather than a biased one." % (
                    null_case["mean_ranking_accuracy"], null_case["mean_pairwise_accuracy"]),
                "<b>The realistic effect is far smaller than assumed.</b> Even at substantial "
                "ranking noise the effect size on the accuracy DV is around d<sub>z</sub> = "
                "%.2f, not 0.5. N = 42 is therefore badly underpowered for H1 as specified." % (
                    worst_case["dvs"]["accuracy"]["effect_size_dz"]),
                "<b>Log-likelihood is the better primary DV,</b> as &sect;3.4 already suspected. "
                "On identical simulated sessions it roughly doubles the effect size "
                "(d<sub>z</sub> = %.2f vs %.2f) because it is continuous and rewards being "
                "right for the right reason, rather than collapsing every session onto %d "
                "possible values." % (
                    worst_case["dvs"]["log_likelihood"]["effect_size_dz"],
                    worst_case["dvs"]["accuracy"]["effect_size_dz"],
                    VALIDATION_N + 1),
            ], S['bullet']),
            Paragraph(
                "<b>Revised recommendation.</b> Promote held-out log-likelihood to the primary "
                "DV for H1, keeping accuracy as the interpretable secondary; raise "
                f"VALIDATION_N above {VALIDATION_N}, since the held-out set is the cheapest "
                "source of precision left in the design; and treat N = 42 as a floor for the "
                "subjective measures (H2, H3) rather than as adequate for H1, which on these "
                "estimates needs a sample in the low hundreds. Detecting a genuinely small "
                "difference between two reasonable elicitation methods is simply expensive "
                "&mdash; which is itself worth knowing before running the study rather than "
                "after.",
                S['body'],
            ),
        ]

    doc.build(story)
    return buf.getvalue()