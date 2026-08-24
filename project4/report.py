"""
Generates the Project 4 PDF report (Tasks 1-3: feature representation,
Plackett-Luce extension of Bradley-Terry, and the full user-study design).
Built with ReportLab, following the same visual style as the Project 3 report.
"""

import io
from datetime import date

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, ListFlowable, ListItem,
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
    # Task 1 — Feature representation
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
    # Task 2 — Bradley-Terry -> Plackett-Luce
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
    # Task 3 — User study design
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
            "in full.",
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
            "&mdash; the ranking task uses drag-and-drop, which is unreliable on some mobile "
            "touchscreens; this is disclosed on the consent screen.",
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

    doc.build(story)
    return buf.getvalue()
