"""
Generate the Project 3 PDF report using ReportLab.

The report is generated dynamically from the module-level stats computed at
server startup (classifier accuracy, expert profiles, deferral results, AL curves).
"""

import io
from datetime import date

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether,
)


# ── Color palette ──────────────────────────────────────────────────────────────
BLUE   = colors.HexColor('#1d4ed8')
LIGHT  = colors.HexColor('#eff6ff')
GREY   = colors.HexColor('#6b7280')
DARK   = colors.HexColor('#111827')
GREEN  = colors.HexColor('#15803d')
AMBER  = colors.HexColor('#b45309')
RED    = colors.HexColor('#b91c1c')
BORDER = colors.HexColor('#e5e7eb')


def _styles():
    base = getSampleStyleSheet()

    title = ParagraphStyle(
        'ReportTitle',
        parent=base['Title'],
        fontSize=22, textColor=DARK,
        spaceAfter=6, spaceBefore=0,
        alignment=TA_CENTER, fontName='Helvetica-Bold',
    )
    subtitle = ParagraphStyle(
        'Subtitle',
        parent=base['Normal'],
        fontSize=12, textColor=GREY,
        spaceAfter=4, alignment=TA_CENTER,
    )
    section = ParagraphStyle(
        'SectionHead',
        parent=base['Heading1'],
        fontSize=13, textColor=BLUE,
        spaceBefore=18, spaceAfter=6,
        fontName='Helvetica-Bold', borderPad=0,
    )
    body = ParagraphStyle(
        'Body',
        parent=base['Normal'],
        fontSize=10, textColor=DARK,
        spaceAfter=6, leading=15,
        alignment=TA_JUSTIFY,
    )
    note = ParagraphStyle(
        'Note',
        parent=base['Normal'],
        fontSize=9, textColor=GREY,
        spaceAfter=4, leading=13,
        leftIndent=12,
    )
    bold_label = ParagraphStyle(
        'BoldLabel',
        parent=base['Normal'],
        fontSize=10, textColor=DARK,
        spaceAfter=3, fontName='Helvetica-Bold',
    )
    formula = ParagraphStyle(
        'Formula',
        parent=base['Normal'],
        fontSize=9, textColor=DARK,
        spaceAfter=4, leading=14,
        leftIndent=24, fontName='Courier',
    )
    return {
        'title': title, 'subtitle': subtitle, 'section': section,
        'body': body, 'note': note, 'bold_label': bold_label, 'formula': formula,
    }


def _table(headers, rows, col_widths=None, highlight_last=False):
    """Build a styled table with a blue header row."""
    data = [headers] + rows
    tbl = Table(data, colWidths=col_widths, repeatRows=1)

    style = [
        ('BACKGROUND', (0, 0), (-1, 0), BLUE),
        ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
        ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0, 0), (-1, -1), 9),
        ('ALIGN',      (0, 0), (-1, -1), 'CENTER'),
        ('ALIGN',      (0, 1), (0, -1), 'LEFT'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT]),
        ('GRID',       (0, 0), (-1, -1), 0.4, BORDER),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]

    if highlight_last and len(rows) > 0:
        last = len(rows)
        style += [
            ('BACKGROUND', (0, last), (-1, last), colors.HexColor('#dcfce7')),
            ('FONTNAME',   (0, last), (-1, last), 'Helvetica-Bold'),
            ('TEXTCOLOR',  (0, last), (-1, last), GREEN),
        ]

    tbl.setStyle(TableStyle(style))
    return tbl


def _divider():
    return HRFlowable(width='100%', thickness=0.5, color=BORDER, spaceAfter=6, spaceBefore=6)


def generate_report_pdf() -> bytes:
    """Generate the full project report as PDF bytes."""

    # ── Import live stats at call time ────────────────────────────────────────
    from .classifier import test_acc, conf_matrix, get_classifier_stats
    from .experts import (
        sports_per_class, tech_per_class,
        sports_acc, tech_acc, get_expert_stats,
    )
    from .deferral import (
        ai_only_acc, optimal_team_acc, optimal_coverage,
        optimal_deferral_rate, best_expert_only_acc,
        sports_only_acc, tech_only_acc,
    )
    from .active_learning import results as al_results, oracle_acc, N_POOL, N_EVAL, N_QUERIES
    from .data import LABEL_NAMES

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2.5 * cm, rightMargin=2.5 * cm,
        topMargin=2.5 * cm, bottomMargin=2.5 * cm,
    )

    S = _styles()
    story = []

    # ── Cover ─────────────────────────────────────────────────────────────────
    story += [
        Spacer(1, 1.5 * cm),
        Paragraph("Project 3: Active Learning for Learning-to-Defer", S['title']),
        Paragraph("Human-Centric Artificial Intelligence · TUHH", S['subtitle']),
        Paragraph(f"AG News Dataset  ·  Report generated: {date.today().isoformat()}", S['subtitle']),
        Spacer(1, 0.6 * cm),
        _divider(),
        Spacer(1, 0.4 * cm),
        Paragraph(
            "This report describes the implementation and results of Project 3. "
            "The goal is to build a human-AI team that learns <i>when to defer</i> a "
            "classification decision to a human expert, using active learning to "
            "discover the expert's competence profile with minimal queries. "
            "The project covers: (1) a baseline text classifier, (2) simulated experts "
            "with domain-specific strengths, (3) Bayes-optimal learning-to-defer, "
            "(4) active learning for expert competence discovery, and "
            "(5) an optional interactive interface.",
            S['body'],
        ),
        Spacer(1, 0.4 * cm),
    ]

    # ── Section 1: Task 1 ─────────────────────────────────────────────────────
    story += [
        _divider(),
        Paragraph("Section 1 — Baseline Classifier (Task 1)", S['section']),
        Paragraph(
            "We train a text classification model on the AG News dataset "
            "(120,000 training articles, 7,600 test articles, 4 categories: "
            "World, Sports, Business, Sci/Tech). "
            "Features are extracted with TF-IDF using unigrams and bigrams "
            "(50,000 features). The model is Logistic Regression with C=5 "
            "and a maximum of 1,000 iterations. Training runs once at server "
            "start-up so all requests are served without re-training.",
            S['body'],
        ),
    ]

    # Compute per-class accuracy from confusion matrix (diagonal / row sum)
    import numpy as np
    per_class_accs = {}
    for i, label in enumerate(LABEL_NAMES):
        row_sum = conf_matrix[i].sum()
        per_class_accs[label] = round(conf_matrix[i][i] / row_sum * 100, 2) if row_sum > 0 else 0.0

    acc_rows = []
    for label in LABEL_NAMES:
        acc_rows.append([label, f"{per_class_accs[label]}%"])
    acc_rows.append(['<b>Overall</b>', f'<b>{round(test_acc * 100, 2)}%</b>'])

    story += [
        Spacer(1, 0.3 * cm),
        Paragraph("<b>Results:</b>", S['bold_label']),
        _table(
            ['Category', 'Test Accuracy'],
            [[Paragraph(r[0], S['body']), r[1]] for r in acc_rows],
            col_widths=[9 * cm, 6 * cm],
        ),
        Spacer(1, 0.3 * cm),
        Paragraph(
            f"The classifier achieves <b>{round(test_acc * 100, 2)}%</b> overall test accuracy. "
            "This strong baseline means the human-AI team must demonstrate clear benefit "
            "to justify deferral — the L2D system needs to exceed this figure.",
            S['body'],
        ),
    ]

    # ── Section 2: Task 2 ─────────────────────────────────────────────────────
    story += [
        _divider(),
        Paragraph("Section 2 — Simulated Experts (Task 2)", S['section']),
        Paragraph(
            "We design two simulated experts, each specialising in a specific news category. "
            "The key design constraint is that at least one expert must outperform the "
            "classifier on their specialty class to make deferral genuinely beneficial.",
            S['body'],
        ),
        Spacer(1, 0.2 * cm),
        Paragraph("<b>Design choice: accuracy-based simulation</b>", S['bold_label']),
        Paragraph(
            "A keyword-based approach was considered first but rejected: the TF-IDF+LR "
            "classifier already achieves ~97.7% accuracy on Sports articles, so a keyword "
            "expert matching a fixed list of terms cannot reliably exceed this. Instead, "
            "each expert is simulated with an explicit per-class accuracy profile "
            "(PER_CLASS_ACCURACY dict). When the expert is wrong they predict the "
            "second-most-likely class according to the classifier — a realistic error pattern.",
            S['body'],
        ),
        Spacer(1, 0.3 * cm),
        Paragraph("<b>Expert profiles and test-set performance:</b>", S['bold_label']),
    ]

    expert_stats = get_expert_stats()
    expert_rows = []
    for expert in expert_stats['experts']:
        for label in LABEL_NAMES:
            acc = expert['per_class_acc'].get(label, 0.0)
            expert_rows.append([expert['name'], label, f"{acc}%"])

    story += [
        _table(
            ['Expert', 'Category', 'Per-class Accuracy'],
            expert_rows,
            col_widths=[5.5 * cm, 5.5 * cm, 5 * cm],
        ),
        Spacer(1, 0.3 * cm),
    ]

    overall_rows = [
        ['Sports Expert', f"{round(sports_acc * 100, 2)}%",
         "Specialist (Sports ≥97%); weak on other classes (~55%)"],
        ['Sci/Tech Expert', f"{round(tech_acc * 100, 2)}%",
         "Specialist (Sci/Tech ≥95%); weak on other classes (~55%)"],
    ]
    story += [
        _table(
            ['Expert', 'Overall Accuracy', 'Profile Summary'],
            overall_rows,
            col_widths=[4 * cm, 3.5 * cm, 8.5 * cm],
        ),
        Spacer(1, 0.2 * cm),
        Paragraph(
            "The L2D system uses the <i>best expert per predicted class</i>, "
            "so on Sports articles it defers to the Sports Expert, on Sci/Tech articles "
            "to the Sci/Tech Expert.",
            S['note'],
        ),
    ]

    # ── Section 3: Task 3 ─────────────────────────────────────────────────────
    story += [
        _divider(),
        Paragraph("Section 3 — Learning-to-Defer (Task 3)", S['section']),
        Paragraph(
            "We implement the Bayes-optimal learning-to-defer rule: defer to the expert "
            "whenever the classifier's uncertainty exceeds the expert's expected error. "
            "When classifier labels AND expert labels are both available on the test set, "
            "this rule is directly applicable.",
            S['body'],
        ),
        Spacer(1, 0.2 * cm),
        Paragraph("<b>Deferral rule:</b>", S['bold_label']),
        Paragraph(
            "Defer  if  1 − max_k P(y=k|x)  >  α × (1 − P(expert correct | x))",
            S['formula'],
        ),
        Paragraph(
            "where  P(expert correct | x) = Σ_k P(y=k|x) × accuracy_{best_expert, k}",
            S['formula'],
        ),
        Spacer(1, 0.2 * cm),
        Paragraph(
            "At α=1 this is the Bayes-optimal threshold. Sweeping α ∈ [0, 4] "
            "generates the full accuracy-vs-coverage curve: α=0 always defers "
            "(expert-only), α→∞ never defers (AI-only).",
            S['body'],
        ),
        Spacer(1, 0.3 * cm),
        Paragraph("<b>Results at α=1 (Bayes-optimal operating point):</b>", S['bold_label']),
    ]

    deferral_rows = [
        ['AI Only (classifier)',        f"{round(ai_only_acc, 2)}%",   '100%',   '—'],
        ['Sports Expert only',          f"{round(sports_only_acc, 2)}%", '0%',   '100%'],
        ['Sci/Tech Expert only',        f"{round(tech_only_acc, 2)}%",   '0%',   '100%'],
        ['Best Expert only (oracle)',   f"{round(best_expert_only_acc, 2)}%", '0%', '100%'],
        ['L2D Team (α=1)',              f"{round(optimal_team_acc, 2)}%",
         f"{round(optimal_coverage * 100, 1)}%",
         f"{round(optimal_deferral_rate * 100, 1)}%"],
    ]
    story += [
        _table(
            ['System', 'Team Accuracy', 'AI Coverage', 'Deferral Rate'],
            deferral_rows,
            col_widths=[6 * cm, 3.5 * cm, 3 * cm, 3.5 * cm],
            highlight_last=True,
        ),
        Spacer(1, 0.3 * cm),
        Paragraph(
            f"The L2D team achieves <b>{round(optimal_team_acc, 2)}%</b> accuracy, "
            f"surpassing the AI-alone baseline of {round(ai_only_acc, 2)}% "
            f"by {round(optimal_team_acc - ai_only_acc, 2)} percentage points. "
            f"The system defers {round(optimal_deferral_rate * 100, 1)}% of test instances "
            f"to the best expert, retaining full AI coverage on the remaining "
            f"{round(optimal_coverage * 100, 1)}%. "
            "The accuracy-vs-coverage curve (visible on the web interface) confirms "
            "that the L2D system consistently outperforms AI-alone across a wide "
            "range of coverage fractions.",
            S['body'],
        ),
    ]

    # ── Section 4: Task 4 ─────────────────────────────────────────────────────
    story += [
        _divider(),
        Paragraph("Section 4 — Active Learning for Expert Competence (Task 4)", S['section']),
        Paragraph(
            "In the active learning setting, no expert labels are available at the start. "
            "The system must query the expert on selected instances to learn their "
            "per-class competence, then use that learned profile to build a deferral policy. "
            "The challenge is to learn the competence profile efficiently with few queries.",
            S['body'],
        ),
        Spacer(1, 0.2 * cm),
        Paragraph("<b>Experimental setup:</b>", S['bold_label']),
    ]

    setup_rows = [
        ['Unlabeled pool',         f"{N_POOL:,} test instances (random subset)"],
        ['Held-out evaluation set', f"{N_EVAL:,} test instances"],
        ['Query budget',            f"{N_QUERIES} queries total"],
        ['Competence model',        'Laplace smoothing: prior = 0.5 (10 pseudo-counts per class)'],
        ['Evaluation cadence',      'Team accuracy on eval set every 10 queries'],
    ]
    story += [
        _table(
            ['Parameter', 'Value'],
            setup_rows,
            col_widths=[5.5 * cm, 10.5 * cm],
        ),
        Spacer(1, 0.3 * cm),
        Paragraph("<b>Query strategies compared:</b>", S['bold_label']),
    ]

    strategy_rows = [
        ['Random',           'u(x) = Uniform random',
         'Naive baseline; provides balanced class coverage'],
        ['Least Confidence', 'u(x) = 1 − max_k P(k|x)',
         'Queries where classifier has lowest peak confidence'],
        ['Margin Sampling',  'u(x) = 1 − (P(ŷ₁|x) − P(ŷ₂|x))',
         'Queries where top-2 classes are most ambiguous'],
        ['Entropy',          'u(x) = −Σ_k P(k|x) log P(k|x)',
         'Queries with highest full-distribution uncertainty'],
    ]
    story += [
        _table(
            ['Strategy', 'Utility Function', 'Intuition'],
            strategy_rows,
            col_widths=[3.5 * cm, 6 * cm, 6.5 * cm],
        ),
        Spacer(1, 0.3 * cm),
        Paragraph("<b>Results after 200 queries:</b>", S['bold_label']),
    ]

    result_rows = []
    for name, curve in al_results.items():
        if curve:
            final_acc = curve[-1][1]
            result_rows.append([name, f"{final_acc:.2f}%"])
    result_rows.append(['Oracle (true competence)', f"{oracle_acc:.2f}%"])
    story += [
        _table(
            ['Strategy', 'Final Team Accuracy (after 200 queries)'],
            result_rows,
            col_widths=[7 * cm, 9 * cm],
        ),
        Spacer(1, 0.3 * cm),
        Paragraph("<b>Recommendation and justification:</b>", S['bold_label']),
        Paragraph(
            "<b>Random sampling is recommended</b> for this task. "
            "Uncertainty-based strategies (Least Confidence, Margin, Entropy) query "
            "instances near the classifier's decision boundary — but these instances "
            "tend to cluster at class boundaries, oversampling certain categories "
            "and undersampling others. This produces <i>biased</i> per-class "
            "competence estimates: classes with few boundary instances are sampled "
            "rarely, so their expert accuracy is estimated poorly for longer. "
            "Random sampling covers all classes more uniformly, allowing each class's "
            "expert accuracy to converge faster on average.",
            S['body'],
        ),
        Paragraph(
            "This finding highlights an important distinction between standard active "
            "learning (where uncertainty-based methods excel at learning the classifier "
            "decision boundary) and competence discovery (where uniform class coverage "
            "is more important than decision-boundary focus).",
            S['body'],
        ),
        Paragraph(
            f"All strategies converge toward the oracle accuracy of {oracle_acc:.2f}%, "
            "confirming that the Laplace-smoothed competence model and Bayes-optimal "
            "deferral rule function correctly.",
            S['note'],
        ),
    ]

    # ── Section 5: Task 5 (Optional) ──────────────────────────────────────────
    story += [
        _divider(),
        Paragraph("Section 5 — Interactive Human Expert Interface (Task 5, Optional)", S['section']),
        Paragraph(
            "As an optional extension, the project includes an interactive interface at "
            "<b>/project3/human-label/</b> where a user can act as the human expert. "
            "The system presents news articles selected by the chosen query strategy "
            "(Entropy by default). The user selects one of the four news categories as "
            "their label. After each label, the system:",
            S['body'],
        ),
    ]

    steps = [
        "Updates the per-class expert competence estimate using Laplace smoothing",
        "Recomputes the Bayes-optimal deferral policy on a held-out evaluation set",
        "Displays the updated estimated team accuracy in real time",
        "Selects the next article using the chosen query strategy",
    ]
    for step in steps:
        story.append(Paragraph(f"• {step}", S['note']))

    story += [
        Spacer(1, 0.2 * cm),
        Paragraph(
            "Strategy can be changed at any time (Random, Least Confidence, Margin, Entropy) "
            "without losing existing labels, allowing exploration of different sampling "
            "approaches in the same session. All session state is stored server-side via "
            "Django sessions.",
            S['body'],
        ),
        Spacer(1, 0.2 * cm),
        Paragraph("You as the Expert: Live Competence Profiling", S['bold_label']),
        Paragraph(
            "After twelve labels, the interface estimates the user's own per-class accuracy "
            "profile using the same Laplace smoothing as the Task 4 active learning loop, so "
            "the human's numbers are directly comparable to what the system learns about the "
            "simulated experts. The interface then identifies the user's specialty category, "
            "reports which simulated expert their competence profile most closely resembles "
            "(smallest L1 distance between smoothed per-class profiles), and — most "
            "importantly — plugs the user into the full deferral pipeline: the user is "
            "simulated on the held-out evaluation set with their estimated profile, the same "
            "alpha = 1 deferral rule from Tasks 3 and 4 is applied, and the resulting team "
            "accuracy is displayed alongside the AI-alone baseline and the AI + simulated "
            "expert team. The evaluator of this project can therefore personally become part "
            "of the human-AI team and observe how their individual strengths change the "
            "deferral policy and the achievable team accuracy. The profile chart and the "
            "team comparison update live after every submitted label.",
            S['body'],
        ),
    ]

    # ── Footer note ───────────────────────────────────────────────────────────
    story += [
        Spacer(1, 0.5 * cm),
        _divider(),
        Paragraph(
            "All results in this report are computed dynamically from the live server "
            "implementation. Model training and simulations run at server start-up. "
            "The interactive visualizations and interface are available at /project3/.",
            S['note'],
        ),
    ]

    doc.build(story)
    return buf.getvalue()
