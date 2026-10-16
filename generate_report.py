"""PDF summary report comparing all trained model results."""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import config


def generate_report(all_metrics: list, all_plots: list):
    os.makedirs(config.REPORT_DIR, exist_ok=True)
    path = os.path.join(config.REPORT_DIR, "experiment_report.pdf")

    doc = SimpleDocTemplate(path, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)

    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    h1 = styles["Heading1"]
    h2 = styles["Heading2"]
    body = styles["BodyText"]
    mono = ParagraphStyle("Mono", parent=body, fontName="Courier", fontSize=8,
                          leading=10)

    story = []

    story.append(Spacer(1, 3 * cm))
    story.append(Paragraph("F-RIaaS ML Experiment Results", title_style))
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph(
        "Comparative evaluation of Autoencoder, Transformer, and LSTM models "
        "for packet-sniffing detection in IaaS environments.",
        body
    ))
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph(
        "<b>Data quality fixes applied:</b> median imputation for missing entropy "
        "(no random placeholders), deterministic FlagCount (0 for non-TCP), "
        "per-dataset ActiveTime/IdleTime derivation (no cross-dataset borrowing).",
        body
    ))
    story.append(PageBreak())

    story.append(Paragraph("Model Comparison Summary", h1))
    story.append(Spacer(1, 0.3 * cm))

    table_data = [["Model", "Accuracy", "Precision", "Recall", "F1-Score", "AUC"]]
    for m in all_metrics:
        table_data.append([
            m["model"],
            f"{m['accuracy']:.4f}",
            f"{m['precision']:.4f}",
            f"{m['recall']:.4f}",
            f"{m['f1_score']:.4f}",
            f"{m['auc']:.4f}",
        ])

    t = Table(table_data, colWidths=[4.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#ecf0f1")]),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.5 * cm))

    best = max(all_metrics, key=lambda m: m["f1_score"])
    story.append(Paragraph(
        f"<b>Best model:</b> {best['model']} "
        f"(F1={best['f1_score']:.4f}, AUC={best['auc']:.4f})",
        body
    ))
    story.append(PageBreak())

    for metrics, plots in zip(all_metrics, all_plots):
        name = metrics["model"]
        story.append(Paragraph(f"{name}", h1))
        story.append(Spacer(1, 0.3 * cm))

        story.append(Paragraph(
            f"Accuracy: {metrics['accuracy']:.4f} &nbsp;|&nbsp; "
            f"Precision: {metrics['precision']:.4f} &nbsp;|&nbsp; "
            f"Recall: {metrics['recall']:.4f} &nbsp;|&nbsp; "
            f"F1: {metrics['f1_score']:.4f} &nbsp;|&nbsp; "
            f"AUC: {metrics['auc']:.4f} &nbsp;|&nbsp; "
            f"Threshold: {metrics['threshold']:.4f}",
            body
        ))
        story.append(Spacer(1, 0.3 * cm))

        story.append(Paragraph("Classification Report:", h2))
        report = metrics.get("report_text", "N/A")
        for line in report.split("\n"):
            story.append(Paragraph(line, mono))
        story.append(Spacer(1, 0.3 * cm))

        img_width = 14 * cm
        for label, key in [
            ("Training Curves", "training_curves"),
            ("ROC Curve", "roc"),
            ("Confusion Matrix", "confusion_matrix"),
        ]:
            if key in plots and os.path.exists(plots[key]):
                story.append(Paragraph(label, h2))
                story.append(Image(plots[key], width=img_width,
                                   height=img_width * 0.4))
                story.append(Spacer(1, 0.3 * cm))

        if "error_distribution" in plots and os.path.exists(plots["error_distribution"]):
            story.append(Paragraph("Reconstruction Error Distribution", h2))
            story.append(Image(plots["error_distribution"], width=img_width,
                               height=img_width * 0.45))

        story.append(PageBreak())

    doc.build(story)
    print(f"\n  Report generated -> {path}")
    return path


if __name__ == "__main__":
    print("Run via run_all.py to generate the full report.")
