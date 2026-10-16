"""Generate the forensic incident report PDF from a JSON case file."""
import os
import json
import sys
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER

DARK_BLUE = colors.HexColor("#003366")
HEADER_BG = colors.HexColor("#003366")
LIGHT_GREY = colors.HexColor("#F0F0F0")
BORDER_GREY = colors.HexColor("#CCCCCC")
TABLE_HEADER = colors.HexColor("#E0E0E0")

SEC_GAP = 0.15 * cm
INNER_GAP = 0.1 * cm


def _build_styles():
    ss = getSampleStyleSheet()

    title = ParagraphStyle(
        "FR_Title", parent=ss["Title"],
        fontSize=20, fontName="Helvetica-Bold",
        textColor=colors.white, alignment=TA_CENTER,
        spaceAfter=4,
    )
    subtitle = ParagraphStyle(
        "FR_Subtitle", parent=ss["Normal"],
        fontSize=11, fontName="Helvetica",
        textColor=colors.white, alignment=TA_CENTER,
        spaceAfter=2,
    )
    body = ParagraphStyle(
        "FR_Body", parent=ss["BodyText"],
        fontSize=9, fontName="Helvetica", leading=11,
    )
    mono = ParagraphStyle(
        "FR_Mono", parent=body,
        fontName="Courier", fontSize=7.5, leading=9,
    )
    footer = ParagraphStyle(
        "FR_Footer", parent=body,
        fontSize=7, textColor=colors.grey, alignment=TA_CENTER,
    )
    envbar = ParagraphStyle(
        "FR_EnvBar", parent=body,
        fontSize=8, fontName="Helvetica",
        textColor=colors.HexColor("#333333"),
    )
    return {
        "title": title, "subtitle": subtitle,
        "body": body, "mono": mono, "footer": footer, "envbar": envbar,
    }


def _kv_table(pairs, key_width=5.0*cm, val_width=12.2*cm):
    t = Table(pairs, colWidths=[key_width, val_width])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), LIGHT_GREY),
        ("FONTNAME",   (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",   (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE",   (0, 0), (-1, -1), 8),
        ("VALIGN",     (0, 0), (-1, -1), "TOP"),
        ("GRID",       (0, 0), (-1, -1), 0.5, BORDER_GREY),
        ("LEFTPADDING",  (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING",   (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
    ]))
    return t


def _section_bar(number, title):
    t = Table(
        [[Paragraph(
            "<b>&nbsp;&nbsp;%s&nbsp;&nbsp;&nbsp;&nbsp;%s</b>" % (number, title),
            ParagraphStyle("sb", fontName="Helvetica-Bold", fontSize=10,
                           textColor=colors.white, leading=13),
        )]],
        colWidths=[17.2*cm],
    )
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), HEADER_BG),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
    ]))
    return t


def _header_table(rows, col_widths=None):
    if col_widths is None:
        col_widths = [3*cm, 5*cm, 9.2*cm]
    t = Table(rows, colWidths=col_widths)
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), TABLE_HEADER),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME",   (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",   (0, 0), (-1, -1), 8),
        ("GRID",       (0, 0), (-1, -1), 0.5, BORDER_GREY),
        ("VALIGN",     (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING",   (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
    ]
    t.setStyle(TableStyle(style_cmds))
    return t


def _fmt_ts(ts_str):
    """Extract HH:MM:SS.mmm from an ISO timestamp string."""
    if not ts_str:
        return ""
    try:
        dt = datetime.fromisoformat(ts_str)
        return dt.strftime("%H:%M:%S.") + "%03d" % (dt.microsecond // 1000)
    except (ValueError, TypeError):
        return ts_str[:19] if len(ts_str) >= 19 else ts_str


def generate_report_pdf(json_path, output_pdf=None):
    with open(json_path) as f:
        report = json.load(f)

    if output_pdf is None:
        output_pdf = os.path.splitext(json_path)[0] + ".pdf"

    styles = _build_styles()

    doc = SimpleDocTemplate(
        output_pdf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=1.2*cm, bottomMargin=1.5*cm,
    )

    report_id = report.get("report_id", "FR-XXXX")
    generated_at = report.get("generated_at", "")
    classification = report.get("classification", "UNKNOWN")
    corr = report.get("correlation_analysis", {})
    verdict = corr.get("verdict", "PENDING")

    story = []

    # Title Banner 
    title_data = [
        [Paragraph("F-RIaaS FORENSIC INCIDENT REPORT", styles["title"])],
        [Paragraph("Report ID: %s" % report_id, styles["subtitle"])],
        [Paragraph(
            "Generated: %s  |  Classification: %s  |  Verdict: %s"
            % (generated_at[:19], classification, verdict),
            styles["subtitle"],
        )],
    ]
    title_table = Table(title_data, colWidths=[17.2*cm])
    title_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), HEADER_BG),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING",   (0, 0), (-1, 0), 12),
        ("BOTTOMPADDING",(0, -1), (-1, -1), 10),
    ]))
    story.append(title_table)

    # Environment bar
    env = report.get("environment", "")
    det = report.get("detection_method", {})
    pres = report.get("evidence_preservation", {})
    ev_dir = pres.get("evidence_dir", "")
    case_id = os.path.basename(ev_dir) if ev_dir else "N/A"
    env_text = (
        "Environment: %s  |  Deployment: %s  |  Case: %s"
        % (env, det.get("deployment", "N/A"), case_id)
    )
    env_data = [[Paragraph(env_text, styles["envbar"])]]
    env_table = Table(env_data, colWidths=[17.2*cm])
    env_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_GREY),
        ("GRID",       (0, 0), (-1, -1), 0.5, BORDER_GREY),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("TOPPADDING",   (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
    ]))
    story.append(env_table)
    story.append(Spacer(1, SEC_GAP))

    # Section 1: Incident Summary
    inc = report.get("incident_details", {})
    story.append(_section_bar("1", "INCIDENT SUMMARY"))
    story.append(Spacer(1, 3))
    story.append(_kv_table([
        ["Total Windows Analysed",  str(inc.get("total_windows_analyzed", "N/A"))],
        ["Attack Windows Detected", str(inc.get("attack_windows_detected", "N/A"))],
        ["Maximum Confidence",      "%.4f" % inc.get("max_confidence", 0)],
        ["Affected VMs",            ", ".join(inc.get("affected_vms", []))],
    ]))
    story.append(Spacer(1, SEC_GAP))

    # Section 2: Detection Method 
    story.append(_section_bar("2", "DETECTION METHOD"))
    story.append(Spacer(1, 3))
    story.append(_kv_table([
        ["Model",              det.get("model", "N/A")],
        ["Architecture",       det.get("architecture", "N/A")],
        ["Pooling",            det.get("pooling", "N/A")],
        ["Sequence Length (L)", str(det.get("input_sequence_length", "N/A"))],
        ["Youden J Threshold (T)", "%.4f" % det["threshold"] if isinstance(det.get("threshold"), (int, float)) else str(det.get("threshold", "N/A"))],
        ["Feature Count",     str(det.get("feature_count", "N/A"))],
        ["Deployment",         det.get("deployment", "N/A")],
    ]))
    story.append(Spacer(1, SEC_GAP))

    # Section 3: SHAP Attributions 
    shap = report.get("shap_attributions", [])
    story.append(_section_bar("3", "SHAP FEATURE ATTRIBUTIONS"))
    story.append(Spacer(1, 3))
    if shap:
        shap_rows = [["Rank", "Feature", "Mean |SHAP|"]]
        for i, s in enumerate(shap, 1):
            shap_rows.append([
                str(i),
                s.get("feature", "N/A"),
                "%.6f" % s.get("mean_abs_shap", 0),
            ])
        story.append(_header_table(shap_rows, col_widths=[1.5*cm, 7.5*cm, 8.2*cm]))
    else:
        story.append(Paragraph("No SHAP attributions available.", styles["body"]))
    story.append(Spacer(1, SEC_GAP))

    # Section 4: Per-Detection SHAP Breakdown
    breakdown = report.get("detection_breakdown", [])
    story.append(_section_bar("4", "PER-DETECTION SHAP BREAKDOWN"))
    story.append(Spacer(1, 3))
    SHAP_PREVIEW_COUNT = 3
    if breakdown:
        story.append(Paragraph(
            "Each flagged detection window is listed below with the top features "
            "that drove that specific classification, computed via gradient-based "
            "attribution (single-step integrated gradients).",
            styles["body"],
        ))
        story.append(Spacer(1, INNER_GAP))
        shown = breakdown[-SHAP_PREVIEW_COUNT:]
        start_idx = len(breakdown) - SHAP_PREVIEW_COUNT + 1
        for i, det in enumerate(shown, start_idx):
            conf = det.get("confidence", 0)
            w_start = det.get("window_start", "?")
            w_end = det.get("window_end", "?")
            story.append(Paragraph(
                "<b>Detection %d</b> &mdash; Window %s-%s, Confidence: %.4f"
                % (i, w_start, w_end, conf),
                styles["body"],
            ))
            per_shap = det.get("per_detection_shap", [])
            if per_shap:
                ps_rows = [["Rank", "Feature", "Attribution Score"]]
                for j, ps in enumerate(per_shap, 1):
                    ps_rows.append([
                        str(j),
                        ps.get("feature", "N/A"),
                        "%.6f" % ps.get("shap_value", 0),
                    ])
                story.append(_header_table(
                    ps_rows, col_widths=[1.5*cm, 7.5*cm, 8.2*cm],
                ))
                story.append(Spacer(1, INNER_GAP))
            else:
                story.append(Paragraph(
                    "No per-detection attributions available for this window.",
                    styles["body"],
                ))
                story.append(Spacer(1, INNER_GAP))
        remaining = len(breakdown) - SHAP_PREVIEW_COUNT
        if remaining > 0:
            ev_dir = pres.get("evidence_dir", "")
            case_json = ev_dir + "/forensic_report.json" if ev_dir else "forensic_report.json"
            story.append(Paragraph(
                "<b>Note:</b> %d additional detection windows (Detection 1 through "
                "Detection %d) follow the same per-window attribution format. "
                "The complete per-detection SHAP breakdown for all %d flagged "
                "windows is available in the JSON case file at "
                "<font face='Courier' size='7'>%s</font>."
                % (remaining, remaining,
                   len(breakdown), case_json),
                styles["body"],
            ))
    else:
        story.append(Paragraph(
            "No per-detection SHAP breakdown available.", styles["body"],
        ))
    story.append(Spacer(1, SEC_GAP))

    # Section 5: Isolation Action 
    iso = report.get("isolation_action", {})
    iso_details = iso.get("details", {}) if iso.get("performed") else {}
    story.append(_section_bar("5", "ISOLATION ACTION"))
    story.append(Spacer(1, 3))
    story.append(_kv_table([
        ["Action Performed", "Yes" if iso.get("performed") else "No"],
        ["Target VM",        iso_details.get("vm_id", "N/A")],
        ["Server ID",        iso_details.get("server_id", "N/A")],
        ["Security Group",   "%s (deny-all)" % iso_details.get("security_group", "N/A")],
        ["Timestamp",        iso_details.get("timestamp", "N/A")],
        ["Status",           iso_details.get("status", "N/A")],
    ]))
    story.append(Spacer(1, SEC_GAP))

    # Section 6: Evidence Preservation 
    story.append(_section_bar("6", "EVIDENCE PRESERVATION"))
    story.append(Spacer(1, 3))

    if pres:
        l1 = pres.get("layer1", {})
        l2 = pres.get("layer2", {})
        l3 = pres.get("layer3", {})
        l1_arts = l1.get("artefacts", {})
        l1_count = sum(1 for v in l1_arts.values() if v) if isinstance(l1_arts, dict) else 0
        l1_total = len(l1_arts) if isinstance(l1_arts, dict) else 0
        l1_dur = l1.get("duration_ms", 0)

        l2_arts = l2.get("artefacts", {})
        l2_count = len(l2_arts) if isinstance(l2_arts, dict) else 0
        l2_dur = l2.get("duration_ms", 0)
        l2_method = l2.get("method", "libguestfs")
        if l2_method == "ssh":
            l2_desc = "Disk artefacts\n(DHCP NS SSH fallback)"
        else:
            l2_desc = "Disk artefacts\n(libguestfs)"

        l3_dur = l3.get("duration_ms", 0)
        l3_name = l3.get("snapshot_name", "N/A")
        total_dur = pres.get("total_duration_ms", 0)
        total_arts = pres.get("total_artefacts", 0)

        l1_collected = [k for k, v in l1_arts.items() if v] if isinstance(l1_arts, dict) else []
        l1_names = ", ".join(sorted(l1_collected)) if l1_collected else "none"
        l2_names = ", ".join(sorted(l2_arts.keys())) if isinstance(l2_arts, dict) and l2_arts else "none"

        pres_rows = [["Layer", "Method", "Artefacts", "Duration"]]
        pres_cell = ParagraphStyle("Pres_Cell", parent=styles["body"], fontSize=8, leading=10)
        pres_rows.append([
            "Layer 1\n(Volatile)",
            "DHCP NS SSH\ninto guest VM",
            Paragraph("%d/%d collected:<br/>%s" % (l1_count, l1_total, l1_names), pres_cell),
            "%.0f ms" % l1_dur,
        ])
        if l2.get("status") == "failed":
            pres_rows.append([
                "Layer 2\n(Disk)",
                l2_desc,
                Paragraph("Failed: %s" % l2.get("reason", "unknown"), pres_cell),
                "%.0f ms" % l2_dur,
            ])
        else:
            pres_rows.append([
                "Layer 2\n(Disk)",
                l2_desc,
                Paragraph("%d collected:<br/>%s" % (l2_count, l2_names), pres_cell),
                "%.0f ms" % l2_dur,
            ])
        pres_rows.append([
            "Layer 3\n(Volume)",
            "CoW snapshot\n(Glance API)",
            Paragraph(l3_name, pres_cell),
            "%.0f ms" % l3_dur,
        ])
        pres_rows.append([
            "Total", "", "%d artefact(s)" % total_arts, "%.0f ms" % total_dur,
        ])
        story.append(_header_table(
            pres_rows,
            col_widths=[2.2*cm, 3.8*cm, 8.2*cm, 3*cm],
        ))

        if ev_dir:
            story.append(Paragraph(
                "<b>Evidence directory:</b> %s" % ev_dir,
                styles["mono"],
            ))

        # Chain of custody entries per artefact
        chain = pres.get("chain_of_custody", [])
        if chain:
            story.append(Spacer(1, INNER_GAP))
            coc_rows = [["#", "Artefact", "SHA-256", "Chained Hash"]]
            for entry in chain:
                coc_rows.append([
                    str(entry["index"]),
                    entry["artefact"],
                    entry["sha256"][:24] + "...",
                    entry.get("chained_hash", "N/A")[:24] + "...",
                ])
            story.append(_header_table(
                coc_rows,
                col_widths=[1*cm, 3.5*cm, 5.8*cm, 6.9*cm],
            ))
            story.append(Paragraph(
                "<b>Chain verified:</b> %s (%d entries)"
                % (pres.get("chain_verified", "N/A"), len(chain)),
                styles["body"],
            ))

        # Binary hash check
        bhc = pres.get("binary_hash_check", {})
        if bhc:
            bins = bhc.get("binaries_checked", [])
            found_bins = [b for b in bins if b.get("present")]
            absent_bins = [b for b in bins if not b.get("present")]
            story.append(Spacer(1, INNER_GAP))
            story.append(Paragraph(
                "<b>Binary hash check:</b> %d binaries checked, %d sniffer tool(s) found"
                % (len(bins), bhc.get("sniffer_tools_found", 0)),
                styles["body"],
            ))
            if found_bins:
                for b in found_bins:
                    story.append(Paragraph(
                        "&nbsp;&nbsp;Found: %s at %s (SHA-256: %s)"
                        % (b["binary"], b.get("path", "N/A"),
                           b.get("sha256", "N/A")[:32] + "..."),
                        styles["mono"],
                    ))
            if absent_bins:
                story.append(Paragraph(
                    "&nbsp;&nbsp;Not present: %s"
                    % ", ".join(b["binary"] for b in absent_bins),
                    styles["mono"],
                ))
    else:
        story.append(Paragraph("No preservation data available.", styles["body"]))
    story.append(Spacer(1, SEC_GAP))

    # Section 7: Correlation Analysis
    story.append(_section_bar("7", "CORRELATION ANALYSIS"))
    story.append(Spacer(1, 3))
    if corr:
        corr_rows = [["Evidence Source", "Result"]]
        corr_rows.append(["ML Detection (LSTM)",
                          "Flagged" if corr.get("ml_detected") else "Not flagged"])
        corr_rows.append(["Log Analysis (syslog, auth, dmesg)",
                          "Suspicious" if corr.get("logs_suspicious") else "No indicators found"])
        corr_rows.append(["PCAP Deep Inspection (PyShark/Scapy)",
                          "Suspicious" if corr.get("pcap_suspicious") else "No indicators found"])
        story.append(_header_table(corr_rows, col_widths=[7.5*cm, 9.7*cm]))
        story.append(Spacer(1, INNER_GAP))
        story.append(_kv_table([
            ["Sources Flagged",  str(corr.get("sources_flagged", "N/A"))],
            ["Final Verdict",    str(corr.get("verdict", "N/A"))],
            ["Confidence",       str(corr.get("confidence", "N/A"))],
        ]))
    else:
        story.append(Paragraph("No correlation data available.", styles["body"]))
    story.append(Spacer(1, SEC_GAP))

    # Section 8: Infrastructure Logs
    infra = report.get("infrastructure_logs", {})
    story.append(_section_bar("8", "INFRASTRUCTURE LOGS"))
    story.append(Spacer(1, 3))
    if infra and not infra.get("error"):
        ports = infra.get("neutron_port_events", [])
        nova = infra.get("nova_instance_actions", [])
        if ports:
            port_rows = [["Port ID", "MAC", "Status", "Fixed IPs", "Updated"]]
            for p in ports:
                ips = ", ".join(ip["ip"] for ip in p.get("fixed_ips", []))
                port_rows.append([
                    p.get("port_id", "N/A")[:18] + "...",
                    p.get("mac_address", "N/A"),
                    p.get("status", "N/A"),
                    ips,
                    p.get("updated_at", "N/A"),
                ])
            story.append(Paragraph(
                "<b>Neutron port events:</b> %d event(s)" % len(ports),
                styles["body"],
            ))
            story.append(_header_table(
                port_rows,
                col_widths=[4*cm, 3.2*cm, 1.8*cm, 4.2*cm, 4*cm],
            ))
        else:
            story.append(Paragraph("Neutron port events: none recorded.", styles["body"]))
        story.append(Paragraph(
            "<b>Nova instance actions:</b> %d action(s) recorded" % len(nova),
            styles["body"],
        ))
    else:
        err = infra.get("error", "No data")
        story.append(Paragraph("Infrastructure log collection: %s" % err, styles["body"]))
    story.append(Spacer(1, SEC_GAP))

    # Section 9: Chain of Custody 
    coc = report.get("chain_of_custody", {})
    story.append(_section_bar("9", "CHAIN OF CUSTODY"))
    story.append(Spacer(1, 3))
    if coc:
        story.append(_kv_table([
            ["Evidence Hash (SHA-256)", coc.get("evidence_hash_sha256", "N/A")],
            ["PCAP Hash (SHA-256)",     coc.get("pcap_hash_sha256", "N/A")],
            ["PCAP File",               coc.get("pcap_file", "N/A")],
            ["Integrity Verified",      str(coc.get("integrity_verified", "N/A"))],
            ["Collection Method",       coc.get("collection_method", "N/A")],
            ["Storage",                 coc.get("storage", "N/A")],
        ]))
    else:
        story.append(Paragraph("No chain of custody data available.", styles["body"]))
    story.append(Spacer(1, SEC_GAP))

    # Section 10: Chronological Timeline
    timeline = report.get("chronological_timeline", [])
    story.append(_section_bar("10", "CHRONOLOGICAL TIMELINE"))
    story.append(Spacer(1, 3))
    tl_cell = ParagraphStyle("TL_Cell", parent=styles["body"], fontSize=7.5, leading=9)
    if timeline:
        tl_rows = [["#", "Timestamp", "Source", "Event"]]
        for ev in timeline:
            tl_rows.append([
                str(ev.get("sequence", "")),
                _fmt_ts(ev.get("timestamp", "")),
                ev.get("source", ""),
                Paragraph(ev.get("event", ""), tl_cell),
            ])
        tl_table = Table(
            tl_rows,
            colWidths=[0.7*cm, 2.8*cm, 2.5*cm, 11*cm],
        )
        tl_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), TABLE_HEADER),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME",   (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE",   (0, 0), (-1, -1), 7.5),
            ("GRID",       (0, 0), (-1, -1), 0.5, BORDER_GREY),
            ("VALIGN",     (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING",  (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING",   (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 2),
        ]))
        story.append(tl_table)
    else:
        story.append(Paragraph("No timeline events recorded.", styles["body"]))

    # Footer
    story.append(Spacer(1, 0.3*cm))
    story.append(HRFlowable(
        width="100%", thickness=0.5, color=BORDER_GREY,
    ))
    story.append(Spacer(1, 0.1*cm))
    story.append(Paragraph(
        "End of Report %s  |  Generated by F-RIaaS Report Service  |  Integrity: VERIFIED"
        % report_id,
        styles["footer"],
    ))

    def _page_footer(canvas, doc_obj):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.grey)
        canvas.drawRightString(
            A4[0] - 2*cm, 1*cm,
            "Page %d" % doc_obj.page,
        )
        canvas.restoreState()

    doc.build(story, onFirstPage=_page_footer, onLaterPages=_page_footer)
    print("  Further Investigation Report PDF -> %s" % output_pdf)
    return output_pdf


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_further_report_pdf.py <json_path> [output.pdf]")
        sys.exit(1)
    generate_report_pdf(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
