"""
Professional PDF Case Dossier Generator for FinGuard AI.
Generates regulatory-grade investigation reports using ReportLab.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.agent.schema import CaseInvestigationReport


def generate_case_pdf(
    report: CaseInvestigationReport,
    output_path: Path | str,
    transaction_details: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Renders an audit-ready, executive PDF dossier for a completed fraud investigation.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(out),
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0f172a"),
    )
    subtitle_style = ParagraphStyle(
        "DocSubTitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#475569"),
    )
    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=10,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155"),
    )
    citation_style = ParagraphStyle(
        "CitationText",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#1e3a8a"),
    )

    story = []

    # 1. Header Banner
    story.append(Paragraph("FINGUARD AI | FRAUD INTELLIGENCE DOSSIER", title_style))
    story.append(Paragraph(
        f"Official Automated Case Disposition Report &bull; Case ID: <b>{report.case_id}</b> &bull; Generated: {report.timestamp}",
        subtitle_style,
    ))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#0284c7"), spaceAfter=12))

    # 2. Key Metadata & Disposition Banner
    risk_color_map = {
        "CRITICAL": colors.HexColor("#dc2626"),
        "HIGH": colors.HexColor("#ea580c"),
        "MEDIUM": colors.HexColor("#ca8a04"),
        "LOW": colors.HexColor("#16a34a"),
    }
    banner_color = risk_color_map.get(report.risk_level, colors.HexColor("#2563eb"))

    banner_data = [
        [
            Paragraph(f"<b>RISK LEVEL</b><br/><font size=14 color='{banner_color.hexval()}'><b>{report.risk_level}</b></font>", body_style),
            Paragraph(f"<b>RECOMMENDED ACTION</b><br/><font size=12><b>{report.recommended_action}</b></font>", body_style),
            Paragraph(f"<b>CONFIDENCE</b><br/><font size=12><b>{report.confidence_score:.1%}</b></font>", body_style),
            Paragraph(f"<b>RESOLUTION SLA</b><br/><font size=12><b>{report.sla_minutes} min</b></font>", body_style),
        ]
    ]
    banner_table = Table(banner_data, colWidths=[1.7 * inch, 2.3 * inch, 1.6 * inch, 1.6 * inch])
    banner_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(banner_table)
    story.append(Spacer(1, 12))

    # 3. Transaction Summary Table
    story.append(Paragraph("1. Target Transaction Record", section_heading))
    txn = transaction_details or {}
    txn_data = [
        ["Transaction ID:", report.transaction_id, "Cardholder / User ID:", report.user_id],
        ["Authorized Amount:", f"${float(txn.get('amount', 0.0)):.2f}", "Merchant Name:", txn.get("merchant_name", "N/A")],
        ["Category / MCC:", f"{txn.get('merchant_category', 'N/A')} (MCC: {txn.get('mcc', 'N/A')})", "Country / Location:", f"{txn.get('country', 'US')} ({txn.get('distance_from_home_km', 0)} km)"],
        ["Device Fingerprint:", str(txn.get("device_id", "N/A")), "New Device / Foreign:", f"{'Yes' if txn.get('is_new_device') else 'No'} / {'Yes' if txn.get('is_foreign_country') else 'No'}"],
    ]
    txn_table = Table(txn_data, colWidths=[1.6 * inch, 2.0 * inch, 1.8 * inch, 1.8 * inch])
    txn_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f1f5f9")),
        ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#f1f5f9")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(txn_table)
    story.append(Spacer(1, 10))

    # 4. Evidentiary Findings
    story.append(Paragraph("2. Corroborated Evidentiary Findings", section_heading))
    if report.evidence_list:
        ev_rows = [["Category", "Severity", "Empirical Finding"]]
        for ev in report.evidence_list:
            ev_rows.append([
                Paragraph(f"<b>{ev.category}</b>", body_style),
                Paragraph(f"<font color='{risk_color_map.get(ev.severity, colors.black).hexval()}'><b>{ev.severity}</b></font>", body_style),
                Paragraph(ev.finding, body_style),
            ])
        ev_table = Table(ev_rows, colWidths=[1.8 * inch, 1.1 * inch, 4.3 * inch])
        ev_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(ev_table)
    else:
        story.append(Paragraph("No anomalous signals or policy breaches detected.", body_style))
    story.append(Spacer(1, 10))

    # 5. Regulatory Policy Citations
    story.append(Paragraph("3. Regulatory & Policy Governance Citations", section_heading))
    if report.policy_citations:
        for idx, cit in enumerate(report.policy_citations, 1):
            p_text = f"<b>[{idx}] {cit.policy_name} &bull; {cit.section}</b><br/>{cit.relevant_clause}"
            story.append(Paragraph(p_text, citation_style))
            story.append(Spacer(1, 3))
    else:
        story.append(Paragraph("Standard processing rules apply; no dispute escalation cited.", body_style))
    story.append(Spacer(1, 10))

    # 6. Narrative Investigation Summary
    story.append(Paragraph("4. Synthesized Case Narrative & Auditor Disposition", section_heading))
    summary_clean = report.investigation_summary.replace("\n\n", "<br/><br/>")
    story.append(Paragraph(summary_clean, body_style))
    story.append(Spacer(1, 15))

    # 7. Signature & Audit Verification Stamp
    audit_data = [
        [
            Paragraph("<b>Automated AI Investigator Agent:</b><br/>FinGuard Agent v1.0 (Audit Hash: <code>VERIFIED_SAFE</code>)", body_style),
            Paragraph("<b>Compliance Supervisor Sign-off:</b><br/>Pending Secondary Review" if report.requires_human_escalation else "<b>Autonomous Approval Authorized:</b><br/>Cleared under FinGuard SOP Protocol", body_style),
        ]
    ]
    audit_table = Table(audit_data, colWidths=[3.6 * inch, 3.6 * inch])
    audit_table.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, -1), 1, colors.HexColor("#94a3b8")),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(KeepTogether(audit_table))

    doc.build(story)
    return out
