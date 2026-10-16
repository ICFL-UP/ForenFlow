"""Notify via Email: the forensic PDF travels with the alert."""
import os
import smtplib
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def send_email(subject, body, recipient=None, attachment_path=None):
    smtp_host = os.environ.get("FRIAAS_SMTP_HOST")
    smtp_port = int(os.environ.get("FRIAAS_SMTP_PORT", 587))
    smtp_user = os.environ.get("FRIAAS_SMTP_USER")
    smtp_pass = os.environ.get("FRIAAS_SMTP_PASS")
    recipient = recipient or os.environ.get("FRIAAS_ALERT_EMAIL")

    if not all([smtp_host, smtp_user, smtp_pass, recipient]):
        return {"status": "skipped",
                "reason": "SMTP not configured (set FRIAAS_SMTP_* env vars)"}

    msg = MIMEMultipart()
    msg["From"] = smtp_user
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))
    if attachment_path and os.path.exists(attachment_path):
        with open(attachment_path, "rb") as f:
            attachment = MIMEApplication(f.read(), Name=os.path.basename(attachment_path))
        attachment["Content-Disposition"] = (
            f'attachment; filename="{os.path.basename(attachment_path)}"')
        msg.attach(attachment)

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
    except (smtplib.SMTPException, OSError) as exc:
        return {"status": "failed", "reason": str(exc)}
    return {"status": "sent", "recipient": recipient}


def notify_incident(incident_summary, report_path=None):
    classification = incident_summary.get("classification", "INCIDENT")
    body = (
        "F-RIaaS Security Alert\n"
        f"{'=' * 40}\n"
        f"Classification: {classification}\n"
        f"Severity: {incident_summary.get('severity', 'UNKNOWN')}\n"
        f"Suspect VM: {incident_summary.get('suspect_vm', 'N/A')}\n"
        f"Confidence: {incident_summary.get('confidence', 'N/A')}\n"
        f"Time: {incident_summary.get('timestamp', 'N/A')}\n"
        "\nSee attached report for details."
    )
    result = send_email(f"[F-RIaaS ALERT] {classification}", body, attachment_path=report_path)
    print(f"    Email: {result['status']}")
    return result
