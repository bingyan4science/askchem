"""
Subscription notification engine for AskChem.

Checks active subscriptions, finds new matching claims, and sends
HTML email digests via SMTP.

Env vars:
    SMTP_HOST     SMTP server hostname (default: localhost)
    SMTP_PORT     SMTP server port (default: 587)
    SMTP_USER     SMTP username (optional)
    SMTP_PASS     SMTP password (optional)
    SMTP_FROM     From address (default: noreply@askchem.org)
    SMTP_USE_TLS  Set to 0 to disable STARTTLS (default: enabled)
    ASKCHEM_URL   Base URL for links in emails (default: https://askchem.org)
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from . import db

SMTP_HOST = os.environ.get("SMTP_HOST", "localhost")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASS = os.environ.get("SMTP_PASS", "")
SMTP_FROM = os.environ.get("SMTP_FROM", "noreply@askchem.org")
SMTP_USE_TLS = os.environ.get("SMTP_USE_TLS", "1") != "0"
ASKCHEM_URL = os.environ.get("ASKCHEM_URL", "https://askchem.org")


def _format_claim_html(claim: dict) -> str:
    doi = claim.get("source_doi", "")
    quote = claim.get("verbatim_quote", "")
    ctype = claim.get("claim_type", "")
    venue = claim.get("venue", "")
    year = claim.get("year", "")
    authors = claim.get("authors", [])

    author_str = ""
    if isinstance(authors, list) and authors:
        names = [str(a) for a in authors]
        author_str = ", ".join(names[:3])
        if len(names) > 3:
            author_str += " et al."

    citation = author_str
    if venue:
        citation += f" <i>{venue}</i>"
    if year:
        citation += f" ({year})"

    doi_link = ""
    if doi:
        doi_link = '<a href="https://doi.org/' + doi + '">' + doi + '</a>'

    truncated = quote[:300]
    if len(quote) > 300:
        truncated += "..."

    parts = []
    parts.append('<div style="margin-bottom:16px;padding:12px;'
                 'border-left:3px solid #2563eb;background:#f8fafc;">')
    parts.append('<div style="font-size:13px;color:#6b7280;margin-bottom:4px;">')
    parts.append(f'<span style="background:#e0e7ff;color:#3730a3;padding:2px 6px;'
                 f'border-radius:3px;font-size:11px;">{ctype}</span>')
    parts.append('</div>')
    parts.append(f'<div style="font-size:14px;color:#1f2937;margin-bottom:6px;">'
                 f'&ldquo;{truncated}&rdquo;</div>')
    cite_suffix = ""
    if doi_link:
        cite_suffix = " &mdash; " + doi_link
    parts.append(f'<div style="font-size:12px;color:#6b7280;">'
                 f'{citation}{cite_suffix}</div>')
    parts.append('</div>')
    return "\n".join(parts)


def _build_digest_html(sub: dict, claims: list[dict]) -> str:
    sub_type = sub["sub_type"]
    target = sub["target"]
    freq = sub.get("frequency", "weekly")

    if sub_type == "topic":
        heading = "New claims in <b>" + target.replace("/", " / ") + "</b>"
    elif sub_type == "author":
        heading = "New claims by <b>" + target + "</b>"
    elif sub_type == "query":
        heading = 'New results for &ldquo;<b>' + target + '</b>&rdquo;'
    else:
        heading = "New claims on AskChem"

    claims_html = "\n".join(_format_claim_html(c) for c in claims[:20])
    more_text = ""
    if len(claims) > 20:
        more_text = (f'<p style="color:#6b7280;font-size:13px;">'
                     f'...and {len(claims) - 20} more claims.</p>')

    n = len(claims)
    count_word = "claim" if n == 1 else "claims"

    lines = [
        '<!DOCTYPE html><html><head><meta charset="utf-8"></head>',
        '<body style="font-family:-apple-system,BlinkMacSystemFont,Segoe UI,'
        'Roboto,sans-serif;max-width:600px;margin:0 auto;padding:20px;'
        'color:#1f2937;">',
        '<div style="text-align:center;margin-bottom:24px;">',
        '<h1 style="font-size:20px;color:#1e40af;margin:0;">AskChem</h1>',
        f'<p style="font-size:13px;color:#6b7280;margin:4px 0 0 0;">'
        f'Your {freq} digest &mdash; {n} new {count_word}</p>',
        '</div>',
        f'<h2 style="font-size:16px;color:#1f2937;border-bottom:1px solid '
        f'#e5e7eb;padding-bottom:8px;">{heading}</h2>',
        claims_html,
        more_text,
        '<div style="margin-top:24px;padding-top:16px;border-top:1px solid '
        '#e5e7eb;text-align:center;">',
        f'<a href="{ASKCHEM_URL}" style="color:#2563eb;text-decoration:none;'
        f'font-size:13px;">Browse AskChem</a>',
        '<span style="color:#d1d5db;margin:0 8px;">|</span>',
        f'<a href="{ASKCHEM_URL}/#subscriptions" style="color:#6b7280;'
        f'text-decoration:none;font-size:12px;">Manage subscriptions</a>',
        '</div>',
        '</body></html>',
    ]
    return "\n".join(lines)


def _send_email(to: str, subject: str, html_body: str) -> bool:
    """Send an HTML email via SMTP. Returns True on success."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = SMTP_FROM
    msg["To"] = to
    msg.attach(MIMEText(html_body, "html"))

    try:
        server = smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30)
        if SMTP_USE_TLS:
            server.starttls()
        if SMTP_USER and SMTP_PASS:
            server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(SMTP_FROM, [to], msg.as_string())
        server.quit()
        return True
    except Exception as e:
        print(f"  SMTP error sending to {to}: {e}", flush=True)
        return False


def check_subscriptions() -> int:
    """Check all due subscriptions and send notification emails.

    Returns the number of notifications sent.
    """
    due = db.get_due_subscriptions()
    if not due:
        return 0

    sent_count = 0
    for sub in due:
        try:
            claims = db.get_new_claims_for_subscription(sub)
            if not claims:
                db.update_subscription_notified(sub["id"])
                continue

            sub_type = sub["sub_type"]
            target = sub["target"]
            if sub_type == "topic":
                subject = f"AskChem: {len(claims)} new claims in {target}"
            elif sub_type == "author":
                subject = f"AskChem: {len(claims)} new claims by {target}"
            elif sub_type == "query":
                subject = f'AskChem: {len(claims)} new results for "{target}"'
            else:
                subject = f"AskChem: {len(claims)} new claims"

            html = _build_digest_html(sub, claims)

            if SMTP_HOST == "localhost" and not SMTP_USER:
                print(f"  [dry-run] Would send to {sub['email']}: "
                      f"{len(claims)} claims ({sub_type}: {target})", flush=True)
                db.log_notification(sub["id"], len(claims), "dry_run")
                db.update_subscription_notified(sub["id"])
                sent_count += 1
                continue

            success = _send_email(sub["email"], subject, html)
            if success:
                db.log_notification(sub["id"], len(claims), "sent")
                db.update_subscription_notified(sub["id"])
                sent_count += 1
            else:
                db.log_notification(sub["id"], len(claims), "failed",
                                    "SMTP send failed")

        except Exception as e:
            db.log_notification(sub["id"], 0, "error", str(e))

    return sent_count
