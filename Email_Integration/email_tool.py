import smtplib
import imaplib
import email
from email.message import EmailMessage
from email.utils import make_msgid
from typing import Tuple
from config import EMAIL_ADDRESS, EMAIL_PASSWORD

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
IMAP_SERVER = "imap.gmail.com"

def send_email(to_address: str, subject: str, body: str, thread_id: str = None) -> str:
    msg = EmailMessage()
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = to_address
    msg["Subject"] = subject

    if thread_id:
        msg["Message-ID"] = make_msgid()
        msg["In-Reply-To"] = thread_id
        msg["References"] = thread_id

    msg.set_content(body)

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as smtp:
        smtp.starttls()
        smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        smtp.send_message(msg)

    return msg["Message-ID"]

def receive_latest_email() -> Tuple[str, str, str]:
    with imaplib.IMAP4_SSL(IMAP_SERVER) as mail:
        mail.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        mail.select("inbox")
        _, data = mail.search(None, "UNSEEN")
        mail_ids = data[0].split()
        if not mail_ids:
            return "", "", ""

        latest_id = mail_ids[-1]
        _, msg_data = mail.fetch(latest_id, "(RFC822)")
        msg = email.message_from_bytes(msg_data[0][1])

        subject = msg["subject"]
        thread_id = msg["Message-ID"]
        from_addr = msg["from"]

        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body = part.get_payload(decode=True).decode()
                    break
        else:
            body = msg.get_payload(decode=True).decode()

        return subject, body.strip(), thread_id
