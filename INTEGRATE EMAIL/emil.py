import yagmail
from config import EMAIL_USER, EMAIL_PASSWORD

def send_email(recipient: str, subject: str, body: str):
    try:
        yag = yagmail.SMTP(user=EMAIL_USER, password=EMAIL_PASSWORD)
        yag.send(to=recipient, subject=subject, contents=body)
        print(f"✅ Email sent to {recipient}")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
