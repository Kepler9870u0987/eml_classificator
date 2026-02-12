"""
Sample email data for testing.

This module contains various .eml file samples as strings/bytes for testing:
- Plain text emails
- HTML emails
- Multipart emails
- Emails with attachments
- Emails with Italian signatures
- Emails with quoted reply chains
- Emails with PII (emails, Italian phone numbers)
- Malformed emails
"""

# Simple plain text email
SIMPLE_PLAIN_TEXT_EML = b"""From: sender@example.com
To: recipient@example.com
Subject: Test Email
Date: Wed, 12 Feb 2026 10:30:00 +0100
Message-ID: <test123@example.com>
Content-Type: text/plain; charset="utf-8"

Hello, this is a simple test email.

Thank you.
"""

# Email with Italian signature
ITALIAN_SIGNATURE_EML = b"""From: mario.rossi@example.com
To: support@company.com
Subject: Richiesta informazioni
Date: Wed, 12 Feb 2026 11:00:00 +0100
Message-ID: <test456@example.com>
Content-Type: text/plain; charset="utf-8"

Buongiorno,

vorrei avere maggiori informazioni sul vostro servizio.

Cordiali saluti,
Mario Rossi
--
Mario Rossi
Responsabile Acquisti
Tel: 02 12345678
"""

# Email with Distinti saluti signature
DISTINTI_SALUTI_EML = b"""From: anna.bianchi@example.com
To: info@company.com
Subject: Ordine #12345
Date: Wed, 12 Feb 2026 12:00:00 +0100
Message-ID: <test789@example.com>
Content-Type: text/plain; charset="utf-8"

Gentile team,

vi scrivo in merito all'ordine numero 12345.

Distinti saluti,
Anna Bianchi
"""

# Email with quoted reply chain
REPLY_CHAIN_EML = b"""From: customer@example.com
To: support@company.com
Subject: Re: Your inquiry
Date: Wed, 12 Feb 2026 13:00:00 +0100
Message-ID: <test-reply@example.com>
In-Reply-To: <original@example.com>
References: <original@example.com>
Content-Type: text/plain; charset="utf-8"

Thank you for your response!

> On Feb 12, 2026, at 10:00 AM, support@company.com wrote:
>
> We have received your inquiry and will respond soon.
>
> Best regards,
> Support Team
"""

# Email with Italian reply pattern
ITALIAN_REPLY_EML = b"""From: cliente@example.com
To: supporto@azienda.com
Subject: Re: La sua richiesta
Date: Wed, 12 Feb 2026 14:00:00 +0100
Message-ID: <test-reply-it@example.com>
In-Reply-To: <originale@azienda.com>
References: <originale@azienda.com>
Content-Type: text/plain; charset="utf-8"

Grazie mille per la risposta!

Il giorno 12 feb 2026, alle ore 10:00, supporto@azienda.com ha scritto:
> Abbiamo ricevuto la sua richiesta e le risponderemo al piu presto.
>
> Cordiali saluti,
> Team di Supporto
"""

# Email with PII (email addresses and Italian phone numbers)
PII_EMAIL_EML = b"""From: contact@example.com
To: privacy@company.com
Subject: Dati personali
Date: Wed, 12 Feb 2026 15:00:00 +0100
Message-ID: <test-pii@example.com>
Content-Type: text/plain; charset="utf-8"

Il mio indirizzo email personale e mario.rossi@gmail.com.

Puoi contattarmi anche al numero mobile 340 1234567 oppure al
fisso 02 87654321.

Per urgenze scrivi a m.rossi@pec.it o chiama il +39 06 12345678.

Grazie!
"""

# Multipart email with HTML and plain text
MULTIPART_HTML_EML = b"""From: newsletter@example.com
To: subscriber@example.com
Subject: Monthly Newsletter
Date: Wed, 12 Feb 2026 16:00:00 +0100
Message-ID: <newsletter-123@example.com>
MIME-Version: 1.0
Content-Type: multipart/alternative; boundary="boundary123"

--boundary123
Content-Type: text/plain; charset="utf-8"

This is the plain text version of the newsletter.

Visit our website for more information.

--boundary123
Content-Type: text/html; charset="utf-8"

<html>
<head><title>Newsletter</title></head>
<body>
<h1>This is the HTML version</h1>
<p>Visit our <a href="https://example.com">website</a> for more information.</p>
</body>
</html>

--boundary123--
"""

# Email with attachment
ATTACHMENT_EML = b"""From: sender@example.com
To: recipient@example.com
Subject: Document attached
Date: Wed, 12 Feb 2026 17:00:00 +0100
Message-ID: <attach-test@example.com>
MIME-Version: 1.0
Content-Type: multipart/mixed; boundary="boundary-attach"

--boundary-attach
Content-Type: text/plain; charset="utf-8"

Please find the document attached.

--boundary-attach
Content-Type: application/pdf; name="document.pdf"
Content-Disposition: attachment; filename="document.pdf"
Content-Transfer-Encoding: base64

JVBERi0xLjQKJeLjz9MK

--boundary-attach--
"""

# Email with multiple attachments
MULTIPLE_ATTACHMENTS_EML = b"""From: sender@example.com
To: recipient@example.com
Subject: Multiple files
Date: Wed, 12 Feb 2026 18:00:00 +0100
Message-ID: <multi-attach@example.com>
MIME-Version: 1.0
Content-Type: multipart/mixed; boundary="boundary-multi"

--boundary-multi
Content-Type: text/plain; charset="utf-8"

Here are the files you requested.

--boundary-multi
Content-Type: application/pdf; name="report.pdf"
Content-Disposition: attachment; filename="report.pdf"
Content-Transfer-Encoding: base64

JVBERi0xLjMK

--boundary-multi
Content-Type: image/png; name="chart.png"
Content-Disposition: attachment; filename="chart.png"
Content-Transfer-Encoding: base64

iVBORw0KGgoAAAANS

--boundary-multi
Content-Type: text/csv; name="data.csv"
Content-Disposition: attachment; filename="data.csv"
Content-Transfer-Encoding: base64

TmFtZSxBZ2UK

--boundary-multi--
"""

# Email with "Sent from my iPhone" signature
SENT_FROM_IPHONE_EML = b"""From: mobile@example.com
To: friend@example.com
Subject: Quick reply
Date: Wed, 12 Feb 2026 19:00:00 +0100
Message-ID: <mobile-test@example.com>
Content-Type: text/plain; charset="utf-8"

Thanks! I'll check that later.

Sent from my iPhone
"""

# Email with "Inviato da" Italian mobile signature
INVIATO_DA_EML = b"""From: mobile-it@example.com
To: amico@example.com
Subject: Risposta veloce
Date: Wed, 12 Feb 2026 20:00:00 +0100
Message-ID: <mobile-it-test@example.com>
Content-Type: text/plain; charset="utf-8"

Grazie! Controllo dopo.

Inviato da iPhone
"""

# Email with original message forward
FORWARDED_MESSAGE_EML = b"""From: forwarder@example.com
To: recipient@example.com
Subject: Fwd: Important message
Date: Wed, 12 Feb 2026 21:00:00 +0100
Message-ID: <fwd-test@example.com>
Content-Type: text/plain; charset="utf-8"

FYI - see below.

-----Original Message-----
From: original@example.com
Sent: Wednesday, February 12, 2026 10:00 AM
To: forwarder@example.com
Subject: Important message

This is the original message content.
"""

# Italian forwarded message
ITALIAN_FORWARDED_EML = b"""From: inoltratore@example.com
To: destinatario@example.com
Subject: I: Messaggio importante
Date: Wed, 12 Feb 2026 22:00:00 +0100
Message-ID: <fwd-it-test@example.com>
Content-Type: text/plain; charset="utf-8"

Per conoscenza - vedi sotto.

-----Messaggio originale-----
Da: originale@example.com
Inviato: mercoledi 12 febbraio 2026 10:00
A: inoltratore@example.com
Oggetto: Messaggio importante

Questo e il contenuto del messaggio originale.
"""

# Email with empty body
EMPTY_BODY_EML = b"""From: sender@example.com
To: recipient@example.com
Subject: Empty email
Date: Wed, 12 Feb 2026 23:00:00 +0100
Message-ID: <empty-test@example.com>
Content-Type: text/plain; charset="utf-8"

"""

# Email with only HTML (no plain text)
HTML_ONLY_EML = b"""From: htmlsender@example.com
To: htmlrecipient@example.com
Subject: HTML Newsletter
Date: Thu, 13 Feb 2026 09:00:00 +0100
Message-ID: <html-only@example.com>
MIME-Version: 1.0
Content-Type: text/html; charset="utf-8"

<html>
<body>
<h1>Welcome!</h1>
<p>This email <strong>only</strong> has HTML content.</p>
<p>Contact us at <a href="mailto:support@example.com">support@example.com</a></p>
</body>
</html>
"""

# Email with ISO-8859-1 (Latin-1) encoding
LATIN1_ENCODING_EML = b"""From: latin@example.com
To: recipient@example.com
Subject: Latin encoding test
Date: Thu, 13 Feb 2026 10:00:00 +0100
Message-ID: <latin1-test@example.com>
Content-Type: text/plain; charset="iso-8859-1"

Caf\xe9 na\xefve r\xe9sum\xe9.
"""

# Malformed email (invalid RFC5322)
MALFORMED_EML = b"""This is not a valid email format
Missing headers
No structure
"""

# Email with CC and BCC headers
CC_BCC_EMAIL_EML = b"""From: sender@example.com
To: to1@example.com, to2@example.com
Cc: cc1@example.com, cc2@example.com
Subject: Multiple recipients
Date: Thu, 13 Feb 2026 11:00:00 +0100
Message-ID: <cc-test@example.com>
Content-Type: text/plain; charset="utf-8"

This email has multiple recipients in To and CC fields.
"""

# Email with References header (threading)
THREADED_EMAIL_EML = b"""From: participant@example.com
To: thread@example.com
Subject: Re: Re: Original topic
Date: Thu, 13 Feb 2026 12:00:00 +0100
Message-ID: <thread-3@example.com>
In-Reply-To: <thread-2@example.com>
References: <thread-1@example.com> <thread-2@example.com>
Content-Type: text/plain; charset="utf-8"

This is part of a longer email thread.
"""

# Email with multiple phone number formats
MULTIPLE_PHONE_FORMATS_EML = b"""From: phones@example.com
To: recipient@example.com
Subject: Phone numbers test
Date: Thu, 13 Feb 2026 13:00:00 +0100
Message-ID: <phones-test@example.com>
Content-Type: text/plain; charset="utf-8"

Mobile: 340 1234567
Mobile with country code: +39 340 1234567
Landline Milan: 02 12345678
Landline Rome: 06 87654321
Mobile alternative format: 3401234567
Landline with dash: 02-12345678
"""

# Email with URLs
URLS_EMAIL_EML = b"""From: urls@example.com
To: recipient@example.com
Subject: Links test
Date: Thu, 13 Feb 2026 14:00:00 +0100
Message-ID: <urls-test@example.com>
Content-Type: text/plain; charset="utf-8"

Check out these links:
https://www.example.com
http://subdomain.example.org/path/to/page
https://secure.site.com/login?param=value

Visit our website!
"""

# Email with excessive whitespace
EXCESSIVE_WHITESPACE_EML = b"""From: whitespace@example.com
To: recipient@example.com
Subject: Whitespace test
Date: Thu, 13 Feb 2026 15:00:00 +0100
Message-ID: <whitespace-test@example.com>
Content-Type: text/plain; charset="utf-8"

This    email    has    excessive    whitespace.


And multiple



blank lines.

    Also indentation.
"""

# Email with script and style tags in HTML
HTML_WITH_SCRIPTS_EML = b"""From: htmlscript@example.com
To: recipient@example.com
Subject: HTML with scripts
Date: Thu, 13 Feb 2026 16:00:00 +0100
Message-ID: <html-script@example.com>
MIME-Version: 1.0
Content-Type: text/html; charset="utf-8"

<html>
<head>
<style>
body { font-family: Arial; }
</style>
<script>
function trackEmail() {
  console.log("tracking");
}
</script>
</head>
<body onload="trackEmail()">
<p>This HTML has scripts and styles that should be removed.</p>
</body>
</html>
"""

# Email with tracking pixel
HTML_WITH_TRACKING_EML = b"""From: tracking@example.com
To: recipient@example.com
Subject: Marketing email
Date: Thu, 13 Feb 2026 17:00:00 +0100
Message-ID: <tracking-test@example.com>
MIME-Version: 1.0
Content-Type: text/html; charset="utf-8"

<html>
<body>
<p>Check out our offer!</p>
<img src="https://tracking.example.com/pixel.gif?id=12345" width="1" height="1" />
</body>
</html>
"""

# Email with HTML entities
HTML_ENTITIES_EML = b"""From: entities@example.com
To: recipient@example.com
Subject: HTML entities test
Date: Thu, 13 Feb 2026 18:00:00 +0100
Message-ID: <entities-test@example.com>
MIME-Version: 1.0
Content-Type: text/html; charset="utf-8"

<html>
<body>
<p>Entities: &amp; &lt; &gt; &quot; &apos; &copy; &nbsp;</p>
</body>
</html>
"""

# Dictionary mapping names to sample emails
SAMPLE_EMAILS = {
    "simple_plain_text": SIMPLE_PLAIN_TEXT_EML,
    "italian_signature": ITALIAN_SIGNATURE_EML,
    "distinti_saluti": DISTINTI_SALUTI_EML,
    "reply_chain": REPLY_CHAIN_EML,
    "italian_reply": ITALIAN_REPLY_EML,
    "pii_email": PII_EMAIL_EML,
    "multipart_html": MULTIPART_HTML_EML,
    "attachment": ATTACHMENT_EML,
    "multiple_attachments": MULTIPLE_ATTACHMENTS_EML,
    "sent_from_iphone": SENT_FROM_IPHONE_EML,
    "inviato_da": INVIATO_DA_EML,
    "forwarded_message": FORWARDED_MESSAGE_EML,
    "italian_forwarded": ITALIAN_FORWARDED_EML,
    "empty_body": EMPTY_BODY_EML,
    "html_only": HTML_ONLY_EML,
    "latin1_encoding": LATIN1_ENCODING_EML,
    "malformed": MALFORMED_EML,
    "cc_bcc": CC_BCC_EMAIL_EML,
    "threaded": THREADED_EMAIL_EML,
    "multiple_phone_formats": MULTIPLE_PHONE_FORMATS_EML,
    "urls": URLS_EMAIL_EML,
    "excessive_whitespace": EXCESSIVE_WHITESPACE_EML,
    "html_with_scripts": HTML_WITH_SCRIPTS_EML,
    "html_with_tracking": HTML_WITH_TRACKING_EML,
    "html_entities": HTML_ENTITIES_EML,
}
