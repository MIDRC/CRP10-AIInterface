import smtplib
import ssl
import threading

from django.conf import settings
from django.core.mail.backends.base import BaseEmailBackend
from django.core.mail.message import sanitize_address
from django.core.mail.utils import DNS_NAME
from email import encoders
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
import smtplib
#from django.core.mail.backends.smtp import EmailBackend as CoreEmailBackend
import os,sys, logging


class EmailBackend(BaseEmailBackend):
    def __init__(self, host=None, port=None, username=None, password=None,
                 use_tls=None, fail_silently=False, use_ssl=None, timeout=None,
                 ssl_keyfile=None, ssl_certfile=None,
                 **kwargs):
        super(EmailBackend, self).__init__(fail_silently=fail_silently)
        self.host = host or settings.EMAIL_HOST
        self.port = port or settings.EMAIL_PORT
        self.username = settings.EMAIL_HOST_USER if username is None else username
        self.password = settings.EMAIL_HOST_PASSWORD if password is None else password
        self.use_tls = settings.EMAIL_USE_TLS if use_tls is None else use_tls
        self.use_ssl = settings.EMAIL_USE_SSL if use_ssl is None else use_ssl
        self.timeout = settings.EMAIL_TIMEOUT if timeout is None else timeout
        self.ssl_keyfile = settings.EMAIL_SSL_KEYFILE if ssl_keyfile is None else ssl_keyfile
        self.ssl_certfile = settings.EMAIL_SSL_CERTFILE if ssl_certfile is None else ssl_certfile
        if self.use_ssl and self.use_tls:
            raise ValueError(
                "EMAIL_USE_TLS/EMAIL_USE_SSL are mutually exclusive, so only set "
                "one of those settings to True.")
        self.connection = None
        self._lock = threading.RLock()

    def open(self):
        """
        Ensures we have a connection to the email server. Returns whether or
        not a new connection was required (True or False).
        """
        if self.connection:
            # Nothing to do if the connection is already open.
            return False

        connection_class = smtplib.SMTP_SSL if self.use_ssl else smtplib.SMTP
        # If local_hostname is not specified, socket.getfqdn() gets used.
        # For performance, we use the cached FQDN for local_hostname.
        connection_params = {'local_hostname': DNS_NAME.get_fqdn()}
        if self.timeout is not None:
            connection_params['timeout'] = self.timeout
        if self.use_ssl:
            connection_params.update({
                'keyfile': self.ssl_keyfile,
                'certfile': self.ssl_certfile,
            })
        try:
            self.connection = connection_class(self.host, self.port, **connection_params)

            # TLS/SSL are mutually exclusive, so only attempt TLS over
            # non-secure connections.
            if not self.use_ssl and self.use_tls:
                self.connection.ehlo()
                self.connection.starttls(keyfile=self.ssl_keyfile, certfile=self.ssl_certfile)
                self.connection.ehlo()
            if self.username and self.password:
                self.connection.login(self.username, self.password)
            return True
        except smtplib.SMTPException:
            if not self.fail_silently:
                raise
              
    def close(self):
        """Closes the connection to the email server."""
        if self.connection is None:
            return
        try:
            try:
                self.connection.quit()
            except (ssl.SSLError, smtplib.SMTPServerDisconnected):
                # This happens when calling quit() on a TLS connection
                # sometimes, or when the connection was already disconnected
                # by the server.
                self.connection.close()
            except smtplib.SMTPException:
                if self.fail_silently:
                    return
                raise
        finally:
            self.connection = None

            
    def send_messages(self, email_messages):
        """
        Sends one or more EmailMessage objects and returns the number of email
        messages sent.
        """
        if not email_messages:
            return
        with self._lock:
            num_sent = 0
            for email_message in email_messages:
                from_email = sanitize_address(email_message.from_email, email_message.encoding)
                recipients = [sanitize_address(addr, email_message.encoding)
                            for addr in email_message.recipients()]
                message = email_message.message()
                
                sent = self._send(from_email, recipients, email_message.attachments, email_message.subject, email_message.body)
                if sent:
                    num_sent += 1
            # if new_conn_created:
            #     self.close()
        return num_sent

    def _send(self,from_user: str, to_users, files, subject: str, body: str = "") -> None:
        # this method allows multiple file attachements
        try:
            #open("%s/%s" % (file_path, file_name), "rb").read()
            msg = MIMEMultipart()
            msg['From'] = from_user
            msg['To'] = ','.join(to_users)
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'html'))
            if files:
                for f in files:
                    attachment = MIMEBase('application', 'octet-stream')
                    attachment.set_payload(f['data'])
                    encoders.encode_base64(attachment)
                    attachment.add_header(
                        "Content-Disposition", "attachment", filename="%s" % f['file_name'])
                    msg.attach(attachment)

            smtp = smtplib.SMTP(settings.EMAIL_HOST)
            smtp.sendmail(from_user, to_users, msg.as_string())
            smtp.quit()
            
        except Exception as ex:
            logging.error(sys.exc_info()[0])
            print(ex)