# Copyright (c) 2023 Yuxuan Shao

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import os
import socket
import typing
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

from .config import config

class email_reminder:
    """
    Email Reminder Class
    
    Attributes:
        email_cfg_path: str, the path of the email configuration file
        in_use: bool, whether to use the email reminder
        sender: str, the email address of the sender
        receivers: str or list, the email address(es) of the receiver(s)
        mail_host: str, the email host
        mail_user: str, the email user
        mail_pwd: str, the email password
        max_size_mb: int, the maximum size of the attachment in MB
        logger: logging.Logger, the logger object
    """
    def __init__(self, email_cfg_path, logger=None):
        if email_cfg_path is None:
            self.in_use = False
            return
        cfg = config.init_by_path(email_cfg_path)
        self.in_use = cfg.get("email_reminder", "in_use")
        self.sender = cfg.get("email_reminder", "sender")
        self.receivers = cfg.get("email_reminder", "receivers")
        self.mail_host = cfg.get("email_reminder", "mail_host")
        self.mail_user = cfg.get("email_reminder", "mail_user")
        self.mail_pwd = cfg.get("email_reminder", "mail_pwd")
        self.max_size_mb = cfg.get("email_reminder", "max_size_mb") * 1024 * 1024
        self.logger = logger
        if self.logger is not None:
            self.logger.info(f"Email Reminder is {'ON' if self.in_use else 'OFF'}")
    
    def construct_message(self, content:str, title:str, attachs:typing.Union[None, typing.Iterable, str]=None):
        def construct_single_receiver(receiver):
            if attachs is not None:
                message = MIMEMultipart()
                part_text = MIMEText(content, 'plain', 'utf-8')
                message.attach(part_text)
                if type(attachs) is list or type(attachs) is tuple:
                    for attach in attachs:
                        if not os.path.exists(attach):
                            continue
                        assert os.path.isfile(attach), "The Value Of The Attachs Must Be File Path Or List Of File Paths!"
                        if os.path.getsize(attach) > self.max_size_mb and self.logger is not None:
                            self.logger.info(f'[Email Reminder] ERROR During Sending Email: File {attach} Is Too Large!')
                            part_attach = MIMEText(f"\nFile {os.path.split(attach)[1]} Is Too Large! Attachment Uploading Denied!", 'plain', 'utf-8')
                            message.attach(part_attach)
                        else:    
                            part_attach = MIMEApplication(open(attach, 'rb').read())
                            part_attach.add_header('Content-Disposition', 'attachment', filename=os.path.split(attach)[1])
                            message.attach(part_attach)
                elif type(attachs) is str and os.path.isfile(attachs):
                    if os.path.getsize(attachs) > self.max_size_mb:
                        if self.logger is not None:
                            self.logger.info(f'[Email Reminder] ERROR During Sending Email: File {attachs} Is Too Large! Make sure the file size is less than {self.max_size_mb} MB!')
                        part_attach = MIMEText(f"\nFile {os.path.split(attachs)[1]} Is Too Large! Attachment Uploading Denied!", 'plain', 'utf-8')
                        message.attach(part_attach)
                    else:
                        part_attach = MIMEApplication(open(attachs, 'rb').read())
                        part_attach.add_header('Content-Disposition', 'attachment', filename=os.path.split(attachs)[1])
                        message.attach(part_attach)
                else:
                    raise TypeError("The Value Of The Attachs Must Be File Path Or List Of File Paths!")
            else:
                message = MIMEText(content, 'plain', 'utf-8')
            message['Subject'] = title
            message['to'] = receiver
            message['from'] = self.sender
            return message
        
        if not self.in_use:
            return None
        messages = []
        title = f"[Email Reminder From {socket.gethostname()}]:" + title
        if type(self.receivers) is list or type(self.receivers) is tuple:
            for receiver in self.receivers:
                messages.append(construct_single_receiver(receiver)) 
        else:
            message = construct_single_receiver(self.receivers)
            messages.append(message)
        return messages
    
    def send_message(self, content:str, title:str, attachs:typing.Union[None, typing.Iterable[str], str]=None):
        """
        Send email to the receiver(s).
        :param content: str, the content of the email
        :param title: str, the title of the email
        :param attachs: str or list, the attachment(s) of the email
        """
        if not self.in_use:
            return
        messages=self.construct_message(content, title, attachs)
        try:
            # smtp_obj = smtplib.SMTP()
            # smtp_obj.connect(mail_host, 25)
            smtp_obj = smtplib.SMTP_SSL(self.mail_host, port=465, timeout=300)
            smtp_obj.login(self.mail_user, self.mail_pwd)
            for msg in messages:
                smtp_obj.sendmail(msg['from'], msg['to'], msg.as_string())
                if self.logger is not None:
                    self.logger.info(f"[Email Reminder] Email Has Been Sent To {msg['to']} Successfully")
            smtp_obj.quit()

        except smtplib.SMTPException as e:
            if self.logger is not None:
                self.logger.info('[Email Reminder] ERROR During Sending Email: ', str(e))