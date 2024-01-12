import argparse
import configparser
import numpy as np
import random
import logging
import os
import torch
import smtplib
import typing
import socket

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

class CaseSensitiveConfigParser(configparser.ConfigParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def optionxform(self, optionstr):
        return optionstr

class config:
    """
    Configuration Class
    
    Attributes:
        cfg: configparser.ConfigParser, configuration object
        split_symbol: str, delimiter of the configuration file
        
        Methods:
            init_by_path: initialize the configuration object by the configuration file path
            get: get the value of the option in the configuration file
            __str__: print the configuration file        
            
    """
    def __init__(self, cfg:CaseSensitiveConfigParser, split_symbol=','):
        self.cfg = cfg
        self.split_symbol = split_symbol
    
    @classmethod
    def init_by_path(self, config_path, split_symbol=','):
        cfg = CaseSensitiveConfigParser()
        cfg.read(config_path)
        return self(cfg, split_symbol)
    
    def get(self, session:str=None, option:str=None):
        """
        Get the value of the option in the configuration file. It will automatically convert the string to the corresponding type.
        :param session: str, session name
        :param option: str, option name
        """
        def is_float(str_data:str):
            if '.' in str_data:
                parts = str_data.split('.')
                if len(parts) == 2 and (parts[0].isdigit() or parts[0] == '') and (parts[1].isdigit() or parts[1] == ''):
                    return True
            elif 'e' in str_data.lower():
                parts = str_data.replace('-', "").lower().split('e')
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    return True
            return False
        
        def is_int(str_data):
            if str_data.isdigit():
                return True
            return False
        
        def is_None(str_data):
            if str_data == "None":
                return True
            return False
        
        def is_Bool(str_data):
            if str_data == "True" or str_data == "False":
                return True
            return False
        
        def process(str_data:str):
            if self.split_symbol not in str_data:
                if is_None(str_data):
                    return None
                elif is_float(str_data):
                    return float(str_data)
                elif is_int(str_data):
                    return int(str_data)
                elif is_Bool(str_data):
                    return True if str_data == "True" else False
                else:
                    return str_data
            else:
                return [process(part.strip()) for part in str_data.split(self.split_symbol)]
        
        str_data = self.cfg.get(session, option)
        return process(str_data)
    
    def __str__(self) -> str:
        rst_str = "Config Setting:\n\n"
        for section in self.cfg.sections():
            if section == "global" or section == self.get("global", "method_name") or section == self.get("global", "dataset"):
                rst_str += f'[{section}]\n'
                for option in self.cfg.options(section):
                    if "pwd" not in option and "password" not in option:
                        value = self.cfg.get(section, option)
                        rst_str += f'{option}:\t{value}\n'
                rst_str += "\n"
        return rst_str

def get_args():
    """
    Get command line arguments.

    :return: argparse.Namespace, command line arguments
    """
    parser = argparse.ArgumentParser(description="Implementation of Clustering Methods, detailed setting is in the configuration file")
    parser.add_argument('-cp', '--config_path', help="Configuration file path")
    parser.add_argument('-ss', '--split_symbol', default=',', help="Configuration file delimiter")
    args = parser.parse_args()
    cfg = configparser.ConfigParser()
    cfg.read(args.config_path)
    cfg = config(cfg, split_symbol=args.split_symbol)
    return args, cfg

def seed_init(seed=None):
    """
    Set random seed.
    :param seed: int, random seed, if None, do not set random seed
    """
    if seed is not None:
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def make_dir(cfg):
    """
    Create directories specified in the configuration.
    :param cfg: config object
    """
    log_dir = cfg.get("global", "log_dir")
    weight_dir = cfg.get("global", "weight_dir")
    result_dir = cfg.get("global", "result_dir")
    figure_dir = cfg.get("global", "figure_dir")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

def get_logger(log_dir, description, std_out=True):
    """
    Create and return a logger object for recording and outputting log information.
    :param log_dir: str, log file storage path
    :param description: str, description of the run
    :param std_out: str, whether to output log information on the terminal
    
    :return: logging.Logger, logger object
    """
    # Configure log information format
    formatter = logging.Formatter('[%(asctime)s]: %(message)s')

    # Create logger object
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    # Set file handler
    log_dir = os.path.join(log_dir, f"{description}.log")
    handler = logging.FileHandler(log_dir)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # If std_out is True, set console handler
    if std_out:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        logger.addHandler(console)

    return logger, log_dir

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