# -*- coding: utf-8 -*-
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.mime.base import MIMEBase
import os
import base64
from email import encoders

def sendemail(from_addr, to_addr_list,
              subject, content,
              login, password,filelist=[]):

    msg = MIMEMultipart()
    # the main content
    msgText = MIMEText(content, 'html', 'gbk')
    msg.attach(msgText)

    # sender
    msg['From'] = from_addr
    # receiver
    msg['To'] = to_addr_list[0]

    # attachment
    for onefile in filelist:
        # Compose attachment
        part = MIMEBase('application', "octet-stream")
        part.set_payload(open(onefile, "rb").read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', "attachment; filename= %s" % os.path.basename(onefile))
        msg.attach(part)

    msg['Subject'] = subject
    # Send the email via our own SMTP server.
    smtpserver = smtplib.SMTP("smtp.qq.com", 587)
    smtpserver.starttls()
    # login
    smtpserver.login(login, password)
    problems = smtpserver.send_message(msg)
    smtpserver.quit()
    return problems

sendemail(from_addr='648467389@qq.com',
          to_addr_list=['zeeutao@163.com'],
          subject='Test',
          content='test content hahahaha',
          login='648467389@qq.com',
          password="eozqcfppquskbcij",filelist=['test.png'])




