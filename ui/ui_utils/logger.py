# import logging
# from logging.handlers import RotatingFileHandler

# class MyLogger:
#     def __init__(self,level=logging.DEBUG, log_file='app.log', max_bytes=1024*1024, backup_count=5):
#         self.logger = logging.getLogger(__name__)
#         self.logger.setLevel(level)
#         # 로테이팅 파일 핸들러 생성
#         self.file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
#         self.file_handler.setLevel(level)

#         # 콘솔 핸들러 생성
#         self.console_handler = logging.StreamHandler()
#         self.console_handler.setLevel(level)

#         # 로그 포매터 생성
#         formatter = logging.Formatter('%(asctime)s\t%(name)s\t%(levelname)s\t%(user_email)s\t%(message)s')
#         self.file_handler.setFormatter(formatter)
#         self.console_handler.setFormatter(formatter)

#         # 핸들러를 로거에 추가
#         self.logger.addHandler(self.file_handler)
#         self.logger.addHandler(self.console_handler)

#     def debug(self, message):
#         self.logger.debug(message, extra={'user_email': self.user_email})

#     def info(self, message):
#         self.logger.info(message, extra={'user_email': self.user_email})

#     def warning(self, message):
#         self.logger.warning(message)

#     def error(self, message):
#         self.logger.error(message)

#     def critical(self, message):
#         self.logger.critical(message)

import logging
from logging.handlers import TimedRotatingFileHandler
import datetime

class MyLogger:
    def __init__(self, level=logging.DEBUG, log_file='app.log', when='midnight', interval=1, backup_count=5):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level)
        self.logger.propagate = False
        # TimedRotatingFileHandler를 사용하여 날짜별로 로그 파일 생성
        # 파일명에 날짜를 추가하여 겹치지 않도록 함
        log_filename = datetime.datetime.now().strftime("%Y%m%d") + '.log'
        self.file_handler = TimedRotatingFileHandler('./prompt_text/' + log_filename, when=when, interval=interval, backupCount=backup_count)
        self.file_handler.setLevel(level)

        # 콘솔 핸들러 생성
        self.console_handler = logging.StreamHandler()
        self.console_handler.setLevel(level)

        # 로그 포매터 생성
        formatter = logging.Formatter('%(asctime)s\t%(name)s\t%(levelname)s\t%(user_email)s\t%(message)s')
        self.file_handler.setFormatter(formatter)
        self.console_handler.setFormatter(formatter)

        # 핸들러를 로거에 추가
        
        self.logger.handlers.clear()
        
        
        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(self.console_handler)

    def debug(self, message):
        self.logger.debug(message, extra={'user_email': self.user_email})

    def info(self, message):
        self.logger.info(message, extra={'user_email': self.user_email})
    
    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)
