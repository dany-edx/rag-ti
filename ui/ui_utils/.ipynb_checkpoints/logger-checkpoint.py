import logging
from logging.handlers import RotatingFileHandler

class MyLogger:
    def __init__(self,level=logging.DEBUG, log_file='app.log', max_bytes=1024*1024, backup_count=5):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level)
        # 로테이팅 파일 핸들러 생성
        self.file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
        self.file_handler.setLevel(level)

        # 콘솔 핸들러 생성
        self.console_handler = logging.StreamHandler()
        self.console_handler.setLevel(level)

        # 로그 포매터 생성
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(user_email)s - %(message)s')
        self.file_handler.setFormatter(formatter)
        self.console_handler.setFormatter(formatter)

        # 핸들러를 로거에 추가
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

class MyLoggerChat:
    def __init__(self, level=logging.DEBUG, log_file='app.log', max_bytes=1024*1024, backup_count=5):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level)
        
        # 로테이팅 파일 핸들러 생성
        self.file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
        self.file_handler.setLevel(level)

        # 콘솔 핸들러 생성
        self.console_handler = logging.StreamHandler()
        self.console_handler.setLevel(level)

        # 로그 포매터 생성
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.file_handler.setFormatter(formatter)
        self.console_handler.setFormatter(formatter)

        # 핸들러를 로거에 추가
        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(self.console_handler)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)