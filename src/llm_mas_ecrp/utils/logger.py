import os
import logging


class CustomFormatter(logging.Formatter):
    """自定义日志格式化器，提供更清晰的日志输出"""
    
    def format(self, record):
        # 为不同级别的日志添加视觉分隔
        if record.levelno == logging.INFO:
            # 检查是否是特殊的分隔日志
            if hasattr(record, 'is_separator'):
                return record.getMessage()
            return f"[INFO] {record.getMessage()}"
        elif record.levelno == logging.ERROR:
            return f"\n{'='*80}\n[ERROR] {record.getMessage()}\n{'='*80}\n"
        elif record.levelno == logging.WARNING:
            return f"[WARNING] {record.getMessage()}"
        else:
            return super().format(record)


def config_logger(filepath):
    """配置日志记录器，提供清晰的日志输出格式"""
    log_path = os.path.join(filepath, "workflow.log")
    logging.root.handlers = []
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_formatter = CustomFormatter()
    file_handler.setFormatter(file_formatter)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = CustomFormatter()
    console_handler.setFormatter(console_formatter)
    
    # 配置根记录器
    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(file_handler)
    logging.root.addHandler(console_handler)
    
    return logging.getLogger(__name__)


def log_separator(logger, title="", char="=", length=80):
    """输出分隔线日志"""
    if title:
        title_str = f" {title} "
        padding = (length - len(title_str)) // 2
        separator = char * padding + title_str + char * padding
        if len(separator) < length:
            separator += char * (length - len(separator))
    else:
        separator = char * length
    
    # 创建特殊的日志记录，标记为分隔符
    record = logger.makeRecord(
        logger.name, logging.INFO, "", 0, separator, (), None
    )
    record.is_separator = True
    logger.handle(record)
