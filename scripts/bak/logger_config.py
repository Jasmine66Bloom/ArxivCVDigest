"""
日志配置模块 - 提供统一的日志记录功能

配置控制台和文件日志记录，支持多级别日志和格式控制。
"""

import os
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler

# 日志级别映射表
LOG_LEVELS = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL
}

def setup_logger(name='cv_papers', level='info', log_dir=None, simple_format=False):
    """
    配置并返回日志记录器
    
    Args:
        name: 记录器名称
        level: 日志级别
        log_dir: 日志文件目录，None则仅控制台输出
        simple_format: 启用简化格式（仅消息内容）
    
    Returns:
        配置好的日志记录器
    """
    # 设置日志级别
    log_level = LOG_LEVELS.get(level.lower(), logging.INFO)
    
    # 创建记录器
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # 清除现有处理器以应用新格式
    if logger.handlers:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
    
    # 配置格式化器
    if simple_format:
        formatter = logging.Formatter('%(message)s')
    else:
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 条件性添加文件处理器
    if log_dir and os.environ.get('SAVE_LOGS', '').lower() == 'true':
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(
            log_dir, 
            f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        )
        
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# 默认日志记录器
logger = setup_logger()
