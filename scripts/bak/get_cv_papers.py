#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CV论文获取与分类程序

从 arXiv 获取最新CV论文并使用ChatGLM模型进行分类。

作者: Jasmine Bloom
日期: 2025-06
"""

import os
import re
import math
import traceback
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from collections import defaultdict
from categories_config import CATEGORY_DISPLAY_ORDER, CATEGORY_THRESHOLDS, CATEGORY_KEYWORDS
from chatglm_helper import ChatGLMHelper
from typing import Dict, List, Tuple, Optional
import traceback
import arxiv
import random

from chatglm_helper import ChatGLMHelper
chatglm = ChatGLMHelper()

# 使用logger_config模块配置日志记录器
from logger_config import setup_logger
# 使用简化格式，不显示时间、级别和日志名
logger = setup_logger(name='cv_papers', level='info', simple_format=True)

# 查询参数
QUERY_DAYS_AGO = 3          # 目标日期偏移量
MAX_RESULTS = 300           # 最大论文数
MAX_WORKERS = 8            # 并行线程数

def get_related_category_pairs():
    """
    根据最新的类别定义生成相关类别对
    这些对在计算类别相似度时使用
    
    Returns:
        List[Tuple[str, str]]: 相关类别对列表
    """
    # 使用基础研究类别（CATEGORY_DISPLAY_ORDER的前10个）
    basic_research_categories = CATEGORY_DISPLAY_ORDER[:10]
    
    # 创建相关类别对
    # 将每个类别名称的中文部分提取出来（去除英文部分）
    related_pairs = [
        # 视觉表征学习与大模型 和 静态图像理解与语义解析
        (basic_research_categories[0].split(' (')[0], basic_research_categories[1].split(' (')[0]),
        # 静态图像理解与语义解析 和 三维重建与几何感知
        (basic_research_categories[1].split(' (')[0], basic_research_categories[3].split(' (')[0]),
        # 生成式视觉与内容创建 和 多模态视觉与跨模态学习
        (basic_research_categories[2].split(' (')[0], basic_research_categories[5].split(' (')[0]),
        # 动态视觉与时序建模 和 感知-动作智能与主动视觉
        (basic_research_categories[4].split(' (')[0], basic_research_categories[8].split(' (')[0]),
        # 模型优化与系统鲁棒性 和 效率学习与适应性智能
        (basic_research_categories[6].split(' (')[0], basic_research_categories[7].split(' (')[0]),
        # 效率学习与适应性智能 和 前沿视觉理论与跨学科融合
        (basic_research_categories[7].split(' (')[0], basic_research_categories[9].split(' (')[0])
    ]
    
    return related_pairs

# 导入NLTK库用于文本预处理
try:
    import nltk
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    
    # 创建标志文件路径，用于记录NLTK数据是否已下载
    nltk_flag_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.nltk_data_downloaded')
    
    # 检查是否已经下载过NLTK数据
    if os.path.exists(nltk_flag_file):
        # 已经下载过，直接使用
        logger.debug("NLTK数据已下载过，使用现有数据")
        NLTK_AVAILABLE = True
    else:
        logger.info("首次运行，检查NLTK数据是否已下载")
        # 检查必要的NLTK数据是否已下载
        needed_data = []
        for data_name in ['punkt', 'wordnet', 'stopwords']:
            try:
                path = f"{'tokenizers/' if data_name == 'punkt' else 'corpora/'}{data_name}"
                nltk.data.find(path)
                logger.debug(f"NLTK数据 '{data_name}' 已存在于: {path}")
            except LookupError:
                needed_data.append(data_name)
                logger.info(f"NLTK数据 '{data_name}' 不存在，将下载")
        
        # 只下载缺失的数据，避免重复下载
        if needed_data:
            logger.info(f"正在下载缺失的NLTK数据文件: {', '.join(needed_data)}")
            for data_name in needed_data:
                logger.info(f"开始下载 '{data_name}'...")
                download_result = nltk.download(data_name, quiet=False)
                logger.info(f"下载 '{data_name}' 结果: {download_result}")
            logger.info("NLTK数据文件下载完成")
        
        # 特别处理punkt_tab（某些NLTK功能需要此数据）
        try:
            nltk.data.find('tokenizers/punkt_tab')
            logger.debug("NLTK数据 'punkt_tab' 已存在")
        except LookupError:
            logger.info("开始下载 'punkt_tab'...")
            # 重新下载punkt可能会包含punkt_tab
            download_result = nltk.download('punkt', quiet=False)
            logger.info(f"下载 'punkt' 结果: {download_result}")
        
        # 创建标志文件表示数据已下载，避免下次重复下载
        with open(nltk_flag_file, 'w') as f:
            f.write(f"NLTK data downloaded at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        logger.info("NLTK初始化完成")
        NLTK_AVAILABLE = True
    
    NLTK_AVAILABLE = True
except ImportError:
    logger.warning("NLTK库未安装，将使用基本文本处理方法")
    NLTK_AVAILABLE = False

def extract_github_link(text, paper_url=None, title=None, authors=None, pdf_url=None):
    """从文本中提取GitHub链接

    Args:
        text: 论文摘要文本
        paper_url: 论文URL
        title: 论文标题
        authors: 作者列表
        pdf_url: PDF文件URL

    Returns:
        str: GitHub链接或None
    """
    # GitHub链接模式
    github_patterns = [
        # GitHub链接
        r'https?://github\.com/[a-zA-Z0-9-]+/[a-zA-Z0-9-_.]+',
        r'github\.com/[a-zA-Z0-9-]+/[a-zA-Z0-9-_.]+',
        r'https?://www\.github\.com/[a-zA-Z0-9-]+/[a-zA-Z0-9-_.]+',
        r'www\.github\.com/[a-zA-Z0-9-]+/[a-zA-Z0-9-_.]+',
        # 项目页面
        r'https?://[a-zA-Z0-9-]+\.github\.io/[a-zA-Z0-9-_.]+',
        # 通用代码链接模式
        r'code.*available.*?(?:https?://github\.com/[^\s<>"]+)',
        r'implementation.*?(?:https?://github\.com/[^\s<>"]+)',
        r'source.*code.*?(?:https?://github\.com/[^\s<>"]+)',
    ]

    # 从摘要中查找
    for pattern in github_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            url = match.group(0)
            if not url.startswith('http'):
                url = 'https://' + url
            return url

    return None


def extract_arxiv_id(url):
    """从ArXiv URL中提取论文ID

    Args:
        url: ArXiv论文URL

    Returns:
        str: 论文ID
    """
    # 处理不同格式的ArXiv URL
    patterns = [
        r"arxiv\.org/abs/(\d+\.\d+)",
        r"arxiv\.org/pdf/(\d+\.\d+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


def df_to_markdown_table(papers_by_category: dict, target_date) -> str:
    """生成表格形式的Markdown内容，支持两级类别标题"""
    markdown = ""
    
    # 过滤掉没有论文的类别
    active_categories = {k: v for k, v in papers_by_category.items() if v}
    
    if not active_categories:
        return "今天没有相关论文。"
    
    # 表格列标题
    headers = ['状态', '英文标题', '中文标题', '作者', 'PDF链接', '代码/贡献']
    
    # 按照CATEGORY_DISPLAY_ORDER的顺序处理类别
    for category in CATEGORY_DISPLAY_ORDER:
        if category not in active_categories:
            continue
        # 只输出一次主类别标题
        markdown += f"\n## {category}\n\n"
        papers_by_subcategory = defaultdict(list)
        for paper in active_categories[category]:
            subcategory = paper.get('subcategory', '')
            if subcategory and subcategory != "未指定":
                papers_by_subcategory[subcategory].append(paper)
            elif category == "其他 (Others)":
                papers_by_subcategory["未分类"].append(paper)
        if not papers_by_subcategory and category != "其他 (Others)":
            continue
        for subcategory, papers in papers_by_subcategory.items():
            markdown += f"\n### {subcategory}\n\n"
            markdown += "|" + "|".join(headers) + "|\n"
            markdown += "|" + "|".join(["---"] * len(headers)) + "|\n"
            for paper in papers:
                if paper['is_updated']:
                    status = f"📝 更新"
                else:
                    status = f"🆕 发布"
                def summarize_contribution(core_contribution):
                    if not core_contribution:
                        return []
                    if "|" in core_contribution:
                        items = [item.strip() for item in core_contribution.split("|")]
                    else:
                        items = [core_contribution.strip()]
                    blacklist = ["代码开源", "提供数据集", "代码已开源", "数据集已公开"]
                    items = [i for i in items if all(b not in i for b in blacklist)]
                    items = items[:2]
                    items = [(i[:50] + ("..." if len(i) > 50 else "")) for i in items]
                    return items
                contrib_list = []
                if "核心贡献" in paper:
                    contrib_list = summarize_contribution(paper["核心贡献"])
                if paper['github_url'] != 'None':
                    code_and_contribution = f"[代码]({paper['github_url']})"
                    if contrib_list:
                        code_and_contribution += "; " + "; ".join(contrib_list)
                elif contrib_list:
                    code_and_contribution = "; ".join(contrib_list)
                else:
                    code_and_contribution = '无'
                values = [
                    status,
                    paper['title'],
                    paper.get('title_zh', ''),
                    paper['authors'],
                    f"<{paper['pdf_url']}>",
                    code_and_contribution,
                ]
                values = [str(v).replace('\n', ' ').replace('|', '&#124;') for v in values]
                markdown += "|" + "|".join(values) + "|\n"
            markdown += "\n"
    return markdown


def df_to_markdown_detailed(papers_by_category: dict, target_date) -> str:
    """生成详细格式的Markdown内容，支持两级类别标题"""
    markdown = ""
    
    # 过滤掉没有论文的类别
    active_categories = {k: v for k, v in papers_by_category.items() if v}
    
    if not active_categories:
        return "今天没有相关论文。"
    
    # 按照CATEGORY_DISPLAY_ORDER的顺序处理类别
    for category in CATEGORY_DISPLAY_ORDER:
        if category not in active_categories:
            continue
            
        # 添加一级类别标题
        markdown += f"\n## {category}\n\n"
        
        # 按子类别组织论文
        papers_by_subcategory = defaultdict(list)
        
        # 将所有论文分配到子类别
        for paper in active_categories[category]:
            subcategory = paper.get('subcategory', '')
            if subcategory and subcategory != "未指定":
                papers_by_subcategory[subcategory].append(paper)
            elif category == "其他 (Others)":
                # 对于"其他"类别，没有子类别的论文直接显示在主类别下
                papers_by_subcategory["未分类"].append(paper)
        
        # 如果当前类别下没有带子类别的论文，跳过
        if not papers_by_subcategory and category != "其他 (Others)":
            continue
            
        # 处理每个子类别
        for subcategory, papers in papers_by_subcategory.items():
            # 添加二级类别标题
            markdown += f"\n### {subcategory}\n\n"
            
            # 添加论文详细信息
            for idx, paper in enumerate(papers, 1):
                # 引用编号
                markdown += f'**index:** {idx}<br />\n'
                # 日期
                markdown += f'**Date:** {target_date.strftime("%Y-%m-%d")}<br />\n'
                # 英文标题
                markdown += f'**Title:** {paper["title"]}<br />\n'
                # 中文标题
                markdown += f'**Title_cn:** {paper.get("title_zh", "")}<br />\n'
                # 作者（已经是格式化好的字符串）
                markdown += f'**Authors:** {paper["authors"]}<br />\n'
                # PDF链接
                markdown += f'**PDF:** <{paper["pdf_url"]}><br />\n'

                # 合并代码链接和精简后的核心贡献
                markdown += '**Code/Contribution:**\n'
                
                # 精简核心贡献内容
                def summarize_contribution(core_contribution):
                    if not core_contribution:
                        return []
                    # 分割为多条
                    if "|" in core_contribution:
                        items = [item.strip() for item in core_contribution.split("|")] 
                    else:
                        items = [core_contribution.strip()]
                    # 去除模板化内容
                    blacklist = ["代码开源", "提供数据集", "代码已开源", "数据集已公开"]
                    items = [i for i in items if all(b not in i for b in blacklist)]
                    # 只保留前三条
                    items = items[:3]
                    return items
                
                # 处理核心贡献
                contrib_list = []
                if "核心问题" in paper:
                    markdown += f'问题：{paper["核心问题"]}\n'
                
                if "核心方法" in paper:
                    markdown += f'方法：{paper["核心方法"]}\n'
                
                if "核心贡献" in paper:
                    contrib_list = summarize_contribution(paper["核心贡献"])
                    if contrib_list:
                        markdown += f'{", ".join(contrib_list)}\n'
                
                # 处理代码链接
                if paper['github_url'] != 'None':
                    markdown += f'[代码]({paper["github_url"]})\n'
                
                # 添加空行
                markdown += '\n'

    return markdown


def preprocess_text(text: str) -> str:
    """
    对文本进行预处理，包括小写转换、分词、去停用词、词干提取和词形还原
    
    Args:
        text: 原始文本
        
    Returns:
        str: 预处理后的文本
    """
    # 转换为小写
    text = text.lower()
    
    # 基本文本处理：先去除特殊字符
    basic_processed = re.sub(r'[^\w\s]', ' ', text)
    
    # 如果NLTK不可用，直接返回基本处理结果
    if not NLTK_AVAILABLE:
        return basic_processed
    
    # 尝试使用NLTK进行高级处理
    try:
        # 分词 - 先使用基本分词作为备选
        try:
            tokens = word_tokenize(text)
        except Exception:
            # 如果高级分词失败，使用基本分词
            tokens = basic_processed.split()
        
        # 去除停用词
        try:
            stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
        except Exception:
            # 如果停用词处理失败，使用基本停用词列表
            basic_stop_words = {'a', 'an', 'the', 'in', 'on', 'at', 'for', 'to', 'of', 'and', 'or', 'with', 'by'}
            tokens = [token for token in tokens if token not in basic_stop_words and len(token) > 2]
        
        # 词干提取和词形还原 - 可选功能
        try:
            stemmer = PorterStemmer()
            stemmed_tokens = [stemmer.stem(token) for token in tokens]
            
            lemmatizer = WordNetLemmatizer()
            lemmatized_tokens = [lemmatizer.lemmatize(token) for token in stemmed_tokens]
            
            # 重新组合成文本
            return " ".join(lemmatized_tokens)
        except Exception:
            # 如果词干提取或词形还原失败，只返回分词和去停用词的结果
            return " ".join(tokens)
    
    except Exception as e:
        logger.error(f"NLTK处理文本时出错: {str(e)}")
        # 如果所有NLTK处理都失败，回退到基本处理
        return basic_processed


def get_category_by_keywords(title: str, abstract: str, categories_config: Dict) -> List[Tuple[str, float, Optional[Tuple[str, float]], Optional[Dict]]]:
    """
    执行基于关键词匹配和优先级规则的层次化论文分类，带有增强的文本处理和置信度评分。
    
    Args:
        title (str): 论文标题，用于主要上下文分析
        abstract (str): 论文摘要，用于全面内容分析
        categories_config (Dict): 包含类别定义、关键词、权重和优先级的配置字典
    
    实现细节:
        1. 增强文本预处理：
           - 大小写标准化和标准化处理
           - 标题和摘要的组合分析，使用差异化权重
           - 高级分词和停用词过滤
           - 多级词干提取和词形还原
           - N-gram分析，提高短语匹配准确性
        
        2. 优化评分机制：
           - 主要得分：加权关键词匹配（动态基础权重）
           - 标题加成：标题匹配的额外权重（优化加权）
           - 精确匹配加成：完整短语匹配的额外权重
           - 优先级乘数：类别特定重要性缩放
           - 负面关键词惩罚：使用改进的逻辑函数平滑惩罚
           - 类别相关性判断：考虑类别间的相关性
        
        3. 智能分类逻辑：
           - 使用类别自定义阈值与动态阈值调整
           - 增强的子类别分类
           - 优先类别的层次化处理
           - 智能回退机制，考虑类别相关性
           - 置信度评分和分类解释
    
    Returns:
        List[Tuple[str, float, Optional[Tuple[str, float]], Optional[Dict]]]: 按置信度降序排序的 
        (类别, 置信度分数, 子类别信息, 分类解释) 元组列表
    """
    # 文本预处理
    title_lower = title.lower()
    abstract_lower = abstract.lower()
    
    # 使用高级文本预处理
    processed_title = preprocess_text(title)
    processed_abstract = preprocess_text(abstract)
    processed_combined = processed_title + " " + processed_abstract
    
    # 移除常见的停用词，提高匹配质量
    stop_words = {'a', 'an', 'the', 'in', 'on', 'at', 'for', 'to', 'of', 'and', 'or', 'with', 'by', 
                 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 
                 'does', 'did', 'but', 'if', 'then', 'else', 'when', 'up', 'down', 'this', 'that'}
    
    # 分词并过滤停用词
    title_words = set(w for w in title_lower.split() if w not in stop_words)
    abstract_words = set(w for w in abstract_lower.split() if w not in stop_words)
    
    # 组合文本用于匹配
    combined_text = title_lower + " " + abstract_lower
    
    # 初始化得分累加器和匹配记录
    scores = defaultdict(float)
    match_details = defaultdict(list)
    
    # 计算每个类别的得分
    for category, config in categories_config.items():
        score = 0.0
        matches = []
        
        # 1. 正向关键词匹配
        for keyword, weight in config["keywords"]:
            keyword_lower = keyword.lower()
            keyword_words = set(w for w in keyword_lower.split() if w not in stop_words)
            
            # 对关键词也进行预处理
            processed_keyword = preprocess_text(keyword)
            
            # 完整短语精确匹配（最高权重）
            if keyword_lower in title_lower:
                match_score = weight * 0.25  # 标题中的精确匹配权重最高
                score += match_score
                matches.append(f"标题精确匹配 [{keyword}]: +{match_score:.2f}")
            elif keyword_lower in abstract_lower:
                match_score = weight * 0.15  # 摘要中的精确匹配权重次之
                score += match_score
                matches.append(f"摘要精确匹配 [{keyword}]: +{match_score:.2f}")
            
            # 使用预处理后的文本进行匹配（提高准确性）
            elif processed_keyword in processed_title:
                match_score = weight * 0.22  # 预处理标题中的匹配权重高
                score += match_score
                matches.append(f"标题语义匹配 [{keyword}]: +{match_score:.2f}")
            elif processed_keyword in processed_abstract:
                match_score = weight * 0.14  # 预处理摘要中的匹配权重中等
                score += match_score
                matches.append(f"摘要语义匹配 [{keyword}]: +{match_score:.2f}")
            
            # 标题中的关键词组合匹配（高权重）
            elif len(keyword_words) > 1 and keyword_words.issubset(title_words):
                match_score = weight * 0.18  # 标题中的词组匹配权重高
                score += match_score
                matches.append(f"标题词组匹配 [{keyword}]: +{match_score:.2f}")
            
            # 摘要中的关键词组合匹配（中等权重）
            elif len(keyword_words) > 1 and keyword_words.issubset(abstract_words):
                match_score = weight * 0.12  # 摘要中的词组匹配权重中等
                score += match_score
                matches.append(f"摘要词组匹配 [{keyword}]: +{match_score:.2f}")
            
            # 单词匹配（低权重）
            else:
                # 将关键词拆分为单词进行匹配
                word_matches = 0
                title_match_bonus = 0
                
                # 分别处理原始文本和预处理文本的匹配
                for word in keyword_words:
                    if len(word) <= 3:  # 忽略过短的词
                        continue
                        
                    if word in title_words:
                        word_matches += 1
                        title_match_bonus += 1  # 标题匹配额外加分
                    elif word in abstract_words:
                        word_matches += 0.6  # 摘要匹配的权重低于标题
                
                # 处理预处理文本中的匹配
                processed_keyword_words = processed_keyword.split()
                for word in processed_keyword_words:
                    if len(word) <= 3:  # 忽略过短的词
                        continue
                        
                    if word in processed_title:
                        word_matches += 0.8  # 预处理文本中的匹配权重稍低
                        title_match_bonus += 0.8
                    elif word in processed_abstract:
                        word_matches += 0.5
                
                # 只有当匹配到足够多的单词时才计算得分
                if word_matches > 0 and len(keyword_words) > 0:
                    # 计算匹配比例
                    match_ratio = word_matches / (len(keyword_words) + len(processed_keyword_words) / 2)
                    if match_ratio >= 0.4:  # 降低阈值以增加灵活性
                        match_score = weight * 0.08 * match_ratio  # 基础分
                        title_bonus = weight * 0.04 * (title_match_bonus / (len(keyword_words) + len(processed_keyword_words) / 2))  # 标题加分
                        
                        total_score = match_score + title_bonus
                        score += total_score
                        matches.append(f"单词匹配 [{keyword}]: +{total_score:.2f} (匹配率: {match_ratio:.1f})")
            
            # 单词频率加成（对于重复出现的关键词给予额外加成）
            if len(keyword_words) == 1 and keyword_lower in combined_text:
                # 计算关键词在文本中出现的次数
                frequency = combined_text.count(keyword_lower)
                if frequency > 1:
                    # 频率加成，但有上限
                    freq_bonus = min(frequency * 0.02, 0.1) * weight
                    score += freq_bonus
                    matches.append(f"频率加成 [{keyword}] (x{frequency}): +{freq_bonus:.2f}")
        
        # 2. 负向关键词惩罚（更严格的惩罚机制）
        if "negative_keywords" in config:
            negative_score = 0
            for keyword_tuple in config["negative_keywords"]:
                if isinstance(keyword_tuple, tuple) and len(keyword_tuple) >= 1:
                    keyword = keyword_tuple[0].lower()
                    neg_weight = keyword_tuple[1] if len(keyword_tuple) > 1 else 1.0
                else:
                    # 兼容字符串格式
                    keyword = str(keyword_tuple).lower()
                    neg_weight = 1.0
                
                # 检查负向关键词是否出现在文本中
                if keyword in combined_text:
                    # 计算惩罚分数
                    penalty = neg_weight * 1.2
                    negative_score += penalty
                    matches.append(f"负向匹配 [{keyword}]: -{penalty:.2f}")
            
            # 使用更平滑的惩罚函数
            if negative_score > 0:
                original_score = score
                # 上下文感知的负向关键词处理
                # 检查负向关键词的上下文，判断是否存在否定词或对立词
                context_adjustment = 1.0
                
                # 检查否定词和对立词
                negation_words = ["not", "without", "no", "non", "instead of", "rather than", "unlike"]
                opposition_words = ["but", "however", "although", "despite", "contrary to"]
                
                # 如果存在否定词或对立词，减少惩罚
                for neg_word in negation_words:
                    if neg_word + " " + keyword_lower in combined_text or neg_word + "-" + keyword_lower in combined_text:
                        context_adjustment = 0.5  # 大幅减少惩罚
                        matches.append(f"检测到否定上下文: '{neg_word} {keyword_lower}', 惩罚减少")
                        break
                
                for opp_word in opposition_words:
                    if opp_word in combined_text and combined_text.find(opp_word) < combined_text.find(keyword_lower):
                        # 如果对立词在关键词之前，减少惩罚
                        context_adjustment = 0.7
                        matches.append(f"检测到对立上下文: '{opp_word}... {keyword_lower}', 惩罚部分减少")
                        break
                
                # 应用上下文调整
                negative_score *= context_adjustment
                
                # 使用改进的惩罚函数
                # 对于较小的负向分数使用线性惩罚，对于较大的负向分数使用指数惩罚
                if negative_score < 0.5:
                    # 轻微负向分数使用线性惩罚
                    penalty_factor = 1 - negative_score * 0.3
                else:
                    # 较大负向分数使用指数惩罚
                    penalty_factor = math.exp(-negative_score * 0.8)
                
                score *= penalty_factor
                penalty = original_score - score
                matches.append(f"负向惩罚总计: -{penalty:.2f} (因子: {penalty_factor:.2f}, 上下文调整: {context_adjustment:.1f})")
        
        # 3. 应用类别优先级缩放
        priority = config.get("priority", 0)
        if priority > 0:
            priority_bonus = score * (priority * 0.12)  # 优先级加成更明显
            score += priority_bonus
            matches.append(f"优先级加成 (级别 {priority}): +{priority_bonus:.2f}")
        
        # 记录得分和匹配详情
        if score > 0:
            scores[category] = score
            match_details[category] = matches
    
    # 4. 分类决策逻辑
    # 验证最低置信度阈值
    max_score = max(scores.values()) if scores else 0
    if max_score < 0.05:
        return []
    
    # 第一步：检测强信号关键词，直接分类某些明确的论文
    combined_text = (title + " " + abstract).lower()
    
    # 强信号关键词检测 - 优先考虑用户指定的基础研究类别
    strong_signals = {
        "生成式视觉与内容创建 (Generative Vision & Content Creation)": [
            "diffusion", "generative", "gan", "vae", "synthesis", "text-to-image", "image generation", 
            "stable diffusion", "latent diffusion", "text to image", "image to image", "style transfer",
            "扩散模型", "生成模型", "图像生成", "文本到图像", "图像到图像", "风格迁移"
        ],
        "三维重建与几何感知 (3D Reconstruction & Geometric Perception)": [
            "3d reconstruction", "point cloud", "depth estimation", "nerf", "3d vision", "geometry", "3d shape", 
            "novel view synthesis", "multi-view", "stereo matching", "structure from motion", "sfm", "slam",
            "三维重建", "点云", "深度估计", "三维视觉", "几何", "新视角合成", "多视角", "立体匹配", "运动结构"
        ],
        "动态视觉与时序建模 (Dynamic Vision & Temporal Modeling)": [
            "video", "temporal", "motion", "tracking", "optical flow", "action recognition", "trajectory", 
            "sequence", "dynamic", "temporal consistency", "video generation", "video prediction",
            "视频", "时序", "运动", "跟踪", "光流", "动作识别", "轨迹", "序列", "动态", "时间一致性"
        ],
        "多模态视觉与跨模态学习 (Multimodal Vision & Cross-modal Learning)": [
            "multimodal", "cross-modal", "vision-language", "text-vision", "audio-visual", "multi-modal", 
            "vision and language", "clip", "align", "多模态", "跨模态", "视觉语言", "文本视觉", "音视频", "视觉与语言"
        ],
        "模型优化与系统鲁棒性 (Model Optimization & System Robustness)": [
            "adversarial", "robustness", "optimization", "quantization", "pruning", "compression", "distillation", 
            "knowledge distillation", "model compression", "efficient inference", "low-rank", "sparsity",
            "对抗", "鲁棒性", "优化", "量化", "剪枝", "压缩", "蒸馏", "知识蒸馏", "模型压缩", "高效推理", "低秩", "稀疏性"
        ],
        "效率学习与适应性智能 (Efficient Learning & Adaptive Intelligence)": [
            "few-shot", "zero-shot", "meta-learning", "transfer learning", "domain adaptation", "continual learning", 
            "lifelong learning", "incremental learning", "curriculum learning", "active learning", "self-supervised",
            "少样本", "零样本", "元学习", "迁移学习", "域适应", "持续学习", "终身学习", "增量学习", "课程学习", "主动学习", "自监督"
        ],
        "静态图像理解与语义解析 (Static Image Understanding & Semantic Analysis)": [
            "object detection", "semantic segmentation", "instance segmentation", "scene understanding",
            "visual reasoning", "object recognition", "image classification", "scene parsing", "panoptic segmentation",
            "目标检测", "语义分割", "实例分割", "场景理解", "视觉推理", "目标识别", "图像分类", "全景分割"
        ]
    }
    
    # 检查强信号关键词
    for category, keywords in strong_signals.items():
        matches = sum(1 for kw in keywords if kw in combined_text)
        if matches >= 2 and category in scores and scores[category] >= 0.05:
            subcategory = get_subcategory(title, abstract, category, scores[category])
            return [(category, scores[category], subcategory)]
    
    # 第二步：收集所有潜在的候选类别
    all_candidates = []
    
    # 定义所有类别及其基础阈值 - 按照用户指定的优先级排序
    all_categories = {
        # 基础研究类别 - 优先级最高的类别
        "生成式视觉与内容创建 (Generative Vision & Content Creation)": 0.06,  # 降低阈值，鼓励分类
        "三维重建与几何感知 (3D Reconstruction & Geometric Perception)": 0.06,  # 降低阈值，鼓励分类
        "动态视觉与时序建模 (Dynamic Vision & Temporal Modeling)": 0.06,  # 降低阈值，鼓励分类
        "多模态视觉与跨模态学习 (Multimodal Vision & Cross-modal Learning)": 0.06,  # 降低阈值，鼓励分类
        "模型优化与系统鲁棒性 (Model Optimization & System Robustness)": 0.06,  # 降低阈值，鼓励分类
        "效率学习与适应性智能 (Efficient Learning & Adaptive Intelligence)": 0.06,  # 降低阈值，鼓励分类
        "静态图像理解与语义解析 (Static Image Understanding & Semantic Analysis)": 0.06,  # 降低阈值，鼓励分类
        
        # 其他基础研究类别
        "视觉表征学习与大模型 (Visual Representation Learning & Foundation Models)": 0.25,  # 提高阈值，减少过度分类
        "感知-动作智能与主动视觉 (Perception-Action Intelligence & Active Vision)": 0.10,
        "前沿视觉理论与跨学科融合 (Advanced Vision Theory & Interdisciplinary Integration)": 0.10,
        
        # 应用类别 - 需要更强的信号
        "生物医学影像计算 (Biomedical Image Computing)": 3.0,  # 进一步提高阈值
        "智能交通与自主系统 (Intelligent Transportation & Autonomous Systems)": 3.0,  # 进一步提高阈值
        "工业视觉与遥感感知 (Industrial Vision & Remote Sensing)": 3.0,  # 进一步提高阈值
        "交互媒体与虚拟现实技术 (Interactive Media & Extended Reality)": 3.0   # 进一步提高阈值
    }
    
    # 收集所有符合阈值的候选类别
    for category, threshold in all_categories.items():
        if category in scores and scores[category] >= threshold:
            subcategory = get_subcategory(title, abstract, category, scores[category])
            all_candidates.append((category, scores[category], subcategory))
    
    # 如果没有任何候选类别，返回空
    if not all_candidates:
        return []
    
    # 第二步：对候选类别进行筛选和优化    
    # 如果只有一个候选类别，直接返回
    if len(all_candidates) == 1:
        # print(f"✅ 唯一候选，直接分类: {all_candidates[0][0]}")
        return all_candidates
    
    # 第三步：多候选类别的智能筛选
    # 3.1 优先选择基础研究类别（除非应用类别得分明显更高）
    basic_research_candidates = [
        cat for cat in all_candidates 
        if not any(app in cat[0] for app in ["生物医学", "智能交通", "工业视觉", "交互媒体"])
    ]
    
    application_candidates = [
        cat for cat in all_candidates 
        if any(app in cat[0] for app in ["生物医学", "智能交通", "工业视觉", "交互媒体"])
    ]
    
    # 3.2 检查标题中是否明确提到应用场景
    title_lower = title.lower()
    has_application_in_title = any(term in title_lower for term in [
        "medical", "healthcare", "clinical", "autonomous", "driving", "vehicle", 
        "industrial", "inspection", "remote sensing", "satellite", "aerial", "drone",
        "ar", "vr", "augmented reality", "virtual reality", "mixed reality", "xr",
        "医学", "医疗", "自主", "驾驶", "车辆", "工业", "检测", "遥感", "卫星", "航空", "无人机",
        "增强现实", "虚拟现实", "混合现实"
    ])
    
    # 3.3 根据用户指定的优先级类别进行筛选
    priority_categories = [
        "生成式视觉与内容创建", "三维重建与几何感知", "动态视觉与时序建模",
        "多模态视觉与跨模态学习", "模型优化与系统鲁棒性", "效率学习与适应性智能",
        "静态图像理解与语义解析"
    ]
    
    # 检查是否有优先级类别
    priority_candidates = [cat for cat in basic_research_candidates if any(pc in cat[0] for pc in priority_categories)]
    
    # 3.4 决策逻辑
    if priority_candidates:
        # 如果有优先级类别，选择这些类别
        candidates_to_consider = priority_candidates
        # print(f"🔝 选择优先级基础研究类别")
    elif basic_research_candidates and application_candidates:
        max_basic_score = max(basic_research_candidates, key=lambda x: x[1])[1]
        max_app_score = max(application_candidates, key=lambda x: x[1])[1]
        
        # 如果标题中明确提到应用场景且应用类别得分足够高
        if has_application_in_title and max_app_score > max_basic_score * 1.5:
            candidates_to_consider = application_candidates
            # print(f"🏭 标题中有应用场景且得分较高 ({max_app_score:.3f} vs {max_basic_score:.3f})")
        # 如果应用类别得分压倒性优势
        elif max_app_score > max_basic_score * 3.0:
            candidates_to_consider = application_candidates
            # print(f"🏭 应用类别得分压倒性优势 ({max_app_score:.3f} vs {max_basic_score:.3f})")
        else:
            candidates_to_consider = basic_research_candidates
            # print(f"🔬 优先基础研究类别 ({max_basic_score:.3f} vs {max_app_score:.3f})")
    elif basic_research_candidates:
        candidates_to_consider = basic_research_candidates
        # print(f"🔬 仅有基础研究候选")
    else:
        candidates_to_consider = application_candidates
        # print(f"🏭 仅有应用类别候选")
    
    # 3.3 在选定的候选类别中进行最终选择
    if len(candidates_to_consider) == 1:
        # print(f"✅ 筛选后唯一候选: {candidates_to_consider[0][0]}")
        return candidates_to_consider
    
    # 3.4 多个候选类别时，使用更智能的决策逻辑
    try:
        # 检查标题和摘要中的关键词，更精确地判断论文的真正类别
        combined_text = (title + " " + abstract).lower()
        
        # 用户指定的优先级类别关键词
        priority_keywords = {
            "生成式视觉与内容创建": ["diffusion", "generative", "gan", "vae", "synthesis", "text-to-image", "image generation"],
            "三维重建与几何感知": ["3d", "point cloud", "depth", "nerf", "geometry", "stereo", "reconstruction"],
            "动态视觉与时序建模": ["video", "temporal", "motion", "tracking", "optical flow", "action"],
            "多模态视觉与跨模态学习": ["multimodal", "cross-modal", "vision-language", "text-vision", "audio-visual"],
            "模型优化与系统鲁棒性": ["adversarial", "robustness", "optimization", "quantization", "pruning", "compression"],
            "效率学习与适应性智能": ["few-shot", "zero-shot", "meta-learning", "transfer", "domain adaptation", "continual"],
            "静态图像理解与语义解析": ["detection", "segmentation", "recognition", "classification", "scene understanding"]
        }
        
        # 检查标题中的关键词匹配
        for cat in candidates_to_consider:
            category_name = cat[0].split(" (")[0]  # 去除英文部分
            if category_name in priority_keywords:
                keywords = priority_keywords[category_name]
                matches = sum(1 for kw in keywords if kw in combined_text)
                if matches >= 3:  # 如果标题和摘要中有多个关键词匹配，直接选择该类别
                    return [cat]
        
        # 使用ChatGLM做决策
        final_category = chatglm.decide_category(title, abstract, candidates_to_consider)
        
        if final_category:
            # 找到对应的完整信息
            for cat, score, subcat in candidates_to_consider:
                if cat == final_category:
                    # 如果标题中明确提到了这个类别的关键词，增强这个决策
                    category_name = cat.split(" (")[0]  # 去除英文部分
                    if category_name in priority_keywords:
                        keywords = priority_keywords[category_name]
                        if any(kw in title.lower() for kw in keywords):
                            return [(cat, score, subcat)]
                    
                    # 如果ChatGLM选择了视觉表征学习类别，检查是否有更适合的优先级类别
                    if "视觉表征学习与大模型" in cat:
                        # 检查是否有优先级类别的候选
                        for priority_cat in candidates_to_consider:
                            priority_name = priority_cat[0].split(" (")[0]
                            if priority_name in priority_keywords and priority_name != "视觉表征学习与大模型":
                                # 如果有优先级类别且得分不低于视觉表征学习的70%
                                if priority_cat[1] >= cat[1] * 0.7:
                                    return [priority_cat]
                    
                    return [(cat, score, subcat)]
        
        # print(f"⚠️ ChatGLM决策失败，选择得分最高的候选")
    except Exception as e:
        print(f"⚠️ ChatGLM调用失败: {e}，选择得分最高的候选")
    
    # 3.5 如果ChatGLM失败，选择得分最高的候选
    best_candidate = max(candidates_to_consider, key=lambda x: x[1])
    # print(f"📊 最高得分候选: {best_candidate[0]} ({best_candidate[1]:.3f})")
    return [best_candidate]


def calculate_category_relation(category1, category2, categories_config):
    """
    计算两个类别之间的相关性
    
    Args:
        category1: 第一个类别名称
        category2: 第二个类别名称
        categories_config: 类别配置字典
        
    Returns:
        float: 相关性分数 (0-1)，越高表示越相关
    """
    # 如果类别相同，相关性为1
    if category1 == category2:
        return 1.0
    
    # 获取两个类别的关键词
    keywords1 = set()
    keywords2 = set()
    
    if category1 in categories_config and "keywords" in categories_config[category1]:
        keywords1 = {kw[0].lower() for kw in categories_config[category1]["keywords"] if isinstance(kw, tuple)}
    
    if category2 in categories_config and "keywords" in categories_config[category2]:
        keywords2 = {kw[0].lower() for kw in categories_config[category2]["keywords"] if isinstance(kw, tuple)}
    
    # 如果任一类别没有关键词，返回0
    if not keywords1 or not keywords2:
        return 0.0
    
    # 计算关键词重叠
    overlap = keywords1.intersection(keywords2)
    
    # 使用Jaccard相似度计算相关性
    similarity = len(overlap) / len(keywords1.union(keywords2))
    
    # 使用最新的类别定义生成相关类别对
    related_pairs = get_related_category_pairs()
    
    # 检查是否为预定义的相关类别对
    for pair in related_pairs:
        if (category1.startswith(pair[0]) and category2.startswith(pair[1])) or \
           (category1.startswith(pair[1]) and category2.startswith(pair[0])):
            # 增加相关性分数
            similarity += 0.2
            break
    
    return min(similarity, 1.0)  # 确保不超过1


def process_paper(paper, glm_helper, target_date):
    """处理单篇论文的所有分析任务

    Args:
        paper: ArXiv论文对象
        glm_helper: ChatGLM助手实例
        target_date: 目标日期

    Returns:
        Dict: 包含论文信息的字典，如果论文不符合日期要求则返回None
    """
    try:
        # 获取论文信息
        title = paper.title
        abstract = paper.summary
        paper_url = paper.entry_id
        author_list = paper.authors
        authors = [author.name for author in author_list]
        authors_str = ', '.join(authors[:8]) + (' .etc.' if len(authors) > 8 else '')  # 限制作者显示数量，超过8个显示etc.
        published = paper.published
        updated = paper.updated
        
        # 检查发布日期或更新日期是否匹配目标日期
        published_date = published.date()
        updated_date = updated.date()
        if published_date != target_date and updated_date != target_date:
            return None
            
        # 获取PDF链接
        pdf_url = next(
            (link.href for link in paper.links if link.title == "pdf"), None)
        
        # 初始化默认值，避免异常时未定义
        github_link = "None"
        category = "其他 (Others)"  # 修改默认值为带英文的格式
        subcategory = "未指定"
        title_cn = f"[翻译失败] {title}"
        analysis = {}

        # 并行执行耗时任务
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                # 提交所有任务
                github_future = executor.submit(extract_github_link, abstract)
                analysis_future = executor.submit(
                    glm_helper.analyze_paper_contribution, title, abstract)
                title_cn_future = executor.submit(
                    glm_helper.translate_title, title)

                # 等待所有任务完成
                github_link = github_future.result() or "None"
                analysis = analysis_future.result() or {}
                title_cn = title_cn_future.result() or f"[翻译失败] {title}"
        except Exception as e:
            logger.error(f"并行处理任务时出错: {str(e)}")
            # 继续处理，使用默认值
        
        # 使用基于关键词的分类方法
        try:
            # 使用新的分类函数
            category_results = get_category_by_keywords(title, abstract, CATEGORY_KEYWORDS)
            
            if category_results:
                # 获取主类别和得分
                result_item = category_results[0]
                
                # 兼容多种返回格式：(category, score) 或 (category, score, subcategory) 或 (category, score, subcategory, explanation)
                if len(result_item) >= 4:  # 新格式，包含解释
                    main_category, main_score, sub_category_tuple, explanation = result_item
                elif len(result_item) == 3:  # 旧格式，包含子类别
                    main_category, main_score, sub_category_tuple = result_item
                    explanation = None
                else:  # 最简单的格式
                    main_category, main_score = result_item
                    sub_category_tuple = None
                    explanation = None
                    
                category = main_category
                
                # 处理子类别
                if sub_category_tuple:
                    subcategory_name, subcategory_score = sub_category_tuple
                    subcategory = subcategory_name
                else:
                    subcategory = "未指定"
                    
                # 不输出分类结果信息，减少日志干扰
            else:
                # 如果没有匹配的类别，使用默认类别
                category = "其他 (Others)"
                subcategory = "未指定"
        except Exception as e:
            logger.error(f"分类论文时出错: {str(e)}")
            traceback.print_exc()
            category = "其他 (Others)"
            subcategory = "未指定"

        paper_info = {
            'title': title,
            'title_zh': title_cn,  # 修改键名为 title_zh 以匹配其他函数
            'abstract': abstract,
            'authors': authors_str,
            'pdf_url': pdf_url,
            'github_url': github_link,  # 修改键名为 github_url 以匹配其他函数
            'url': paper_url,  # 添加 arxiv URL
            'category': category,
            'subcategory': subcategory,  # 添加子类别信息
            'published': published,
            'updated': updated,
            'is_updated': updated_date == target_date and published_date != target_date
        }

        # 合并分析结果
        if analysis:
            paper_info.update(analysis)

        return paper_info

    except Exception as e:
        logger.error(f"\u5904\u7406\u8bba\u6587\u65f6\u51fa\u9519: {str(e)}")
        return None


def get_subcategory(title: str, abstract: str, main_category: str, main_score: float) -> Optional[Tuple[str, float]]:
    """
    在确定主类别后，进一步确定子类别
    
    Args:
        title: 论文标题
        abstract: 论文摘要
        main_category: 主类别
        main_score: 主类别得分
        
    Returns:
        Optional[Tuple[str, float]]: 子类别及其得分，如果无法确定则返回None
    """
    # 增强的文本预处理
    title_lower = title.lower()
    abstract_lower = abstract.lower()
    combined_text = title_lower + " " + abstract_lower
    
    # 使用NLTK进行多级文本预处理
    processed_title = preprocess_text(title)
    processed_abstract = preprocess_text(abstract)
    processed_combined = processed_title + " " + processed_abstract
    
    # 创建N-gram版本的文本用于短语匹配
    # 这有助于捕获多词短语，即使它们的顺序或形式略有不同
    from nltk.util import ngrams
    import re
    
    # 清理并标准化文本用于N-gram处理
    clean_title = re.sub(r'[^\w\s]', ' ', title_lower)
    clean_abstract = re.sub(r'[^\w\s]', ' ', abstract_lower)
    
    # 生成2-gram和3-gram
    title_words = clean_title.split()
    abstract_words = clean_abstract.split()
    
    # 生成2-gram
    title_bigrams = [' '.join(ng) for ng in ngrams(title_words, 2)] if len(title_words) >= 2 else []
    abstract_bigrams = [' '.join(ng) for ng in ngrams(abstract_words, 2)] if len(abstract_words) >= 2 else []
    
    # 生成3-gram
    title_trigrams = [' '.join(ng) for ng in ngrams(title_words, 3)] if len(title_words) >= 3 else []
    abstract_trigrams = [' '.join(ng) for ng in ngrams(abstract_words, 3)] if len(abstract_words) >= 3 else []
    
    # 合并所有N-gram
    all_ngrams = set(title_bigrams + abstract_bigrams + title_trigrams + abstract_trigrams)
    
    # 检查主类别是否有子类别定义
    if main_category in CATEGORY_THRESHOLDS and "subcategories" in CATEGORY_THRESHOLDS[main_category]:
        subcategories = CATEGORY_THRESHOLDS[main_category]["subcategories"]
        
        # 计算每个子类别的得分
        subcategory_scores = {}
        for subcategory_name, subcategory_threshold in subcategories.items():
            # 提取子类别名称中的关键词
            subcategory_keywords = subcategory_name.lower().split()
            score = 0.0
            
            # 完整短语精确匹配（最高权重）
            if subcategory_name.lower() in combined_text:
                score += 3.5  # 大幅增加精确匹配的权重，从2.5提高到3.5
            elif subcategory_name.lower() in title_lower:
                score += 4.0  # 如果子类别名称直接出现在标题中，给予更高权重
            
            # 使用预处理后的文本进行匹配
            processed_subcategory = preprocess_text(subcategory_name)
            if processed_subcategory in processed_combined:
                score += 2.5  # 增加预处理文本匹配的权重，从1.8提高到2.5
            elif processed_subcategory in processed_title:
                score += 3.0  # 如果预处理后的子类别名称出现在标题中
            
            # 增强的关键词匹配（语义相似度和上下文感知）
            for keyword in subcategory_keywords:
                if len(keyword) > 3:  # 忽略过短的词
                    # 原始文本匹配
                    if keyword in title_lower:
                        score += 1.5  # 标题中的精确匹配
                    elif keyword in abstract_lower:
                        score += 0.8  # 摘要中的精确匹配
                    
                    # 预处理文本匹配
                    processed_keyword = preprocess_text(keyword)
                    if processed_keyword in processed_title:
                        score += 1.2  # 预处理后的标题匹配
                    elif processed_keyword in processed_abstract:
                        score += 0.6  # 预处理后的摘要匹配
                    
                    # N-gram匹配（捕获短语变体）
                    for ngram in all_ngrams:
                        if keyword in ngram:
                            score += 0.4  # N-gram中的关键词匹配
                            break
                    
                    # 词根匹配（处理词形变化）
                    keyword_root = preprocess_text(keyword)
                    for word in processed_title.split():
                        if keyword_root in word and len(keyword_root) > 4:  # 确保足够长以避免误匹配
                            score += 0.3
                            break
                    for word in processed_abstract.split():
                        if keyword_root in word and len(keyword_root) > 4:
                            score += 0.2
                            break
            
            # 大幅降低子类别阈值，确保大多数论文能被分配到子类别
            if score > 0:
                # 子类别得分需要达到主类别得分的一定比例，但大幅降低要求
                relative_threshold = main_score * 0.15 * subcategory_threshold  # 从0.25降低到0.15
                if score >= relative_threshold or score >= 0.5:  # 添加绝对分数阈值
                    subcategory_scores[subcategory_name] = score
        
        # 返回得分最高的子类别
        if subcategory_scores:
            best_subcategory = max(subcategory_scores.items(), key=lambda x: x[1])
            return best_subcategory
    
    return None


def calculate_category_relation(category1, category2, categories_config):
    """
    计算两个类别之间的相关性
    
    Args:
        category1: 第一个类别名称
        category2: 第二个类别名称
        categories_config: 类别配置字典
        
    Returns:
        float: 相关性分数 (0-1)，越高表示越相关
    """
    # 如果类别相同，相关性为1
    if category1 == category2:
        return 1.0
    
    # 获取两个类别的关键词
    keywords1 = set()
    keywords2 = set()
    
    if category1 in categories_config and "keywords" in categories_config[category1]:
        keywords1 = {kw[0].lower() for kw in categories_config[category1]["keywords"] if isinstance(kw, tuple)}
    
    if category2 in categories_config and "keywords" in categories_config[category2]:
        keywords2 = {kw[0].lower() for kw in categories_config[category2]["keywords"] if isinstance(kw, tuple)}
    
    # 如果任一类别没有关键词，返回0
    if not keywords1 or not keywords2:
        return 0.0
    
    # 计算关键词重叠
    overlap = keywords1.intersection(keywords2)
    
    # 使用Jaccard相似度计算相关性
    similarity = len(overlap) / len(keywords1.union(keywords2))
    
    # 使用最新的类别定义生成相关类别对
    related_pairs = get_related_category_pairs()
    
    # 检查是否为预定义的相关类别对
    for pair in related_pairs:
        if (category1.startswith(pair[0]) and category2.startswith(pair[1])) or \
           (category1.startswith(pair[1]) and category2.startswith(pair[0])):
            # 增加相关性分数
            similarity += 0.2
            break
    
    return min(similarity, 1.0)  # 确保不超过1


def process_paper(paper, glm_helper, target_date):
    """处理单篇论文的所有分析任务

    Args:
        paper: ArXiv论文对象
        glm_helper: ChatGLM助手实例
        target_date: 目标日期

    Returns:
        Dict: 包含论文信息的字典，如果论文不符合日期要求则返回None
    """
    try:
        # 获取论文信息
        title = paper.title
        abstract = paper.summary
        paper_url = paper.entry_id
        author_list = paper.authors
        authors = [author.name for author in author_list]
        authors_str = ', '.join(authors[:8]) + (' .etc.' if len(authors) > 8 else '')  # 限制作者显示数量，超过8个显示etc.
        published = paper.published
        updated = paper.updated
        
        # 检查发布日期或更新日期是否匹配目标日期
        published_date = published.date()
        updated_date = updated.date()
        if published_date != target_date and updated_date != target_date:
            return None
            
        # 获取PDF链接
        pdf_url = next(
            (link.href for link in paper.links if link.title == "pdf"), None)
        
        # 初始化默认值，避免异常时未定义
        github_link = "None"
        category = "其他 (Others)"  # 修改默认值为带英文的格式
        subcategory = "未指定"
        title_cn = f"[翻译失败] {title}"
        analysis = {}

        # 并行执行耗时任务
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                # 提交所有任务
                github_future = executor.submit(extract_github_link, abstract)
                analysis_future = executor.submit(
                    glm_helper.analyze_paper_contribution, title, abstract)
                title_cn_future = executor.submit(
                    glm_helper.translate_title, title)

                # 等待所有任务完成
                github_link = github_future.result() or "None"
                analysis = analysis_future.result() or {}
                title_cn = title_cn_future.result() or f"[翻译失败] {title}"
        except Exception as e:
            logger.error(f"并行处理任务时出错: {str(e)}")
            # 继续处理，使用默认值
        
        # 使用基于关键词的分类方法
        try:
            # 使用新的分类函数
            category_results = get_category_by_keywords(title, abstract, CATEGORY_KEYWORDS)
            
            if category_results:
                # 获取主类别和得分
                result_item = category_results[0]
                
                # 兼容多种返回格式：(category, score) 或 (category, score, subcategory) 或 (category, score, subcategory, explanation)
                if len(result_item) >= 4:  # 新格式，包含解释
                    main_category, main_score, sub_category_tuple, explanation = result_item
                elif len(result_item) == 3:  # 旧格式，包含子类别
                    main_category, main_score, sub_category_tuple = result_item
                    explanation = None
                else:  # 最简单的格式
                    main_category, main_score = result_item
                    sub_category_tuple = None
                    explanation = None
                    
                category = main_category
                
                # 处理子类别
                if sub_category_tuple:
                    subcategory_name, subcategory_score = sub_category_tuple
                    subcategory = subcategory_name
                else:
                    subcategory = "未指定"
                    
                # 不输出分类结果信息，减少日志干扰
            else:
                # 如果没有匹配的类别，使用默认类别
                category = "其他 (Others)"
                subcategory = "未指定"
        except Exception as e:
            logger.error(f"分类论文时出错: {str(e)}")
            traceback.print_exc()
            category = "其他 (Others)"
            subcategory = "未指定"

        paper_info = {
            'title': title,
            'title_zh': title_cn,  # 修改键名为 title_zh 以匹配其他函数
            'abstract': abstract,
            'authors': authors_str,
            'pdf_url': pdf_url,
            'github_url': github_link,  # 修改键名为 github_url 以匹配其他函数
            'url': paper_url,  # 添加 arxiv URL
            'category': category,
            'subcategory': subcategory,  # 添加子类别信息
            'published': published,
            'updated': updated,
            'is_updated': updated_date == target_date and published_date != target_date
        }

        # 合并分析结果
        if analysis:
            paper_info.update(analysis)

        return paper_info

    except Exception as e:
        logger.error(f"\u5904\u7406\u8bba\u6587\u65f6\u51fa\u9519: {str(e)}")
        return None


def get_cv_papers():
    """
    获取并分类 CV 论文的主函数
    """
    logger.info("="*50)
    logger.info(f"开始获取CV论文 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*50)
    
    try:
        # 计算目标日期
        target_date = (datetime.now() - timedelta(days=QUERY_DAYS_AGO)).date()
        logger.info(f"\n📅 目标日期: {target_date}")
        logger.info(f"📊 论文数量: {MAX_RESULTS}")
        logger.info(f"🧵 线程数量: {MAX_WORKERS}\n")

        # 初始化ChatGLM助手
        logger.info("🤖 初始化ChatGLM助手...")
        glm_helper = ChatGLMHelper()

        # 初始化arxiv客户端
        logger.info("🔄 初始化arXiv客户端...")
        client = arxiv.Client(
            page_size=100,
            delay_seconds=3,
            num_retries=5
        )

        # 构建查询
        search = arxiv.Search(
            query='cat:cs.CV',
            max_results=MAX_RESULTS,
            sort_by=arxiv.SortCriterion.LastUpdatedDate,
            sort_order=arxiv.SortOrder.Descending
        )

        # 创建线程池
        total_papers = 0
        papers_by_category = defaultdict(list)
        
        # 确保"其他 (Others)"类别总是存在
        papers_by_category["其他 (Others)"]  # 初始化空列表

        # 使用线程池并行处理论文
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 创建进度条并开始获取论文
            logger.info("\n🔍 开始获取论文...")
            results = client.results(search)  # 从 arXiv API 获取论文结果
            
            # 创建总进度条
            total_pbar = tqdm(
                total=MAX_RESULTS,
                desc="总进度",
                unit="篇",
                position=0,
                leave=True
            )
            
            # 创建批处理进度条
            batch_pbar = tqdm(
                total=0,  # 初始值为0，后面会更新
                desc="当前批次",
                unit="篇",
                position=1,
                leave=True
            )
            
            # 批量处理论文
            batch_size = 10  # 每批处理10篇论文
            papers = []
            futures = []
            
            for i, paper in enumerate(results):
                papers.append(paper)
                
                # 当收集到一批论文或达到最大数量时处理
                if len(papers) >= batch_size or i >= MAX_RESULTS - 1:
                    batch_pbar.reset()  # 重置批处理进度条
                    batch_pbar.total = len(papers)  # 设置正确的总数
                    
                    # 提交所有任务
                    batch_futures = [
                        executor.submit(process_paper, paper, glm_helper, target_date)
                        for paper in papers
                    ]
                    
                    # 等待当前批次完成
                    for future in as_completed(batch_futures):
                        paper_info = future.result()
                        if paper_info:  # 如果论文符合日期要求
                            total_papers += 1
                            category = paper_info['category']
                            papers_by_category[category].append(paper_info)
                            total_pbar.update(1)  # 更新总进度
                        batch_pbar.update(1)  # 更新批处理进度
                    
                    # 清空当前批次
                    papers = []
                
                # 如果达到最大数量，停止获取
                if i >= MAX_RESULTS - 1:
                    break
            
            # 处理剩余的论文
            if papers:
                batch_pbar.reset()  # 重置批处理进度条
                batch_pbar.total = len(papers)  # 设置正确的总数
                
                # 提交所有任务
                futures = [
                    executor.submit(process_paper, paper, glm_helper, target_date)
                    for paper in papers
                ]
                
                # 等待所有任务完成
                for future in as_completed(futures):
                    paper_info = future.result()
                    if paper_info:  # 如果论文符合日期要求
                        total_papers += 1
                        category = paper_info['category']
                        papers_by_category[category].append(paper_info)
                        total_pbar.update(1)  # 更新总进度
                    batch_pbar.update(1)  # 更新批处理进度

            # 关闭进度条
            batch_pbar.close()
            total_pbar.close()

        if total_papers == 0:
            logger.warning(f"没有找到{target_date}发布的论文。")
            return

        # 输出论文统计信息
        logger.info(f"\n📊 论文统计信息：")
        logger.info(f"{'='*50}")
        
        # 按论文数量降序排序类别
        sorted_categories = sorted(
            papers_by_category.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )
        
        # 统计总论文数
        total_papers = sum(len(papers) for papers in papers_by_category.values())
        
        # 一级分类和对应的二级分类统计
        for category, papers in sorted_categories:
            # 即使没有论文，也要打印"其他"类别
            if len(papers) == 0 and category != "其他 (Others)":
                continue
                
            # 输出一级分类标题
            logger.info(f"\n【{category}】")
            
            # 如果不是"其他"类别，将没有子类别的论文移动到"其他"类别
            if category != "其他 (Others)":
                # 分离有子类别的论文和无子类别的论文
                papers_with_subcategory = []
                papers_without_subcategory = []
                
                for paper in list(papers):  # 创建副本以避免在遍历时修改
                    subcategory = paper.get('subcategory', '')
                    if not subcategory or subcategory == "未指定":
                        papers_without_subcategory.append(paper)
                        # 将没有子类别的论文移动到"其他"类别
                        papers_by_category["其他 (Others)"].append(paper)
                        papers.remove(paper)  # 从当前类别中移除
                    else:
                        papers_with_subcategory.append(paper)
            
            # 如果当前类别下没有论文，跳过
            if len(papers) == 0 and category != "其他 (Others)":
                continue
            
            # 按子类别分组论文
            papers_by_subcategory = defaultdict(list)
            for paper in papers:  # 使用更新后的papers
                subcategory = paper.get('subcategory', '')
                if subcategory and subcategory != "未指定":
                    papers_by_subcategory[subcategory].append(paper)
                else:
                    # 对于没有子类别的论文，如果是"其他"类别，显示为"未分类"
                    papers_by_subcategory["未分类"].append(paper)
            
            # 按论文数量降序排序子类别
            sorted_subcategories = sorted(
                papers_by_subcategory.items(),
                key=lambda x: len(x[1]),
                reverse=True
            )
            
            # 打印一级分类总数
            num_new = sum(1 for p in papers if not p['is_updated'])
            num_updated = sum(1 for p in papers if p['is_updated'])
            logger.info(f"总计: {len(papers):3d} 篇 (🔥 {num_new:3d} 新发布, 📝 {num_updated:3d} 更新)")
            
            # 打印子类别统计
            for subcategory, subpapers in sorted_subcategories:
                num_new = sum(1 for p in subpapers if not p['is_updated'])
                num_updated = sum(1 for p in subpapers if p['is_updated'])
                logger.info(
                    f"└─ {subcategory:15s}: {len(subpapers):3d} 篇 (🔥 {num_new:3d} 新发布, 📝 {num_updated:3d} 更新)")
            
            # 不再打印"直接归类"，因为这些论文已经被移动到"其他 (Others)"类别中
        
        logger.info(f"\n{'='*50}")
        logger.info(f"总计: {total_papers} 篇")
        
        # 保存结果到Markdown文件
        logger.info("\n💾 正在保存结果到Markdown文件...")
        save_papers_to_markdown(papers_by_category, target_date)
        
        logger.info("\n" + "="*50)
        logger.info(f"CV论文获取完成 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*50 + "\n")

    except Exception as e:
        logger.error("\n❌ 处理CV论文时出错:")
        logger.error(f"错误信息: {str(e)}")
        logger.error(f"发生时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        # 记录堆栈跟踪信息以便调试
        logger.debug(traceback.format_exc())
        raise  # 抛出异常以便查看详细错误信息


def save_papers_to_markdown(papers_by_category: dict, target_date):
    """保存论文信息到Markdown文件"""
    # 使用目标日期作为文件名
    filename = target_date.strftime("%Y-%m-%d") + ".md"
    year_month = target_date.strftime("%Y-%m")

    # 获取当前文件所在目录(scripts)的父级路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)

    # 设置基础目录，与scripts同级
    data_base = os.path.join(parent_dir, 'data')
    local_base = os.path.join(parent_dir, 'local')

    # 创建年月子目录
    data_year_month = os.path.join(data_base, year_month)
    local_year_month = os.path.join(local_base, year_month)

    # 创建所需的目录结构
    os.makedirs(data_year_month, exist_ok=True)
    os.makedirs(local_year_month, exist_ok=True)

    # 生成完整的文件路径
    table_filepath = os.path.join(data_year_month, filename)
    detailed_filepath = os.path.join(local_year_month, filename)

    # 生成标题
    title = f"## [UPDATED!] **{target_date.strftime('%Y-%m-%d')}** (Update Time)\n\n"

    # 保存表格格式的markdown文件到data/年-月目录
    with open(table_filepath, 'w', encoding='utf-8') as f:
        f.write(title)
        # f.write("\n## 论文列表\n\n")
        f.write(df_to_markdown_table(papers_by_category, target_date))

    # 保存详细格式的markdown文件到local/年-月目录
    with open(detailed_filepath, 'w', encoding='utf-8') as f:
        f.write(title)
        # f.write("\n## 论文详情\n\n")
        f.write(df_to_markdown_detailed(papers_by_category, target_date))

    logger.info(f"\n表格格式文件已保存到: {table_filepath}")
    logger.info(f"详细格式文件已保存到: {detailed_filepath}")


def generate_statistics_markdown(papers_by_category: dict) -> str:
    """生成统计信息的Markdown格式文本
    
    Args:
        papers_by_category: 按类别组织的论文字典
        
    Returns:
        str: Markdown格式的统计信息
    """
    markdown = "## 统计信息\n\n"
    
    # 计算总论文数
    total_papers = sum(len(papers) for papers in papers_by_category.values())
    markdown += f"**总论文数**: {total_papers} 篇\n\n"
    
    # 按论文数量降序排序类别
    sorted_categories = sorted(
        papers_by_category.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )
    
    # 一级分类统计
    markdown += "### 一级分类统计\n\n"
    markdown += "| 类别 | 论文数 | 新发布 | 更新 |\n"
    markdown += "|------|--------|--------|------|\n"
    
    for category, papers in sorted_categories:
        num_new = sum(1 for p in papers if not p['is_updated'])
        num_updated = sum(1 for p in papers if p['is_updated'])
        markdown += f"| {category} | {len(papers)} | {num_new} | {num_updated} |\n"
    
    # 二级分类统计
    markdown += "\n### 二级分类统计\n\n"
    
    for category, papers in sorted_categories:
        if len(papers) == 0:
            continue
            
        markdown += f"#### {category}\n\n"
        markdown += "| 子类别 | 论文数 | 新发布 | 更新 |\n"
        markdown += "|--------|--------|--------|------|\n"
        
        # 如果是"其他"类别，没有子类别的论文直接显示在主类别下
        papers_with_subcategory = []
        
        if category != "其他 (Others)":
            # 分离有子类别的论文和无子类别的论文
            for paper in papers:
                subcategory = paper.get('subcategory', '')
                if subcategory and subcategory != "未指定":
                    papers_with_subcategory.append(paper)
                # 没有子类别的论文已经被移动到"其他 (Others)"类别中
        else:
            # 对于"其他"类别，所有论文都直接处理
            papers_with_subcategory = papers
        
        # 按子类别分组有子类别的论文
        papers_by_subcategory = defaultdict(list)
        for paper in papers_with_subcategory:
            subcategory = paper.get('subcategory', '')
            papers_by_subcategory[subcategory].append(paper)
        
        # 按论文数量降序排序子类别
        sorted_subcategories = sorted(
            papers_by_subcategory.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )
        
        # 添加子类别统计
        for subcategory, subpapers in sorted_subcategories:
            num_new = sum(1 for p in subpapers if not p['is_updated'])
            num_updated = sum(1 for p in subpapers if p['is_updated'])
            markdown += f"| {subcategory} | {len(subpapers)} | {num_new} | {num_updated} |\n"
        
        markdown += "\n"
    
    return markdown


if __name__ == "__main__":    
    # 运行论文获取与分类程序
    logger.info(f"CV论文获取程序启动 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    get_cv_papers()
