"""获取CV论文"""
import os
import re
import json
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from collections import defaultdict
import categories_config
from categories_config import CATEGORY_DISPLAY_ORDER, CATEGORY_THRESHOLDS
from chatglm_helper import ChatGLMHelper
from typing import Dict, List, Tuple, Optional
import traceback
import arxiv

# 查询参数设置
QUERY_DAYS_AGO = 1          # 查询几天前的论文，0=今天，1=昨天，2=前天
MAX_RESULTS = 200           # 最大返回论文数量
MAX_WORKERS = 8            # 并行处理的最大线程数


def extract_github_link(text, paper_url=None, title=None, authors=None, pdf_url=None):
    """从文本中提取GitHub链接

    Args:
        text: 论文摘要文本
        paper_url: 论文URL（未使用）
        title: 论文标题（未使用）
        authors: 作者列表（未使用）
        pdf_url: PDF文件URL（未使用）

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
            
        # 添加一级类别标题
        markdown += f"\n## {category}\n\n"
        
        # 按子类别组织论文
        papers_by_subcategory = defaultdict(list)
        for paper in active_categories[category]:
            subcategory = paper.get('subcategory', '')
            if not subcategory or subcategory == "未指定":
                subcategory = "其他"
            papers_by_subcategory[subcategory].append(paper)
            
        # 处理每个子类别
        for subcategory, papers in papers_by_subcategory.items():
            # 添加二级类别标题
            markdown += f"\n### {subcategory}\n\n"
            
            # 创建表格头
            markdown += "|" + "|".join(headers) + "|\n"
            markdown += "|" + "|".join(["---"] * len(headers)) + "|\n"
            
            # 添加论文
            for paper in papers:
                # 确定论文状态
                if paper['is_updated']:
                    status = f"📝 更新"
                else:
                    status = f"🆕 发布"
                
                # 合并代码链接和核心贡献
                code_and_contribution = ""
                if paper['github_url'] != 'None':
                    code_and_contribution = f"[[代码]](<{paper['github_url']}>)"
                    if "核心贡献" in paper:
                        code_and_contribution += "<br />"
                if "核心贡献" in paper:
                    core_contribution = paper["核心贡献"]
                    if " | " in core_contribution:
                        items = core_contribution.split(" | ")
                        code_and_contribution += "<br />".join([f"- {item.strip()}" for item in items])
                    else:
                        code_and_contribution += core_contribution
                if not code_and_contribution:
                    code_and_contribution = 'None'
                
                # 准备每个字段的值
                values = [
                    status,
                    paper['title'],
                    paper.get('title_zh', ''),
                    paper['authors'],  # 已经是格式化好的字符串
                    f"<{paper['pdf_url']}>",
                    code_and_contribution,
                ]
                
                # 处理特殊字符
                values = [str(v).replace('\n', ' ').replace('|', '&#124;')
                          for v in values]
                
                # 添加到表格
                markdown += "|" + "|".join(values) + "|\n"
            
            # 在每个表格后添加空行
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
        for paper in active_categories[category]:
            subcategory = paper.get('subcategory', '')
            if not subcategory or subcategory == "未指定":
                subcategory = "其他"
            papers_by_subcategory[subcategory].append(paper)
        
        # 处理每个子类别
        for subcategory, papers in papers_by_subcategory.items():
            # 添加二级类别标题
            markdown += f"\n### {subcategory}\n\n"
            
            # 添加论文
            for idx, paper in enumerate(papers, 1):
                markdown += f'**index:** {idx}<br />\n'
                markdown += f'**Date:** {target_date.strftime("%Y-%m-%d")}<br />\n'
                markdown += f'**Title:** {paper["title"]}<br />\n'
                markdown += f'**Title_cn:** {paper.get("title_zh", "")}<br />\n'
                # 已经是格式化好的字符串
                markdown += f'**Authors:** {paper["authors"]}<br />\n'
                markdown += f'**PDF:** <{paper["pdf_url"]}><br />\n'

                # 合并代码链接和核心贡献
                markdown += '**Code/Contribution:**<br />\n'
                if paper["github_url"] != 'None':
                    markdown += f'- [[代码]](<{paper["github_url"]}>)<br />\n'
                if "核心贡献" in paper:
                    core_contribution = paper["核心贡献"]
                    if " | " in core_contribution:
                        for item in core_contribution.split(" | "):
                            markdown += f'- {item.strip()}<br />\n'
                    else:
                        markdown += f'- {core_contribution}<br />\n'
                elif paper["github_url"] == 'None':
                    markdown += 'None<br />\n'

                if "核心问题" in paper:
                    markdown += f'**Core Problem:** {paper["核心问题"]}<br />\n'

                markdown += "\n"

    return markdown


def get_category_by_keywords(title: str, abstract: str, categories_config: Dict) -> List[Tuple[str, float]]:
    """
    执行基于关键词匹配和优先级规则的层次化论文分类。
    
    Args:
        title (str): 论文标题，用于主要上下文分析
        abstract (str): 论文摘要，用于全面内容分析
        categories_config (Dict): 包含类别定义、关键词、权重和优先级的配置字典
    
    实现细节:
        1. 文本预处理:
           - 大小写标准化，确保匹配稳健性
           - 标题和摘要的组合分析，使用差异化权重
           - 分词和停用词过滤，提高匹配质量
        
        2. 评分机制:
           - 主要得分: 加权关键词匹配 (基础权重 0.15)
           - 标题加成: 标题匹配的额外权重 (0.08 权重)
           - 精确匹配加成: 完整短语匹配的额外权重
           - 优先级乘数: 类别特定重要性缩放
           - 负面关键词惩罚: 指数级分数减少
        
        3. 分类逻辑:
           - 最低置信度阈值: 0.25
           - 优先类别阈值: 最高分的 65%
           - 一般类别阈值: 最高分的 45%
           - 优先类别的层次化处理
    
    Returns:
        List[Tuple[str, float]]: 按置信度降序排序的 (类别, 置信度分数) 对列表
    """
    # 文本预处理
    title_lower = title.lower()
    abstract_lower = abstract.lower()
    
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
            
            # 完整短语精确匹配（最高权重）
            if keyword_lower in title_lower:
                match_score = weight * 0.25  # 标题中的精确匹配权重最高
                score += match_score
                matches.append(f"标题精确匹配 [{keyword}]: +{match_score:.2f}")
            elif keyword_lower in abstract_lower:
                match_score = weight * 0.15  # 摘要中的精确匹配权重次之
                score += match_score
                matches.append(f"摘要精确匹配 [{keyword}]: +{match_score:.2f}")
            
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
            
            # 部分关键词匹配（较低权重，但考虑匹配比例）
            else:
                # 计算在标题中匹配的单词
                title_matched = keyword_words & title_words
                if title_matched:
                    # 匹配比例影响得分
                    match_ratio = len(title_matched) / len(keyword_words)
                    # 匹配比例越高，得分越高
                    match_score = weight * 0.08 * match_ratio * (1 + match_ratio)
                    score += match_score
                    matches.append(f"标题部分匹配 [{keyword}] ({match_ratio:.1%}): +{match_score:.2f}")
                
                # 计算在摘要中匹配的单词
                abstract_matched = keyword_words & abstract_words
                if abstract_matched and abstract_matched != title_matched:
                    # 匹配比例影响得分
                    match_ratio = len(abstract_matched) / len(keyword_words)
                    # 匹配比例越高，得分越高
                    match_score = weight * 0.05 * match_ratio * (1 + match_ratio)
                    score += match_score
                    matches.append(f"摘要部分匹配 [{keyword}] ({match_ratio:.1%}): +{match_score:.2f}")
            
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
            
            # 使用指数衰减进行惩罚，惩罚力度更大
            if negative_score > 0:
                original_score = score
                score *= math.exp(-negative_score * 1.2)
                penalty = original_score - score
                matches.append(f"负向惩罚总计: -{penalty:.2f}")
        
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
    if max_score < 0.25:  # 提高最低置信度要求
        return []
    
    # 处理高优先级类别
    high_priority_categories = ["基础智能与认知", "生成式建模", "多模态学习", "感知与识别", "医学影像与分析"]
    for category in high_priority_categories:
        if category in scores:
            category_score = scores[category]
            # 对高优先级类别使用较低的相对阈值
            if category_score >= max_score * 0.65 and category_score >= 0.25:
                return [(category, category_score)]
    
    # 处理一般类别
    significant_categories = [
        (category, score) 
        for category, score in scores.items() 
        if score >= max_score * 0.45  # 降低相对阈值，捕获更多相关类别
    ]
    
    # 按置信度降序排序返回结果
    return sorted(significant_categories, key=lambda x: x[1], reverse=True)


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
        category = "其他"
        subcategory = "未指定"
        title_cn = f"[翻译失败] {title}"
        analysis = {}

        # 并行执行耗时任务
        try:
            with ThreadPoolExecutor(max_workers=3) as executor:
                # 提交所有任务
                github_future = executor.submit(extract_github_link, abstract)
                analysis_future = executor.submit(
                    glm_helper.analyze_paper_contribution, title, abstract)
                category_future = executor.submit(
                    glm_helper.categorize_paper, title, abstract)
                title_cn_future = executor.submit(
                    glm_helper.translate_title, title)

                # 等待所有任务完成
                github_link = github_future.result() or "None"
                analysis = analysis_future.result() or {}
                
                # 确保分类结果是有效的类别
                category_result = category_future.result()
                if category_result and category_result in CATEGORY_THRESHOLDS:
                    category = category_result
                else:
                    print(f"警告: 无效的分类结果 '{category_result}'，使用默认分类'其他'")
                    category = "其他"
                    
                title_cn = title_cn_future.result() or f"[翻译失败] {title}"
        except Exception as e:
            print(f"并行处理任务时出错: {str(e)}")
            # 继续处理，使用默认值
        
        # 获取子类别信息
        try:
            subcategory = glm_helper.determine_subcategory(title, abstract, category)
        except Exception as e:
            print(f"获取子类别时出错: {str(e)}")
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
        print(f"处理论文时出错: {str(e)}")
        return None


def get_cv_papers():
    """获取CV领域论文并保存为Markdown"""
    print("\n" + "="*50)
    print(f"开始获取CV论文 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)
    
    try:
        # 获取目标日期（前一天）
        target_date = (datetime.now() - timedelta(days=QUERY_DAYS_AGO)).date()
        print(f"\n📅 目标日期: {target_date}")
        print(f"📊 最大论文数: {MAX_RESULTS}")
        print(f"🧵 最大线程数: {MAX_WORKERS}\n")

        # 初始化ChatGLM助手
        print("🤖 初始化ChatGLM助手...")
        glm_helper = ChatGLMHelper()

        # 初始化arxiv客户端
        print("🔄 初始化arXiv客户端...")
        client = arxiv.Client(
            page_size=100,  # 每页获取100篇论文
            delay_seconds=3,  # 请求间隔3秒
            num_retries=5    # 失败重试5次
        )

        # 构建查询
        search = arxiv.Search(
            query='cat:cs.CV',  # 计算机视觉类别
            max_results=MAX_RESULTS,
            sort_by=arxiv.SortCriterion.LastUpdatedDate,
            sort_order=arxiv.SortOrder.Descending  # 确保按时间降序排序
        )

        # 创建线程池
        total_papers = 0
        papers_by_category = defaultdict(list)

        # 使用线程池并行处理论文
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 创建进度条
            print("\n🔍 开始获取论文...")
            results = client.results(search)
            
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
            print(f"没有找到{target_date}发布的论文。")
            return

        # 打印统计信息
        print(f"\n📊 论文统计信息：")
        print(f"{'='*50}")
        
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
            if len(papers) == 0:
                continue
                
            # 打印一级分类标题
            print(f"\n【{category}】")
            
            # 按子类别分组
            papers_by_subcategory = defaultdict(list)
            for paper in papers:
                subcategory = paper.get('subcategory', '')
                if not subcategory or subcategory == "未指定":
                    subcategory = "其他"
                papers_by_subcategory[subcategory].append(paper)
            
            # 按论文数量降序排序子类别
            sorted_subcategories = sorted(
                papers_by_subcategory.items(),
                key=lambda x: len(x[1]),
                reverse=True
            )
            
            # 打印一级分类总数
            num_new = sum(1 for p in papers if not p['is_updated'])
            num_updated = sum(1 for p in papers if p['is_updated'])
            print(f"总计: {len(papers):3d} 篇 (🆕 {num_new:3d} 新发布, 📝 {num_updated:3d} 更新)")
            
            # 打印子类别统计
            for subcategory, subpapers in sorted_subcategories:
                num_new = sum(1 for p in subpapers if not p['is_updated'])
                num_updated = sum(1 for p in subpapers if p['is_updated'])
                print(
                    f"└─ {subcategory:15s}: {len(subpapers):3d} 篇 (🆕 {num_new:3d} 新发布, 📝 {num_updated:3d} 更新)")
        
        print(f"\n{'='*50}")
        print(f"总计: {total_papers} 篇")
        
        # 保存结果到Markdown文件
        print("\n💾 正在保存结果到Markdown文件...")
        save_papers_to_markdown(papers_by_category, target_date)
        
        print("\n" + "="*50)
        print(f"CV论文获取完成 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*50 + "\n")

    except Exception as e:
        print("\n❌ 处理CV论文时出错:")
        print(f"错误信息: {str(e)}")
        print(f"发生时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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

    print(f"\n表格格式文件已保存到: {table_filepath}")
    print(f"详细格式文件已保存到: {detailed_filepath}")


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
        
        # 按子类别分组
        papers_by_subcategory = defaultdict(list)
        for paper in papers:
            subcategory = paper.get('subcategory', '')
            if not subcategory or subcategory == "未指定":
                subcategory = "其他"
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
    # 直接运行查询
    get_cv_papers()
