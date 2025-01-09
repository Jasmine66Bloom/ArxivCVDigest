from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import io
import os
import re
import time
from typing import Any, Dict

import arxiv
import requests
from tqdm import tqdm

from chatglm_helper import ChatGLMHelper

# 查询参数设置
QUERY_DAYS_AGO = 1          # 查询几天前的论文，0=今天，1=昨天，2=前天
MAX_RESULTS = 200           # 最大返回论文数量
MAX_WORKERS = 5            # 并行处理的最大线程数


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
    """生成表格形式的Markdown内容"""
    markdown = ""

    # 过滤掉没有论文的类别
    active_categories = {k: v for k, v in papers_by_category.items() if v}

    if not active_categories:
        return "今天没有相关论文。"

    # 表格列标题
    headers = ['状态', '英文标题', '中文标题', '作者', 'PDF链接', '代码链接']

    # 为每个有论文的类别创建表格
    for category, papers in sorted(((k, v) for k, v in active_categories.items() if k != "其他")):
        # 添加类别标题
        markdown += f"\n## {category}\n\n"

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

            # 准备每个字段的值
            values = [
                status,
                paper['title'],
                paper['title_cn'],
                paper['authors'],  # 已经是格式化好的字符串
                f"<{paper['pdf_url']}>",
                f"<{paper['github_link']}>" if paper['github_link'] != 'None' else 'None'
            ]

            # 处理特殊字符
            values = [str(v).replace('\n', ' ').replace('|', '&#124;')
                      for v in values]

            # 添加到表格
            markdown += "|" + "|".join(values) + "|\n"

        # 在每个表格后添加空行
        markdown += "\n"

    # 处理"其他"类别
    if "其他" in active_categories:
        # 添加类别标题
        markdown += f"\n## 其他\n\n"

        # 创建表格头
        markdown += "|" + "|".join(headers) + "|\n"
        markdown += "|" + "|".join(["---"] * len(headers)) + "|\n"

        # 添加论文
        for paper in active_categories["其他"]:
            # 确定论文状态
            if paper['is_updated']:
                status = f"📝 更新"
            else:
                status = f"🆕 发布"

            # 准备每个字段的值
            values = [
                status,
                paper['title'],
                paper['title_cn'],
                paper['authors'],  # 已经是格式化好的字符串
                f"<{paper['pdf_url']}>",
                f"<{paper['github_link']}>" if paper['github_link'] != 'None' else 'None'
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
    """生成详细格式的Markdown内容"""
    markdown = ""

    # 过滤掉没有论文的类别
    active_categories = {k: v for k, v in papers_by_category.items() if v}

    if not active_categories:
        return "今天没有相关论文。"

    # 为每个有论文的类别创建详细内容
    for category, papers in sorted(((k, v) for k, v in active_categories.items() if k != "其他")):
        # 添加类别标题
        markdown += f"## **{category}**\n"
        markdown += "\n"

        # 添加论文
        for idx, paper in enumerate(papers, 1):
            markdown += f'**index:** {idx}<br />\n'
            markdown += f'**Date:** {target_date.strftime("%Y-%m-%d")}<br />\n'
            markdown += f'**Title:** {paper["title"]}<br />\n'
            markdown += f'**Title_cn:** {paper["title_cn"]}<br />\n'
            # 已经是格式化好的字符串
            markdown += f'**Authors:** {paper["authors"]}<br />\n'
            markdown += f'**PDF:** <{paper["pdf_url"]}><br />\n'

            if paper["github_link"] != 'None':
                markdown += f'**Code:** <{paper["github_link"]}><br />\n'
            else:
                markdown += '**Code:** None<br />\n'

            if "核心贡献" in paper:
                markdown += f'**Core Contribution:** {paper["核心贡献"]}<br />\n'
            if "核心问题" in paper:
                markdown += f'**Core Problem:** {paper["核心问题"]}<br />\n'

            markdown += "\n"

        markdown += "\n"

    # 处理"其他"类别
    if "其他" in active_categories:
        markdown += f"## **其他**\n"
        markdown += "\n"

        for idx, paper in enumerate(active_categories["其他"], 1):
            markdown += f'**index:** {idx}<br />\n'
            markdown += f'**Date:** {target_date.strftime("%Y-%m-%d")}<br />\n'
            markdown += f'**Title:** {paper["title"]}<br />\n'
            markdown += f'**Title_cn:** {paper["title_cn"]}<br />\n'
            # 已经是格式化好的字符串
            markdown += f'**Authors:** {paper["authors"]}<br />\n'
            markdown += f'**PDF:** <{paper["pdf_url"]}><br />\n'

            if paper["github_link"] != 'None':
                markdown += f'**Code:** <{paper["github_link"]}><br />\n'
            else:
                markdown += '**Code:** None<br />\n'

            if "核心贡献" in paper:
                markdown += f'**Core Contribution:** {paper["核心贡献"]}<br />\n'
            if "核心问题" in paper:
                markdown += f'**Core Problem:** {paper["核心问题"]}<br />\n'

            markdown += "\n"

    return markdown


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
        f.write(df_to_markdown_table(papers_by_category, target_date))

    # 保存详细格式的markdown文件到local/年-月目录
    with open(detailed_filepath, 'w', encoding='utf-8') as f:
        f.write(title)
        f.write(df_to_markdown_detailed(papers_by_category, target_date))

    print(f"\n表格格式文件已保存到: {table_filepath}")
    print(f"详细格式文件已保存到: {detailed_filepath}")


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

        # 并行执行耗时任务
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
            analysis = analysis_future.result()
            category = category_future.result()
            title_cn = title_cn_future.result()

        return {
            'title': title,
            'title_cn': title_cn,
            'authors': authors_str,
            'abstract': abstract,
            'paper_url': paper_url,
            'pdf_url': pdf_url,
            'github_link': github_link,
            'analysis_result': analysis,
            'category': category,
            'published': published,
            'updated': updated,
            'is_updated': updated_date == target_date and published_date != target_date
        }

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
            # 分批获取和处理论文
            batch_size = 30  # 每批处理30篇论文
            futures = []
            processed_papers = set()  # 用于跟踪已处理的论文ID

            # 创建总进度条
            total_pbar = tqdm(total=MAX_RESULTS, desc="总进度", unit="篇")
            # 创建批处理进度条
            batch_pbar = tqdm(total=batch_size, desc="当前批次",
                              unit="篇", leave=False)

            for paper in client.results(search):
                paper_id = paper.get_short_id()

                # 跳过已处理的论文
                if paper_id in processed_papers:
                    continue
                processed_papers.add(paper_id)

                # 提交处理任务
                future = executor.submit(
                    process_paper, paper, glm_helper, target_date)
                futures.append(future)

                # 当累积了一批论文时，等待它们处理完成
                if len(futures) >= batch_size:
                    batch_pbar.reset()  # 重置批处理进度条
                    for completed_future in as_completed(futures):
                        paper_info = completed_future.result()
                        if paper_info:  # 如果论文符合日期要求
                            total_papers += 1
                            category = paper_info['category']
                            papers_by_category[category].append(paper_info)
                            total_pbar.update(1)  # 更新总进度
                        batch_pbar.update(1)  # 更新批处理进度
                    futures = []  # 清空已处理的futures

                # 如果已经处理了足够多的论文，就停止
                if total_papers >= MAX_RESULTS:
                    break

            # 处理剩余的论文
            if futures:
                batch_pbar.reset()  # 重置批处理进度条
                batch_pbar.total = len(futures)  # 设置正确的总数
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

        # 按类别对论文进行排序（按发布时间降序）
        for category in papers_by_category:
            papers_by_category[category].sort(
                key=lambda x: x.get('published', ''),
                reverse=True
            )

        # 保存结果
        print("\n💾 正在保存结果到Markdown文件...")
        save_papers_to_markdown(papers_by_category, target_date)

        # 打印统计信息
        print(f"\n📊 论文统计信息：")
        print(f"{'='*30}")
        # 按论文数量降序排序类别
        sorted_categories = sorted(
            papers_by_category.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )
        for category, papers in sorted_categories:
            num_new = sum(1 for p in papers if not p['is_updated'])
            num_updated = sum(1 for p in papers if p['is_updated'])
            print(
                f"{category:15s}: {len(papers):3d} 篇 (🆕 {num_new:3d} 新发布, 📝 {num_updated:3d} 更新)")
        print(f"{'='*30}")
        print(f"总计: {total_papers} 篇")
        
        print("\n" + "="*50)
        print(f"CV论文获取完成 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*50 + "\n")

    except Exception as e:
        print("\n❌ 处理CV论文时出错:")
        print(f"错误信息: {str(e)}")
        print(f"发生时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        raise  # 抛出异常以便查看详细错误信息


if __name__ == "__main__":
    # 直接运行查询
    get_cv_papers()
