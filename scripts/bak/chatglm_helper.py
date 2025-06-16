#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ChatGLM助手模块 - 与ChatGLM模型交互接口

封装ChatGLM API调用，提供论文分类、翻译和内容分析功能。

作者: Jasmine Bloom
日期: 2025-06
"""

import os
import re
import json
import time
import requests
import random
import traceback
import math
from typing import Dict, List, Tuple, Optional, Any, Union
from zhipuai import ZhipuAI
from collections import defaultdict

# 导入日志配置
from logger_config import setup_logger
import categories_config
from config import CHATGLM_API_KEY  # 导入配置文件

# 初始化日志记录器
logger = setup_logger(name='chatglm_helper', level='info', simple_format=True)

class ChatGLMHelper:
    def __init__(self):
        """初始化ChatGLM客户端"""
        if not CHATGLM_API_KEY:
            logger.error("缺少API密钥，请在config.py中设置CHATGLM_API_KEY")
            raise ValueError("缺少API密钥")
            
        # 初始化客户端
        self.client = ZhipuAI(api_key=CHATGLM_API_KEY)
        logger.info("初始化完成")

    def translate_title(self, title: str, abstract: str = "") -> str:
        """
        使用ChatGLM翻译论文标题，增强的提示词和错误处理
        Args:
            title: 论文英文标题
            abstract: 论文摘要，用于提供上下文（可选）
        Returns:
            str: 中文标题
        """
        max_retries = 3
        retry_delay = 2  # 重试延迟秒数
        
        # 提取摘要中的关键句子作为上下文
        context = ""
        if abstract and len(abstract) > 50:
            # 取摘要的前150个字符作为上下文
            context = f"""论文摘要开头：
{abstract[:150]}..."""

        # 改进的提示词，更结构化和清晰，强调不要翻译专业术语
        prompt = f"""任务：将计算机视觉领域的学术论文标题从英文翻译成中文。

论文标题：
{title}

{context}

翻译要求：
1. 不要翻译专业用语、算法名称、模型名称、缩写和专有名词，必须原样保留
2. 以下类型的术语必须原样保留：
   - 模型名称：如 CLIP, ViT, BERT, GPT, ResNet, Transformer 等
   - 算法名称：如 RANSAC, SLAM, NeRF, GAN, VAE 等
   - 缩写：如 CNN, RNN, LSTM, 3D, RGB, IoU, mAP, FPS 等
   - 数据集名称：如 COCO, ImageNet, CIFAR 等
3. 翻译风格要简洁专业、符合中文学术表达习惯
4. 只返回中文标题，不要包含其他解释或说明

输出格式：直接返回中文标题，不要添加任何前缀或其他文字"""

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="glm-4-air",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=100,
                    top_p=0.7,
                )
                translation = response.choices[0].message.content.strip()
                # 确保返回的是中文
                if any('一' <= char <= '鿿' for char in translation):
                    return translation
                else:
                    logger.warning(f"警告：第{attempt + 1}次翻译未返回中文结果，重试中...")
            except Exception as e:
                logger.error(f"翻译出错 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                logger.debug(traceback.format_exc())
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                continue
        
        logger.error(f"翻译失败，返回原标题")
        return title

    def clean_json_string(self, text: str) -> str:
        """清理并提取JSON字符串"""
        # 移除可能的markdown代码块标记
        if '```' in text:
            # 提取代码块内容
            pattern = r'```(?:json)?(.*?)```'
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                text = matches[0]
        
        # 移除所有空行和前后空白
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = ''.join(lines)
        
        # 尝试找到JSON对象的开始和结束
        start = text.find('{')
        end = text.rfind('}')
        
        if start != -1 and end != -1:
            text = text[start:end+1]
        else:
            return "{}"  # 如果没有找到有效的JSON对象，返回空对象
        
        # 替换单引号为双引号
        text = text.replace("'", '"')
        
        # 处理中文引号
        text = text.replace(""", '"').replace(""", '"')
        
        return text

    def get_category_by_keywords(self, title: str, abstract: str) -> List[Tuple[str, float]]:
        """通过关键词匹配进行分类
        
        Args:
            title: 论文标题
            abstract: 论文摘要
            
        Returns:
            List[Tuple[str, float]]: [(类别名称, 置信度), ...] 按置信度排序的类别列表
        """
        title_lower = title.lower()
        abstract_lower = abstract.lower()
        
        # 预处理：移除常见的无意义词
        common_words = {'a', 'an', 'the', 'in', 'on', 'at', 'for', 'to', 'of', 'and', 'or', 'with', 'by'}
        title_words = set(title_lower.split()) - common_words
        abstract_words = set(abstract_lower.split()) - common_words
        
        category_scores = defaultdict(float)
        category_matches = defaultdict(list)  # 记录每个类别匹配到的关键词
        
        # 第一阶段：计算每个类别的初始得分
        for category, config in categories_config.CATEGORY_KEYWORDS.items():
            keywords = config["keywords"]
            negative_keywords = config["negative_keywords"]
            score = 0.0
            negative_score = 0.0
            matched_keywords = []
            
            # 计算正向关键词得分
            for keyword, weight in keywords:
                keyword_lower = keyword.lower()
                keyword_words = set(keyword_lower.split()) - common_words
                
                # 完整短语匹配（权重最高）
                if keyword_lower in title_lower:
                    score += weight * 3.0  # 标题中完整短语匹配权重最高
                    matched_keywords.append(f"标题完整匹配: {keyword}")
                elif keyword_lower in abstract_lower:
                    score += weight * 1.5  # 摘要中完整短语匹配权重其次
                    matched_keywords.append(f"摘要完整匹配: {keyword}")
                
                # 关键词组合匹配（权重适中）
                elif keyword_words.issubset(title_words):
                    score += weight * 2.0  # 标题中词组匹配
                    matched_keywords.append(f"标题词组匹配: {keyword}")
                elif keyword_words.issubset(abstract_words):
                    score += weight * 1.0  # 摘要中词组匹配
                    matched_keywords.append(f"摘要词组匹配: {keyword}")
                
                # 部分关键词匹配（权重较低）
                else:
                    matched_words = keyword_words & (title_words | abstract_words)
                    if matched_words:
                        partial_score = len(matched_words) / len(keyword_words) * weight * 0.5
                        score += partial_score
                        matched_keywords.append(f"部分匹配: {keyword} ({', '.join(matched_words)})")
            
            # 检查负向关键词（更严格的惩罚机制）
            for neg_keyword, neg_weight in negative_keywords:
                neg_keyword_lower = neg_keyword.lower()
                neg_keyword_words = set(neg_keyword_lower.split()) - common_words
                
                # 完整短语匹配（严重惩罚）
                if neg_keyword_lower in title_lower:
                    negative_score += neg_weight * 1.0  # 标题中的负向关键词严重惩罚
                    matched_keywords.append(f"负向完整匹配(标题): {neg_keyword}")
                elif neg_keyword_lower in abstract_lower:
                    negative_score += neg_weight * 0.7  # 摘要中的负向关键词严重惩罚
                    matched_keywords.append(f"负向完整匹配(摘要): {neg_keyword}")
                
                # 关键词组合匹配（中度惩罚）
                elif neg_keyword_words.issubset(title_words):
                    negative_score += neg_weight * 0.8
                    matched_keywords.append(f"负向词组匹配(标题): {neg_keyword}")
                elif neg_keyword_words.issubset(abstract_words):
                    negative_score += neg_weight * 0.5
                    matched_keywords.append(f"负向词组匹配(摘要): {neg_keyword}")
            
            # 根据负向得分调整最终得分（更严格的惩罚机制）
            if negative_score > 0:
                # 使用指数衰减进行惩罚
                score *= math.exp(-negative_score)
            
            # 记录匹配信息
            if score > 0:
                category_scores[category] = score
                category_matches[category] = matched_keywords
        
        # 第二阶段：相对得分分析和类别选择
        if category_scores:
            max_score = max(category_scores.values())
            # 使用"其他"类别的阈值作为基准
            other_config = categories_config.CATEGORY_THRESHOLDS.get("其他", {})
            other_threshold = other_config.get("threshold", 1.8) if isinstance(other_config, dict) else 1.8
            
            # 筛选得分显著的类别（得分至少为最高分的70%）
            significant_categories = []
            for category, score in category_scores.items():
                # 获取类别的阈值
                category_config = categories_config.CATEGORY_THRESHOLDS.get(category, {})
                category_threshold = category_config.get("threshold", 1.0) if isinstance(category_config, dict) else 1.0
                relative_threshold = max_score * 0.7  # 相对阈值：最高分的70%
                
                # 同时满足绝对阈值和相对阈值
                if score > category_threshold and score > relative_threshold:
                    significant_categories.append((category, score))
            
            # 如果有显著类别，按分数排序返回
            if significant_categories:
                return sorted(significant_categories, key=lambda x: x[1], reverse=True)
            
            # 如果没有显著类别，但有类别得分超过其他类别阈值的一半，选择得分最高的
            elif max_score > other_threshold / 2:
                return [(max(category_scores.items(), key=lambda x: x[1])[0], max_score)]
        
        logger.error("无法确定类别，返回'其他'")
        return [("其他", 0.0)]

    def get_category_by_semantic(self, title: str, abstract: str) -> List[Tuple[str, float]]:
        """使用语义分析进行分类"""
        categories_str = "\n".join(f"- {cat}" for cat in categories_config.CATEGORY_DISPLAY_ORDER)
        
        prompt = f"""分析这篇计算机视觉论文属于哪个类别。

标题：{title}
摘要：{abstract}

可选类别：
{categories_str}

请分析：
1. 论文的主要技术路线和方法
2. 论文的核心创新点
3. 论文的应用场景
4. 与各个类别的相关度（0-1分）

请用JSON格式返回分类结果，格式如下：
{{
    "analysis": {{
        "main_method": "主要技术路线",
        "innovation": "核心创新点",
        "application": "应用场景"
    }},
    "categories": [
        {{"name": "类别名", "score": 相关度, "reason": "原因"}}
    ]
}}"""

        try:
            response = self.client.chat.completions.create(
                model="glm-4-air",  # 使用功能更强的模型
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000,
                top_p=0.7,
            )
            
            result = response.choices[0].message.content
            json_str = self.clean_json_string(result)
            
            try:
                data = json.loads(json_str)
                categories = [(cat["name"], cat["score"]) for cat in data.get("categories", [])]
                return sorted(categories, key=lambda x: x[1], reverse=True)
            except json.JSONDecodeError:
                logger.error(f"JSON解析失败: {json_str}")
                return [("其他", 0.0)]
                
        except Exception as e:
            logger.error(f"语义分析出错: {str(e)}")
            return [("其他", 0.0)]

    def combine_results(self, keyword_results: List[Tuple[str, float]], 
                       semantic_results: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        Synthesize classification results from keyword matching and semantic analysis.
        
        Implementation Details:
            1. Scoring Integration:
               - Keyword matching: 30% weight (precision-oriented)
               - Semantic analysis: 70% weight (context-oriented)
            
            2. Confidence Thresholds:
               - Minimum confidence: 0.3
               - Priority category threshold: 60% of max confidence
               - General category threshold: 40% of max confidence
            
            3. Priority Processing:
               - Hierarchical evaluation of high-priority categories
               - Early return for high-confidence priority matches
               - Fallback to general category evaluation
        
        Args:
            keyword_results: List of (category, confidence) pairs from lexical analysis
            semantic_results: List of (category, confidence) pairs from semantic analysis
        
        Returns:
            List[Tuple[str, float]]: Synthesized classification results,
                                    sorted by confidence in descending order
        """
        if not keyword_results and not semantic_results:
            return []

        # Initialize combined confidence scores
        combined_scores = defaultdict(float)
        
        # Integrate keyword matching results (precision-oriented)
        for category, score in keyword_results:
            combined_scores[category] += score * 0.3
        
        # Integrate semantic analysis results (context-oriented)
        for category, score in semantic_results:
            combined_scores[category] += score * 0.7
        
        # Validate against minimum confidence threshold
        max_score = max(combined_scores.values()) if combined_scores else 0
        if max_score < 0.3:
            return []
        
        # Process high-priority categories
        high_priority_categories = ["基础智能与认知", "生成式建模", "多模态学习", "感知与识别", "医学影像与分析"]
        for category in high_priority_categories:
            if category in combined_scores:
                score = combined_scores[category]
                # Early return for high-confidence priority matches
                if score >= max_score * 0.6:
                    return [(category, score)]
        
        # Process general categories
        significant_results = [
            (category, score)
            for category, score in combined_scores.items()
            if score >= max_score * 0.4  # Significance threshold
        ]
        
        return sorted(significant_results, key=lambda x: x[1], reverse=True)

    def classify_paper(self, title: str, abstract: str) -> Tuple[str, float]:
        """改进的论文分类方法"""
        # 1. 关键词匹配（第一层）
        keyword_results = self.get_category_by_keywords(title, abstract)
        
        # 如果关键词匹配非常明确，直接返回
        if keyword_results and keyword_results[0][1] >= 2.5:
            return keyword_results[0]
        
        # 2. 语义分析（第二层）
        semantic_results = self.get_category_by_semantic(title, abstract)
        
        # 3. 综合分析（第三层）
        combined_scores = self.combine_results(keyword_results, semantic_results)
        
        # 4. 二次确认（第四层，针对不确定的情况）
        best_category, confidence = combined_scores[0]
        if confidence < 1.8:  # 如果置信度较低
            return self.confirm_category(title, abstract, best_category)
        
        return combined_scores[0]

    def categorize_paper(self, title: str, abstract: str) -> str:
        """使用ChatGLM对论文进行分类
        
        Args:
            title: 论文标题
            abstract: 论文摘要
        
        Returns:
            str: 论文类别
        """
        # 初始化默认值，避免异常时未定义
        keyword_category = "其他"
        confidence = 0.0
        
        try:
            # 先尝试使用关键词匹配
            keyword_matches = self.get_category_by_keywords(title, abstract)
            if keyword_matches:
                keyword_category, confidence = keyword_matches[0]
                
                # 如果关键词匹配非常确定（高置信度），直接返回结果
                if keyword_category != "其他" and confidence >= 3.0:
                    logger.info(f"关键词高置信度匹配: {keyword_category} (置信度: {confidence:.2f})")
                    return keyword_category
            
            # 获取前三个关键词匹配结果作为参考
            top_matches = keyword_matches[:3] if keyword_matches else []
            top_categories = [f"{cat} (置信度: {conf:.2f})" for cat, conf in top_matches if cat != "其他"]
            top_categories_str = "、".join(top_categories) if top_categories else "无明确匹配"
            
            # 提取论文的核心信息，用于更好的分类
            core_info = self.extract_paper_core_info(title, abstract)
            
            # 构建增强的分类提示词，提供更多上下文和指导
            prompt = f"""请作为一位专业的计算机视觉研究专家，对以下论文进行精确分类。

论文标题: {title}

论文摘要: {abstract}

论文核心信息:
- 主要研究方向: {core_info.get('research_direction', '未提取')}
- 核心技术: {core_info.get('core_technology', '未提取')}
- 主要贡献: {core_info.get('main_contribution', '未提取')}
- 应用领域: {core_info.get('application_area', '未提取')}

关键词匹配结果: {top_categories_str}

{categories_config.CATEGORY_PROMPT}

分类指南:
1. 仔细分析论文的核心技术和主要贡献，不要被表面的应用领域误导
2. 优先考虑论文的技术本质和创新点，而非应用场景
3. 如果论文跨多个领域，请选择最能体现其核心创新的类别
4. 如果关键词匹配结果有参考价值，可以考虑这些建议，但你应该做出独立判断
5. 只有在完全无法确定时才返回"其他"类别

请直接返回最合适的类别名称，不要有任何解释或额外文本。"""
            
            # 调用 ChatGLM 进行分类
            response = self.client.chat.completions.create(
                model="glm-4-air",  # 修改为 flashx 版本
                messages=[{"role": "user", "content": prompt}],
                temperature=0.05,
                max_tokens=50,
                top_p=0.5,
            )
            
            # 获取分类结果
            category = response.choices[0].message.content.strip()
            
            # 验证返回的类别是否在预定义类别中
            if category in categories_config.CATEGORY_DISPLAY_ORDER:
                logger.info(f"ChatGLM 分类结果: {category}")
                return category
            
            # 尝试模糊匹配 - 检查返回的类别是否包含在任何预定义类别中
            for cat in categories_config.CATEGORY_DISPLAY_ORDER:
                if cat.lower() in category.lower() or category.lower() in cat.lower():
                    logger.info(f"从回复中提取到有效类别: {cat}")
                    return cat
            
            # 如果不在预定义类别中，尝试关键词匹配的结果（如果有较高置信度）
            if keyword_category != "其他" and confidence >= 1.5:
                logger.info(f"使用关键词匹配结果: {keyword_category} (置信度: {confidence:.2f})")
                return keyword_category
            
            # 最后尝试使用语义分类
            semantic_results = self.get_category_by_semantic(title, abstract)
            if semantic_results:
                semantic_category, semantic_confidence = semantic_results[0]
                if semantic_category != "其他":
                    logger.info(f"使用语义分类结果: {semantic_category} (置信度: {semantic_confidence:.2f})")
                    return semantic_category
            
            logger.warning("无法确定类别，返回'其他'")
            return "其他"
        except Exception as e:
            logger.error(f"分类过程出错: {str(e)}")
            logger.debug(traceback.format_exc())
            return "其他"

    def analyze_paper_contribution(self, title: str, abstract: str) -> dict:
        """Analyze the paper's core contribution and return a summary in one sentence
    
        Args:
            title: Paper title
            abstract: Paper abstract
    
        Returns:
            dict: Dictionary containing the analysis result with a single key "核心贡献" (core contribution)
        """
        prompt = f"""Please summarize the core contribution of the following computer vision paper in one sentence.
    
Paper title: {title}
Paper abstract: {abstract}

Format requirements:
- Must be a complete sentence, no more than 50 characters
- Include the problem solved and the method proposed
- Use simple language, avoiding technical terms
- Do not use numbering or bullet points
- Do not use quotes or other punctuation to enclose the content

Example output:
Proposed a lightweight image segmentation method based on attention mechanism, significantly improving boundary clarity and processing speed.
"""
        try:
            response = self.client.chat.completions.create(
                model="glm-4-air",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200,
                top_p=0.7,
            )
            
            # Get the summary sentence
            contribution_summary = response.choices[0].message.content.strip()
            
            # Remove any extra content, keeping only the first sentence
            if '\n' in contribution_summary:
                contribution_summary = contribution_summary.split('\n')[0].strip()
                
            # Ensure the length is appropriate
            if len(contribution_summary) > 100:
                contribution_summary = contribution_summary[:97] + '...'
                
            return {
                "核心贡献": contribution_summary
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze paper contribution: {str(e)}")
            return {
                "核心贡献": "分析失败"
            }

    def translate_title(self, title: str) -> str:
        """Translate the paper title to Chinese
    
        Args:
            title: English title
    
        Returns:
            str: Chinese title
        """
        prompt = f"""Please translate the title of the following computer vision paper into Chinese, maintaining professionalism and accuracy:

{title}

Only return the translated title, without any explanation."""
        try:
            response = self.client.chat.completions.create(
                model="glm-4-air",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200,
                top_p=0.7,
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Failed to translate title: {str(e)}")
            return title  # If translation fails, return the original title

    def confirm_category(self, title: str, abstract: str, initial_category: str) -> Tuple[str, float]:
        """Confirm the classification result"""
        prompt = f"""Confirm whether the paper belongs to the category "{initial_category}".
    
Title: {title}
Abstract: {abstract}

Analysis points:
1. Key technologies in the title
2. The paper's main objective
3. Whether there is a more suitable category

Please strictly follow the JSON format below, without any additional content:
{
    "category": "Final category name",
    "confidence": Score
}"""

        try:
            response = self.client.chat.completions.create(
                model="glm-4-air",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100,
                top_p=0.7,
            )
            result = response.choices[0].message.content.strip()
            
            # Clean and parse JSON
            try:
                # First try to parse directly
                result_json = json.loads(result)
            except json.JSONDecodeError:
                # If failed, try cleaning and parsing again
                cleaned_result = self.clean_json_string(result)
                try:
                    result_json = json.loads(cleaned_result)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing failed: {str(e)}")
                    logger.debug(traceback.format_exc())
                    return (initial_category, 1.5)
        
            category = result_json.get("category", initial_category)
            # Ensure confidence is a float
            try:
                confidence = float(result_json.get("confidence", 0))
            except (TypeError, ValueError):
                confidence = 1.5
        
            return (category, confidence)
        except Exception as e:
            logger.error(f"Classification confirmation failed: {str(e)}")
            return (initial_category, 1.5)

    def decide_category(self, title: str, abstract: str, candidate_categories: List[Tuple], match_details: Dict = None) -> str:
        """Use ChatGLM to decide the final category from candidate categories
    
        Args:
            title: Paper title
            abstract: Paper abstract
            candidate_categories: List of candidate categories, each element is a tuple of (category, score, subcategory)
            match_details: Dictionary of keyword matching details, optional
    
        Returns:
            str: Final decided category name
        """
        # Initialize match details dictionary if not provided
        if match_details is None:
            match_details = {}
        try:
            import categories_config
            
            # Format candidate category information
            candidates_info = ""
            for i, (category, score, subcategory) in enumerate(candidate_categories[:3], 1):
                subcategory_info = f", 子类别: {subcategory[0] if subcategory else '无'}" if subcategory else ""
                candidates_info += f"\n{i}. {category} (得分: {score:.2f}{subcategory_info})"
            
            # Extract key information from the abstract
            key_info = ""
            if len(abstract) > 100:
                # Extract the beginning and end of the abstract, which usually contains the most important information
                intro = abstract[:200] if len(abstract) > 200 else abstract
                conclusion = abstract[-200:] if len(abstract) > 400 else ""
                key_info = f"摘要开头: {intro}\n"
                if conclusion:
                    key_info += f"摘要结尾: {conclusion}\n"
            
            # Format candidate category information with more context
            detailed_candidates = ""
            for i, (category, score, subcategory) in enumerate(candidate_categories[:5], 1):
                subcategory_info = f", 子类别: {subcategory[0] if subcategory else '无'}" if subcategory else ""
                matches = match_details.get(category, [])[:3]
                match_info = f", 关键匹配: {', '.join(matches)}" if matches else ""
                detailed_candidates += f"\n{i}. {category} (得分: {score:.2f}{subcategory_info}{match_info})"
            
            # Construct an enhanced prompt with more context and guidance
            prompt = f"""作为计算机视觉领域的资深研究专家，请对以下论文进行准确分类。

论文标题: {title}

{key_info}
候选类别:{detailed_candidates}

{categories_config.CATEGORY_PROMPT}

**重要分类原则**：
1. **优先考虑基础研究类别**：如果论文提出了新的算法、模型、架构或理论方法，应归类为相应的基础研究类别
2. **应用类别的严格标准**：只有当论文主要是将现有技术应用到特定领域，且缺乏显著的技术创新时，才归类为应用类别
3. **技术创新优先**：即使论文在特定应用领域（如医学、自动驾驶），但如果提出了新的技术方法，应归类为对应的基础研究类别
4. **跨领域论文**：选择最能体现其核心技术贡献的类别

**具体判断标准**：
- 提出新的神经网络架构、训练方法 → 视觉表征学习与大模型
- 改进检测、分割、识别算法 → 静态图像理解与语义解析  
- 新的生成模型、扩散方法、GAN、VAE等 → 生成式视觉与内容创建
- 3D重建、几何理解、点云处理、深度估计 → 三维重建与几何感知
- 视频理解、时序建模、动作识别 → 动态视觉与时序建模
- 多模态融合、跨模态学习、视觉-语言模型 → 多模态视觉与跨模态学习
- 医学图像分析、放射学、病理学、医学影像处理 → 生物医学影像计算
- 自动驾驶、机器人视觉、SLAM → 感知-动作智能与主动视觉

**特别注意**：
- 包含"medical", "radiology", "pathology", "clinical", "diagnosis"等关键词的论文应优先考虑"生物医学影像计算"
- 包含"generation", "diffusion", "GAN", "synthesis", "create"等关键词的论文应优先考虑"生成式视觉与内容创建"  
- 包含"3D", "point cloud", "depth", "geometry", "reconstruction"等关键词的论文应优先考虑"三维重建与几何感知"
- 不要过度将论文归类到"静态图像理解与语义解析"，该类别主要针对传统的2D图像分析任务

请仅返回最合适的类别名称，不要添加任何解释或其他文字。"""
            
            # 调用 ChatGLM 进行分类
            response = self.client.chat.completions.create(
                model="glm-4-air",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.01,
                max_tokens=50,
                top_p=0.3,
            )
            
            # Get the classification result
            category = response.choices[0].message.content.strip()
            
            # Verify if the returned category is in the candidate categories
            candidate_names = [c[0] for c in candidate_categories]
            
            # If the returned category is in the candidate categories, return directly
            if category in candidate_names:
                return category
            
            # If the returned category is not in the candidate categories but in the predefined categories, return
            if category in categories_config.CATEGORY_DISPLAY_ORDER:
                return category
            
            # If not matched, return the category with the highest score
            return candidate_categories[0][0]
            
        except Exception as e:
            logger.error(f"ChatGLM classification decision failed: {str(e)}")
            # If an error occurs, return the category with the highest score
            return candidate_categories[0][0] if candidate_categories else "其他"

    def determine_subcategory(self, title: str, abstract: str, main_category: str) -> str:
        """Determine the paper's subcategory
    
        Args:
            title: Paper title
            abstract: Paper abstract
            main_category: Main category
    
        Returns:
            str: Subcategory name
        """
        try:
            # 如果主类别是"其他"或者主类别没有子类别，直接返回"未指定"
            if main_category == "其他" or main_category not in categories_config.CATEGORY_THRESHOLDS:
                return "未指定"
                
            # 获取主类别的子类别配置
            subcategories = categories_config.CATEGORY_THRESHOLDS[main_category].get("subcategories", {})
            if not subcategories:
                return "未指定"
            
            # 提取论文的核心信息，用于更好的分类
            core_info = self.extract_paper_core_info(title, abstract)
                
            # 构建增强的子类别提示词
            subcategory_list = ", ".join(subcategories.keys())
            subcategory_descriptions = "\n".join([f"- {sub}: {sub}相关技术和方法" for sub in subcategories.keys()])
            
            prompt = f"""请作为计算机视觉专家，将以下论文分类到最合适的子类别中。

论文标题：{title}

论文摘要：{abstract}

论文核心信息:
- 主要研究方向: {core_info.get('research_direction', '未提取')}
- 核心技术: {core_info.get('core_technology', '未提取')}
- 主要贡献: {core_info.get('main_contribution', '未提取')}
- 应用领域: {core_info.get('application_area', '未提取')}

主类别：{main_category}

可选的子类别：
{subcategory_descriptions}

分类指南:
1. 仔细分析论文的核心技术和主要贡献，确定最匹配的子类别
2. 考虑论文的技术路线、创新点和应用场景
3. 如果论文跨多个子类别，选择最能体现其核心创新的子类别
4. 只有在完全无法确定时才返回"未指定"

请直接返回最合适的子类别名称，不要有任何解释或额外文本。如果无法确定，请返回"未指定"。"""
            
            # 调用 ChatGLM 进行子类别分类
            response = self.client.chat.completions.create(
                model="glm-4-air",  # 修改为 flashx 版本
                messages=[{"role": "user", "content": prompt}],
                temperature=0.05,
                max_tokens=50,
                top_p=0.5,
            )
            
            # 获取分类结果
            subcategory = response.choices[0].message.content.strip()
            
            # 验证返回的子类别是否在预定义子类别中
            if subcategory in subcategories:
                logger.info(f"子类别分类结果: {subcategory}")
                return subcategory
            
            # 尝试模糊匹配 - 检查返回的子类别是否包含在任何预定义子类别中
            for sub in subcategories.keys():
                if sub.lower() in subcategory.lower() or subcategory.lower() in sub.lower():
                    logger.info(f"子类别模糊匹配: 从 '{subcategory}' 匹配到 '{sub}'")
                    return sub
            
            # 如果不在预定义子类别中，尝试关键词匹配
            logger.info(f"子类别 '{subcategory}' 不在预定义子类别中，尝试关键词匹配...")
            text = (title + " " + abstract).lower()
            best_match = ""
            best_score = 0
            
            # 使用更复杂的匹配算法
            for sub in subcategories.keys():
                # 计算匹配分数
                sub_lower = sub.lower()
                sub_words = set(sub_lower.split())
                
                # 基础分数：完整匹配
                score = 0
                if sub_lower in text:
                    score += 10
                
                # 单词匹配分数
                text_words = set(text.split())
                matched_words = sub_words & text_words
                if matched_words:
                    word_score = len(matched_words) / len(sub_words) * 5
                    score += word_score
                
                # 部分匹配分数
                for word in sub_words:
                    if len(word) > 3 and word in text:  # 只考虑长度大于3的单词
                        score += 2
                
                if score > best_score:
                    best_score = score
                    best_match = sub
            
            if best_match and best_score >= 2:  # 设置最低匹配阈值
                logger.info(f"子类别关键词匹配: {best_match} (分数: {best_score:.2f})")
                return best_match
            
            # 如果关键词匹配也失败，尝试第二次 ChatGLM 分类，使用更简单的提示词
            simple_prompt = f"""请将以下计算机视觉论文分类到最合适的子类别中。

论文标题：{title}
摘要：{abstract}
主类别：{main_category}

可选的子类别（请只从以下选项中选择一个）：
{subcategory_list}

请直接返回子类别名称，不要有任何解释。如果无法确定，请返回"未指定"。"""
            
            try:
                response = self.client.chat.completions.create(
                    model="glm-4-air",  # 修改为 flashx 版本
                    messages=[{"role": "user", "content": simple_prompt}],
                    temperature=0.1,
                    max_tokens=50,
                    top_p=0.7,
                )
                
                second_subcategory = response.choices[0].message.content.strip()
                
                if second_subcategory in subcategories:
                    logger.info(f"第二次子类别分类结果: {second_subcategory}")
                    return second_subcategory
            except Exception as e:
                logger.error(f"第二次子类别分类出错: {str(e)}")
                
            logger.warning("无法确定子类别，返回'未指定'")
            return "未指定"
        
        except Exception as e:
            logger.error(f"确定子类别出错: {str(e)}")
            logger.debug(traceback.format_exc())
            return "未指定"

    def _call_chatglm_api(self, messages, model="glm-4", temperature=0.01, max_tokens=2000):
        """
        调用ChatGLM API
        
        Args:
            messages: 消息列表
            model: 模型名称
            temperature: 采样温度
            max_tokens: 最大输出长度
            
        Returns:
            模型响应或None(错误时)
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"API调用失败: {str(e)}")
            logger.error(traceback.format_exc())
            return None
