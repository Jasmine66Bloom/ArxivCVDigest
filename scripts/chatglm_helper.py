"""ChatGLM助手：用于论文标题翻译和分类"""
from zhipuai import ZhipuAI
from typing import Tuple, List
import time
import re
from collections import defaultdict
import categories_config
import json
import math

class ChatGLMHelper:
    def __init__(self):
        """初始化ChatGLM客户端"""
        api_key = '1e8f0dd1b174e7b06bb3bce603856f62.qY4xvp6L8wEOJlRN'
        if not api_key:
            raise ValueError("请设置环境变量 ZHIPUAI_API_KEY")
        self.client = ZhipuAI(api_key=api_key)

    def translate_title(self, title: str, abstract: str) -> str:
        """
        使用ChatGLM翻译论文标题
        Args:
            title: 论文英文标题
            abstract: 论文摘要，用于提供上下文
        Returns:
            str: 中文标题
        """
        max_retries = 3
        retry_delay = 2  # 重试延迟秒数

        prompt = f"""将以下计算机视觉论文标题翻译成中文：
{title}

要求：
1. 保持专业术语准确性
2. 保留原文缩写和专有名词
3. 翻译要简洁明了
4. 只返回中文标题"""

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="glm-4-flash",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=100,
                    top_p=0.7,
                )
                translation = response.choices[0].message.content.strip()
                # 确保返回的是中文
                if any('\u4e00' <= char <= '\u9fff' for char in translation):
                    return translation
                else:
                    print(f"警告：第{attempt + 1}次翻译未返回中文结果，重试中...")
            except Exception as e:
                print(f"翻译出错 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                continue
        
        return f"[翻译失败] {title}"

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
            for neg_keyword in negative_keywords:
                neg_keyword_lower = neg_keyword.lower()
                neg_keyword_words = set(neg_keyword_lower.split()) - common_words
                
                # 完整短语匹配（严重惩罚）
                if neg_keyword_lower in title_lower:
                    negative_score += 1.0  # 标题中的负向关键词严重惩罚
                    matched_keywords.append(f"负向完整匹配(标题): {neg_keyword}")
                elif neg_keyword_lower in abstract_lower:
                    negative_score += 0.7  # 摘要中的负向关键词严重惩罚
                    matched_keywords.append(f"负向完整匹配(摘要): {neg_keyword}")
                
                # 关键词组合匹配（中度惩罚）
                elif neg_keyword_words.issubset(title_words):
                    negative_score += 0.8
                    matched_keywords.append(f"负向词组匹配(标题): {neg_keyword}")
                elif neg_keyword_words.issubset(abstract_words):
                    negative_score += 0.5
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
            threshold = categories_config.CATEGORY_THRESHOLDS.get("其他", 1.8)  # 使用"其他"类别的阈值作为基准
            
            # 筛选得分显著的类别（得分至少为最高分的70%）
            significant_categories = []
            for category, score in category_scores.items():
                category_threshold = categories_config.CATEGORY_THRESHOLDS.get(category, 1.0)
                relative_threshold = max_score * 0.7  # 相对阈值：最高分的70%
                
                # 同时满足绝对阈值和相对阈值
                if score > category_threshold and score > relative_threshold:
                    significant_categories.append((category, score))
            
            # 如果有显著类别，按分数排序返回
            if significant_categories:
                return sorted(significant_categories, key=lambda x: x[1], reverse=True)
            
            # 如果没有显著类别，但有类别得分超过其他类别阈值的一半，选择得分最高的
            elif max_score > threshold / 2:
                return [(max(category_scores.items(), key=lambda x: x[1])[0], max_score)]
        
        # 如果没有合适的类别，返回"其他"
        return [("其他", 0.0)]

    def get_category_by_semantic(self, title: str, abstract: str) -> List[Tuple[str, float]]:
        """使用语义分析进行分类"""
        categories_str = "\n".join(f"- {cat}" for cat in categories_config.ALL_CATEGORIES)
        
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
                model="glm-4",  # 使用功能更强的模型
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
                print(f"JSON解析失败: {json_str}")
                return [("其他", 0.0)]
                
        except Exception as e:
            print(f"语义分析出错: {str(e)}")
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
        high_priority_categories = ["扩散桥", "具身智能", "流模型"]
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
        try:
            # 先尝试使用关键词匹配
            keyword_category, confidence = self.get_category_by_keywords(title, abstract)[0]
            if keyword_category != "其他" and confidence >= 2.0:
                return keyword_category
            
            # 构建分类提示词，强调分类准确性
            prompt = f"{categories_config.CATEGORY_PROMPT}\n\n论文标题：{title}\n摘要：{abstract}\n\n注意：\n1. 请仔细分析论文的核心技术和主要贡献\n2. 选择最能体现论文主要工作的类别\n3. 如果论文跨多个领域，选择最核心的一个\n4. 只有在完全无法确定时才返回'其他'"
            
            # 调用 ChatGLM 进行分类
            response = self.client.chat.completions.create(
                model="glm-4-flash",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=50,
                top_p=0.7,
            )
            
            # 获取分类结果
            category = response.choices[0].message.content.strip()
            
            # 验证返回的类别是否在预定义类别中
            if category in categories_config.ALL_CATEGORIES:
                # 如果ChatGLM返回的类别与关键词匹配的类别不同，且都不是"其他"
                if keyword_category != "其他" and category != keyword_category:
                    # 进行第二次确认
                    confirm_prompt = f"{prompt}\n\n系统发现这篇论文可能属于'{keyword_category}'类别，请再次确认最合适的分类。"
                    response = self.client.chat.completions.create(
                        model="glm-4-flash",
                        messages=[{"role": "user", "content": confirm_prompt}],
                        temperature=0.1,
                        max_tokens=50,
                        top_p=0.7,
                    )
                    final_category = response.choices[0].message.content.strip()
                    if final_category in categories_config.ALL_CATEGORIES:
                        return final_category
                
                return category
            
            # 如果返回的类别不在预定义类别中，使用关键词匹配的结果
            if keyword_category != "其他":
                return keyword_category
            
            return "其他"
                
        except Exception as e:
            print(f"ChatGLM 分类出错: {str(e)}")
            # 发生错误时，如果关键词匹配有结果就返回
            if keyword_category != "其他":
                return keyword_category
            return "其他"

    def analyze_paper_contribution(self, title: str, abstract: str) -> dict:
        """分析论文的核心贡献和解决的问题
    
        Args:
            title: 论文标题
            abstract: 论文摘要
    
        Returns:
            dict: 包含分析结果的字典
        """
        prompt = f"""请分析以下计算机视觉论文的核心贡献。
    
论文标题：{title}
论文摘要：{abstract}

请用中文简要总结：
1. 这篇论文解决了什么问题
2. 提出了什么方法或创新点
3. 取得了什么效果

请用简洁的语言描述，避免技术细节。
"""
        try:
            response = self.client.chat.completions.create(
                model="glm-4-flash",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500,
                top_p=0.7,
            )
            
            analysis = response.choices[0].message.content.strip()
            return {
                "核心贡献": analysis
            }
            
        except Exception as e:
            print(f"分析论文贡献时出错: {str(e)}")
            return {
                "核心贡献": "分析失败"
            }

    def translate_title(self, title: str) -> str:
        """将论文标题翻译为中文
    
        Args:
            title: 英文标题
    
        Returns:
            str: 中文标题
        """
        prompt = f"""请将以下计算机视觉论文的标题翻译成中文，保持专业性和准确性：

{title}

只返回翻译结果，不需要解释。"""

        try:
            response = self.client.chat.completions.create(
                model="glm-4-flash",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200,
                top_p=0.7,
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"翻译标题时出错: {str(e)}")
            return title  # 如果翻译失败，返回原标题

    def confirm_category(self, title: str, abstract: str, initial_category: str) -> Tuple[str, float]:
        """对分类结果进行二次确认"""
        prompt = f"""确认论文是否属于"{initial_category}"类别。

标题：{title}
摘要：{abstract}

分析要点：
1. 标题中的关键技术词
2. 论文的主要目标
3. 是否有更合适的类别

请严格按照以下JSON格式回复，不要有任何其他内容：
{{
    "category": "最终类别名称",
    "confidence": 分数
}}"""

        try:
            response = self.client.chat.completions.create(
                model="glm-4-flash",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100,
                top_p=0.7,
            )
            result = response.choices[0].message.content.strip()
            
            # 清理和解析JSON
            try:
                # 首先尝试直接解析
                result_json = json.loads(result)
            except json.JSONDecodeError:
                # 如果失败，尝试清理后再解析
                cleaned_result = self.clean_json_string(result)
                try:
                    result_json = json.loads(cleaned_result)
                except json.JSONDecodeError as e:
                    print(f"JSON解析失败: {str(e)}")
                    print(f"原始响应: {result}")
                    return (initial_category, 1.5)
            
            category = result_json.get("category", initial_category)
            # 确保confidence是浮点数
            try:
                confidence = float(result_json.get("confidence", 0))
            except (TypeError, ValueError):
                confidence = 1.5
            
            return (category, confidence)
        except Exception as e:
            print(f"分类确认出错: {str(e)}")
            return (initial_category, 1.5)
