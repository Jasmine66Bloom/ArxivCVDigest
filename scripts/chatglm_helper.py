"""LLM助手：用于论文标题翻译和分类（支持豆包和ChatGLM双模型）"""
from typing import Tuple, List, Dict
import time
import re
from collections import defaultdict
import categories_config
import json
import math

# 导入配置
from config import LLM_PROVIDER

class ChatGLMHelper:
    """LLM助手类（支持豆包和ChatGLM双模型）"""
    
    def __init__(self):
        """根据配置初始化对应的LLM客户端"""
        self.provider = LLM_PROVIDER.lower()
        
        if self.provider == "doubao":
            from doubao_client import DoubaoClient
            from config import DOUBAO_API_KEY, DOUBAO_MODEL
            if not DOUBAO_API_KEY:
                raise ValueError("请在config.py中设置DOUBAO_API_KEY")
            self.client = DoubaoClient(api_key=DOUBAO_API_KEY, model=DOUBAO_MODEL)
            self.model = DOUBAO_MODEL
            print(f"🤖 使用豆包大模型: {self.model}")
        elif self.provider == "chatglm":
            from chatglm_client import ChatGLMClient
            from config import CHATGLM_API_KEY, CHATGLM_MODEL, CHATGLM_BASE_URL, CHATGLM_ENABLE_THINKING
            if not CHATGLM_API_KEY:
                raise ValueError("请在config.py中设置CHATGLM_API_KEY")
            self.client = ChatGLMClient(api_key=CHATGLM_API_KEY, model=CHATGLM_MODEL, base_url=CHATGLM_BASE_URL)
            self.enable_thinking = CHATGLM_ENABLE_THINKING
            self.model = CHATGLM_MODEL
            print(f"🤖 使用ChatGLM模型: {self.model}")
        else:
            raise ValueError(f"不支持的LLM提供商: {LLM_PROVIDER}，请在config.py中设置LLM_PROVIDER为'doubao'或'chatglm'")

    def translate_title(self, title: str, abstract: str = "") -> str:
        """
        使用ChatGLM翻译论文标题，增强的提示词和错误处理
        Args:
            title: 论文英文标题
            abstract: 论文摘要，用于提供上下文（可选）
        Returns:
            str: 中文标题
        """
        max_retries = 10
        retry_delay = 2  # 重试延迟秒数
        
        # 提取摘要中的关键句子作为上下文
        context = ""
        if abstract and len(abstract) > 50:
            # 取摘要的前150个字符作为上下文
            context = f"""论文摘要开头：
{abstract[:150]}..."""

        # 简洁的提示词，直接要求返回中文标题
        prompt = f"""将以下计算机视觉论文标题翻译成中文。

要求：
- 专业术语、模型名称、算法名称、缩写保持英文原样（如CLIP、ViT、NeRF、3D等）
- 只输出翻译结果，不要解释

标题：{title}"""

        for attempt in range(max_retries):
            try:
                # 构建 API 请求参数
                request_params = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 150,
                    "top_p": 0.7,
                }
                
                # 对于 glm-4.7 模型，根据配置设置 thinking 参数
                if self.provider == "chatglm" and "glm-4.7" in self.model:
                    if self.enable_thinking:
                        request_params["thinking"] = {"type": "enabled"}
                    else:
                        request_params["thinking"] = {"type": "disabled"}
                
                response = self.client.chat.completions.create(**request_params)
                translation = response.choices[0].message.content.strip()
                # 清理可能的多余内容，只保留第一行
                if '\n' in translation:
                    translation = translation.split('\n')[0].strip()
                # 确保返回的是中文
                if translation and any('\u4e00' <= char <= '\u9fff' for char in translation):
                    return translation
                else:
                    print(f"警告：第{attempt + 1}次翻译未返回中文结果，重试中...")
                    time.sleep(retry_delay)
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
        
        # 如果没有合适的类别，返回"其他"
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
                model=self.model,  # 使用功能更强的模型
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
                    # print(f"关键词高置信度匹配: {keyword_category} (置信度: {confidence:.2f})")
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
                model=self.model,  # 修改为 flashx 版本
                messages=[{"role": "user", "content": prompt}],
                temperature=0.05,
                max_tokens=50,
                top_p=0.5,
            )
            
            # 获取分类结果
            category = response.choices[0].message.content.strip()
            
            # 验证返回的类别是否在预定义类别中
            if category in categories_config.CATEGORY_DISPLAY_ORDER:
                # 如果ChatGLM返回的类别与关键词匹配的类别不同，且关键词匹配有较高置信度
                if keyword_category != "其他" and category != keyword_category and confidence >= 2.0:
                    # 进行第二次确认，提供更多上下文和详细比较
                    confirm_prompt = f"""我需要你再次确认这篇论文的最佳分类类别。

论文标题: {title}

论文摘要: {abstract}

当前有两个候选类别:
1. "{keyword_category}" - 由关键词匹配算法推荐，置信度为 {confidence:.2f}
2. "{category}" - 由你的初步分析推荐

请仔细比较这两个类别，考虑以下因素:
- 论文的核心技术路线是什么？
- 论文的主要创新点在哪个领域？
- 论文的方法和技术与哪个类别的描述更匹配？
- 如果论文跨越多个领域，哪个是最核心的？

{categories_config.CATEGORY_PROMPT}

请选择最能准确反映论文核心内容的类别，直接返回类别名称，不要有任何解释。"""
                    
                    # 使用更强大的模型进行确认
                    response = self.client.chat.completions.create(
                        model=self.model,  # 使用更强大的模型
                        messages=[{"role": "user", "content": confirm_prompt}],
                        temperature=0.05,
                        max_tokens=50,
                        top_p=0.5,
                    )
                    final_category = response.choices[0].message.content.strip()
                    
                    if final_category in categories_config.CATEGORY_DISPLAY_ORDER:
                        # print(f"分类冲突解决: 从 '{keyword_category}' 和 '{category}' 中选择 '{final_category}'")
                        return final_category
                    else:
                        # 如果确认结果不在预定义类别中，使用初始 ChatGLM 结果
                        print(f"使用初始 ChatGLM 分类结果: {category}")
                        return category
                
                # print(f"ChatGLM 分类结果: {category}")
                return category
            
            # 如果返回的类别不在预定义类别中，尝试修正
            # print(f"ChatGLM 返回的类别 '{category}' 不在预定义类别中，尝试修正...")
            
            # 尝试从 ChatGLM 的回复中提取有效类别
            for cat in categories_config.CATEGORY_DISPLAY_ORDER:
                if cat.lower() in category.lower():
                    print(f"从回复中提取到有效类别: {cat}")
                    return cat
            
            # 如果无法从回复中提取，使用关键词匹配的结果（如果有较高置信度）
            if keyword_category != "其他" and confidence >= 1.5:
                # print(f"使用关键词匹配结果: {keyword_category} (置信度: {confidence:.2f})")
                return keyword_category
            
            # 最后尝试使用语义分类
            semantic_results = self.get_category_by_semantic(title, abstract)
            if semantic_results:
                semantic_category, semantic_confidence = semantic_results[0]
                if semantic_category != "其他":
                    # print(f"使用语义分类结果: {semantic_category} (置信度: {semantic_confidence:.2f})")
                    return semantic_category
            
            # 如果所有方法都失败，返回"其他"
            print("无法确定类别，返回'其他'")
            return "其他"
                
        except Exception as e:
            print(f"ChatGLM 分类出错: {str(e)}")
            # 发生错误时，如果关键词匹配有结果就返回
            if keyword_category != "其他" and confidence >= 1.2:
                return keyword_category
            return "其他"

    def analyze_paper_contribution(self, title: str, abstract: str) -> dict:
        """分析论文的核心贡献，以单句话总结的形式返回
    
        Args:
            title: 论文标题
            abstract: 论文摘要
    
        Returns:
            dict: 包含分析结果的字典，只有一个键"核心贡献"，值为单句话总结
        """
        prompt = f"""用一句话（不超过50字）总结这篇论文的核心贡献，只输出总结内容。

标题：{title}
摘要：{abstract[:500] if len(abstract) > 500 else abstract}"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200,
                top_p=0.7,
            )
            
            # 获取单句话总结
            contribution_summary = response.choices[0].message.content.strip()
            
            # 去除可能的多余内容，只保留第一句话
            if '\n' in contribution_summary:
                contribution_summary = contribution_summary.split('\n')[0].strip()
                
            # 确保长度适当
            if len(contribution_summary) > 100:
                contribution_summary = contribution_summary[:97] + '...'
                
            return {
                "核心贡献": contribution_summary
            }
            
        except Exception as e:
            print(f"分析论文贡献时出错: {str(e)}")
            return {
                "核心贡献": "分析失败"
            }

    def _extract_json_field(self, text: str, field: str) -> str:
        """从文本中提取 JSON 字段值
        
        Args:
            text: 可能包含 JSON 的文本
            field: 要提取的字段名
            
        Returns:
            str: 字段值，如果提取失败返回空字符串
        """
        try:
            # 先清理 JSON 字符串
            json_str = self.clean_json_string(text)
            data = json.loads(json_str)
            return data.get(field, "")
        except (json.JSONDecodeError, Exception):
            # 如果 JSON 解析失败，尝试直接从文本中提取
            # 查找 "field": "value" 模式
            import re
            pattern = rf'"{field}"\s*:\s*"([^"]*)"'
            match = re.search(pattern, text)
            if match:
                return match.group(1)
            return ""

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
                model=self.model,
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

    def decide_category(self, title: str, abstract: str, candidate_categories: List[Tuple], match_details: Dict = None) -> str:
        """使用ChatGLM从候选类别中决定最终分类
        
        Args:
            title: 论文标题
            abstract: 论文摘要
            candidate_categories: 候选类别列表，每个元素是(category, score, subcategory)的元组
            match_details: 关键词匹配详情字典，可选
            
        Returns:
            最终决定的类别名称
        """
        # 初始化匹配详情字典，如果没有提供
        if match_details is None:
            match_details = {}
        try:
            import categories_config
            
            # 格式化候选类别信息
            candidates_info = ""
            for i, (category, score, subcategory) in enumerate(candidate_categories[:3], 1):
                subcategory_info = f", 子类别: {subcategory[0] if subcategory else '无'}" if subcategory else ""
                candidates_info += f"\n{i}. {category} (得分: {score:.2f}{subcategory_info})"
            
            # 提取摘要中的关键信息
            key_info = ""
            if len(abstract) > 100:
                # 提取摘要的开头和结尾，这通常包含最重要的信息
                intro = abstract[:200] if len(abstract) > 200 else abstract
                conclusion = abstract[-200:] if len(abstract) > 400 else ""
                key_info = f"摘要开头: {intro}\n"
                if conclusion:
                    key_info += f"摘要结尾: {conclusion}\n"
            
            # 格式化候选类别信息，增加更多上下文
            detailed_candidates = ""
            for i, (category, score, subcategory) in enumerate(candidate_categories[:5], 1):
                subcategory_info = f", 子类别: {subcategory[0] if subcategory else '无'}" if subcategory else ""
                matches = match_details.get(category, [])[:3]
                match_info = f", 关键匹配: {', '.join(matches)}" if matches else ""
                detailed_candidates += f"\n{i}. {category} (得分: {score:.2f}{subcategory_info}{match_info})"
            
            # 构建增强的提示词，提供更多上下文和更精确的指导
            prompt = f"""请作为一位资深的计算机视觉领域研究专家，对以下论文进行精确分类。请仔细分析论文的核心技术和创新点。

论文标题: {title}

{key_info}
候选类别:{detailed_candidates}

{categories_config.CATEGORY_PROMPT}

分类指南:
1. 仅从上述候选类别中选择一个最能代表论文核心创新点的类别
2. 请基于论文的技术本质做出判断，而不是仅基于关键词匹配
3. 如果论文主要是将现有技术应用到特定领域，而没有显著的技术创新，请选择“领域特定视觉应用”
4. 如果论文提出了新的算法、模型或方法，请选择最能代表这一创新的技术类别
5. 如果论文涉及多个领域，请选择最能代表其核心创新的类别

请直接返回最合适的类别名称，不要有任何解释或额外文本。只返回类别名称。"""
            
            # 调整模型参数，提高稳定性
            temperature = 0.01  # 极低的温度提高稳定性
            
            # 调用 ChatGLM 进行分类决策
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,  # 使用极低的温度提高稳定性
                max_tokens=50,
                top_p=0.3,  # 降低top_p以提高稳定性
            )
            
            # 获取分类结果
            category = response.choices[0].message.content.strip()
            
            # 验证返回的类别是否在候选类别中
            candidate_names = [c[0] for c in candidate_categories]
            
            # 如果返回的类别在候选类别中，直接返回
            if category in candidate_names:
                return category
            
            # 如果返回的类别不在候选类别中，但在预定义类别中，也返回
            if category in categories_config.CATEGORY_DISPLAY_ORDER:
                return category
            
            # 如果都不匹配，返回得分最高的候选类别
            return candidate_categories[0][0]
            
        except Exception as e:
            print(f"ChatGLM 分类决策出错: {str(e)}")
            # 发生错误时，返回得分最高的候选类别
            return candidate_categories[0][0] if candidate_categories else "其他"

    def determine_subcategory(self, title: str, abstract: str, main_category: str) -> str:
        """确定论文的子类别
        
        Args:
            title: 论文标题
            abstract: 论文摘要
            main_category: 主类别
            
        Returns:
            str: 子类别名称
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
                model=self.model,  # 修改为 flashx 版本
                messages=[{"role": "user", "content": prompt}],
                temperature=0.05,
                max_tokens=50,
                top_p=0.5,
            )
            
            # 获取分类结果
            subcategory = response.choices[0].message.content.strip()
            
            # 验证返回的子类别是否在预定义子类别中
            if subcategory in subcategories:
                # print(f"子类别分类结果: {subcategory}")
                return subcategory
            
            # 尝试模糊匹配 - 检查返回的子类别是否包含在任何预定义子类别中
            for sub in subcategories.keys():
                if sub.lower() in subcategory.lower() or subcategory.lower() in sub.lower():
                    # print(f"子类别模糊匹配: 从 '{subcategory}' 匹配到 '{sub}'")
                    return sub
            
            # 如果不在预定义子类别中，尝试关键词匹配
            # print(f"子类别 '{subcategory}' 不在预定义子类别中，尝试关键词匹配...")
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
                # print(f"子类别关键词匹配: {best_match} (分数: {best_score:.2f})")
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
                    model=self.model,  # 修改为 flashx 版本
                    messages=[{"role": "user", "content": simple_prompt}],
                    temperature=0.1,
                    max_tokens=50,
                    top_p=0.7,
                )
                
                second_subcategory = response.choices[0].message.content.strip()
                
                if second_subcategory in subcategories:
                    # print(f"第二次子类别分类结果: {second_subcategory}")
                    return second_subcategory
            except Exception as e:
                print(f"第二次子类别分类出错: {str(e)}")
                
            # print(f"无法确定子类别，返回'未指定'")
            return "未指定"
        
        except Exception as e:
            print(f"确定子类别出错: {str(e)}")
            return "未指定"
