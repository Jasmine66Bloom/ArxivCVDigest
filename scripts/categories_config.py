"""
计算机视觉论文分类配置文件 (2025前瞻版)
包含一级分类和二级分类的层次结构，以及中英文对照和元数据
"""

# 类别阈值配置（越大越严格）
CATEGORY_THRESHOLDS = {
    # 一、前沿研究方向（优先展示）
    "基础智能与认知 (Foundation Intelligence & Cognition)": {
        "threshold": 1.15,  # 降低阈值，更容易匹配
        "subcategories": {
            "涌现能力 (Emergent Capabilities)": 1.25,
            "自监督学习 (Self-supervised Learning)": 1.25,
            "神经符号推理 (Neuro-symbolic Reasoning)": 1.25,
            "认知计算 (Cognitive Computing)": 1.25,
            "大规模预训练 (Large-scale Pretraining)": 1.25,
            "智能评估 (Intelligence Evaluation)": 1.25,
        },
        "priority": 5,  # 最高优先级
        "metadata": {
            "description": "研究视觉智能的基础理论、大规模视觉模型、涌现能力和认知计算等前沿课题",
            "related_venues": ["CVPR", "ICLR", "NeurIPS", "ICCV"],
            "trending_score": 9.8,  # 1-10分，表示研究热度
            "industry_relevance": 9.5  # 1-10分，表示产业相关性
        }
    },
    
    "生成式建模 (Generative Modeling)": {
        "threshold": 1.15,
        "subcategories": {
            "扩散模型 (Diffusion Models)": 1.25,
            "视频生成 (Video Generation)": 1.25,
            "3D生成建模 (3D Generation)": 1.25,
            "可控生成 (Controllable Generation)": 1.25,
            "物理约束生成 (Physics-constrained Generation)": 1.25,
            "生成式编辑 (Generative Editing)": 1.25,
        },
        "priority": 4.8,  # 高优先级，略微提高
        "metadata": {
            "description": "研究生成高质量视觉内容的方法，包括图像、视频和3D内容的生成与编辑",
            "related_venues": ["CVPR", "ICCV", "ECCV", "SIGGRAPH"],
            "trending_score": 9.9,  # 1-10分，表示研究热度
            "industry_relevance": 9.8  # 1-10分，表示产业相关性
        }
    },
    
    "多模态学习 (Multimodal Learning)": {
        "threshold": 1.15,
        "subcategories": {
            "视觉语言模型 (Vision-Language Models)": 1.25,
            "跨模态对齐 (Cross-modal Alignment)": 1.25,
            "多模态推理 (Multimodal Reasoning)": 1.25,
            "多模态表征 (Multimodal Representation)": 1.25,
            "多感知融合 (Multi-sensory Fusion)": 1.25,
            "多模态生成 (Multimodal Generation)": 1.25,
        },
        "priority": 4.7,  # 高优先级，略微提高
        "metadata": {
            "description": "研究视觉与其他模态（如语言、音频）的协同理解和交互，包括视觉-语言模型和跨模态推理",
            "related_venues": ["CVPR", "ICCV", "ACL", "EMNLP", "NeurIPS"],
            "trending_score": 9.7,  # 1-10分，表示研究热度
            "industry_relevance": 9.6  # 1-10分，表示产业相关性
        }
    },
    
    "自适应学习 (Adaptive Learning)": {
        "threshold": 1.15,
        "subcategories": {
            "持续学习 (Continual Learning)": 1.25,
            "开放世界识别 (Open-world Recognition)": 1.25,
            "少样本学习 (Few-shot Learning)": 1.25,
            "域适应与泛化 (Domain Adaptation & Generalization)": 1.25,
            "自校准系统 (Self-calibrating Systems)": 1.25,
            "主动学习 (Active Learning)": 1.25,
        },
        "priority": 4.5,  # 高优先级
        "metadata": {
            "description": "研究视觉系统在新环境、新任务中的适应能力，包括持续学习、域适应和少样本学习",
            "related_venues": ["CVPR", "ICLR", "NeurIPS", "ECCV"],
            "trending_score": 8.7,  # 1-10分，表示研究热度
            "industry_relevance": 8.5  # 1-10分，表示产业相关性
        }
    },
    
    # 二、核心技术方向
    "感知与识别 (Perception & Recognition)": {
        "threshold": 1.2,
        "subcategories": {
            "目标检测 (Object Detection)": 1.3,
            "图像分割 (Image Segmentation)": 1.3,
            "图像分类 (Image Classification)": 1.3,
            "视频理解 (Video Understanding)": 1.3,
            "图像增强 (Image Enhancement)": 1.3,
            "密集预测 (Dense Prediction)": 1.3,
        },
        "priority": 3.5,  # 中等优先级，略微提高
        "metadata": {
            "description": "研究计算机视觉的基础任务和技术，如检测、分割、分类和视频理解等",
            "related_venues": ["CVPR", "ICCV", "ECCV", "TPAMI"],
            "trending_score": 7.8,  # 1-10分，表示研究热度
            "industry_relevance": 9.0  # 1-10分，表示产业相关性
        }
    },
    
    "神经渲染与场景理解 (Neural Rendering & Scene Understanding)": {
        "threshold": 1.15,
        "subcategories": {
            "神经辐射场 (Neural Radiance Fields)": 1.25,
            "动态场景表示 (Dynamic Scene Representation)": 1.25,
            "隐式几何 (Implicit Geometry)": 1.25,
            "场景分解 (Scene Decomposition)": 1.25,
            "时空预测 (Spatio-temporal Prediction)": 1.25,
            "新视角合成 (Novel View Synthesis)": 1.25,
        },
        "priority": 4.2,  # 中等优先级，提高优先级
        "metadata": {
            "description": "研究基于神经网络的场景表示和理解方法，包括NeRF、3D高斯散射和隐式几何等",
            "related_venues": ["CVPR", "ICCV", "ECCV", "SIGGRAPH"],
            "trending_score": 9.5,  # 1-10分，表示研究热度
            "industry_relevance": 8.8  # 1-10分，表示产业相关性
        }
    },
    
    "具身智能 (Embodied Intelligence)": {
        "threshold": 1.2,
        "subcategories": {
            "导航与规划 (Navigation & Planning)": 1.3,
            "交互感知 (Interactive Perception)": 1.3,
            "物理属性感知 (Physical Properties)": 1.3,
            "强化学习 (Reinforcement Learning)": 1.3,
            "协作系统 (Collaborative Systems)": 1.3,
            "操作控制 (Manipulation Control)": 1.3,
        },
        "priority": 3.8,  # 中等优先级，略微提高
        "metadata": {
            "description": "研究视觉系统与物理世界交互的能力，包括视觉引导的动作规划和物理感知等",
            "related_venues": ["CVPR", "ICCV", "CoRL", "RSS", "ICRA"],
            "trending_score": 8.6,  # 1-10分，表示研究热度
            "industry_relevance": 8.2  # 1-10分，表示产业相关性
        }
    },
    
    "高效计算 (Efficient Computing)": {
        "threshold": 1.2,
        "subcategories": {
            "模型压缩 (Model Compression)": 1.3,
            "稀疏计算 (Sparse Computation)": 1.3,
            "硬件协同 (Hardware Co-design)": 1.3,
            "低功耗系统 (Low-power Systems)": 1.3,
            "边缘计算 (Edge Computing)": 1.3,
            "架构搜索 (Architecture Search)": 1.3,
        },
        "priority": 3.5,  # 中等优先级，略微提高
        "metadata": {
            "description": "研究提高视觉系统效率的方法，包括模型压缩、硬件优化和边缘计算等",
            "related_venues": ["CVPR", "ICCV", "ECCV", "ICLR", "MLSys"],
            "trending_score": 8.4,  # 1-10分，表示研究热度
            "industry_relevance": 9.2  # 1-10分，表示产业相关性
        }
    },
    
    "可信系统 (Trustworthy Systems)": {
        "threshold": 1.2,
        "subcategories": {
            "鲁棒性 (Robustness)": 1.3,
            "可解释性 (Interpretability)": 1.3,
            "不确定性 (Uncertainty)": 1.3,
            "公平与伦理 (Fairness & Ethics)": 1.3,
            "隐私保护 (Privacy Preservation)": 1.3,
            "安全性 (Security)": 1.3,
        },
        "priority": 3.6,  # 中等优先级，略微提高
        "metadata": {
            "description": "研究视觉系统的可靠性、安全性和伦理问题，包括鲁棒性、可解释性和隐私保护等",
            "related_venues": ["CVPR", "ICCV", "ECCV", "FAccT", "AIES"],
            "trending_score": 8.5,  # 1-10分，表示研究热度
            "industry_relevance": 8.7  # 1-10分，表示产业相关性
        }
    },
    
    # 三、应用方向（后展示）
    "领域应用 (Domain Applications)": {
        "threshold": 1.2,
        "subcategories": {
            "医学影像与分析 (Medical Imaging & Analysis)": 1.3,
            "自动驾驶 (Autonomous Driving)": 1.3,
            "机器人技术 (Robotics)": 1.3,
            "增强与虚拟现实 (AR/VR)": 1.3,
            "工业检测 (Industrial Inspection)": 1.3,
            "农业与环境 (Agriculture & Environment)": 1.3,
        },
        "priority": 2.5,  # 较低优先级，略微提高
        "metadata": {
            "description": "研究视觉技术在特定领域的应用，如医疗、自动驾驶、机器人和工业等",
            "related_venues": ["CVPR", "ICCV", "MICCAI", "ITSC", "ICRA"],
            "trending_score": 8.0,  # 1-10分，表示研究热度
            "industry_relevance": 9.5  # 1-10分，表示产业相关性
        }
    },
    
    # 其他类别
    "其他 (Others)": {
        "threshold": 1.7,           # 提高阈值，更难匹配
        "subcategories": {},
        "priority": 1,  # 最低优先级
        "metadata": {
            "description": "其他不能明确归类到上述类别的视觉研究工作",
            "related_venues": ["CVPR", "ICCV", "ECCV"],
            "trending_score": 5.0,  # 1-10分，表示研究热度
            "industry_relevance": 5.0  # 1-10分，表示产业相关性
        }
    }
}

# 类别显示顺序配置（从上到下）
CATEGORY_DISPLAY_ORDER = [
    # 一、前沿研究方向（优先展示）
    "基础智能与认知 (Foundation Intelligence & Cognition)",
    "生成式建模 (Generative Modeling)",
    "多模态学习 (Multimodal Learning)",
    "自适应学习 (Adaptive Learning)",
    
    # 二、核心技术方向
    "神经渲染与场景理解 (Neural Rendering & Scene Understanding)",
    "感知与识别 (Perception & Recognition)",
    "具身智能 (Embodied Intelligence)",
    "高效计算 (Efficient Computing)",
    "可信系统 (Trustworthy Systems)",
    
    # 三、应用方向（后展示）
    "领域应用 (Domain Applications)",
    
    # 其他类别（永远放在最后）
    "其他 (Others)"
]

# 分类提示词
CATEGORY_PROMPT = """
请将以下计算机视觉论文分类到最合适的类别中。

2025年计算机视觉分类体系：

一、前沿研究方向
1. 基础智能与认知 (Foundation Intelligence & Cognition)：涌现能力、自监督学习、神经符号推理、认知计算、大规模预训练、智能评估
   定义：研究视觉智能的基础理论、大规模视觉模型、涌现能力和认知计算等前沿课题。

2. 生成式建模 (Generative Modeling)：扩散模型、视频生成、3D生成建模、可控生成、物理约束生成、生成式编辑
   定义：研究生成高质量视觉内容的方法，包括图像、视频和3D内容的生成与编辑。

3. 多模态学习 (Multimodal Learning)：视觉语言模型、跨模态对齐、多模态推理、多模态表征、多感知融合、多模态生成
   定义：研究视觉与其他模态（如语言、音频）的协同理解和交互，包括视觉-语言模型和跨模态推理。

4. 自适应学习 (Adaptive Learning)：持续学习、开放世界识别、少样本学习、域适应与泛化、自校准系统、主动学习
   定义：研究视觉系统在新环境、新任务中的适应能力，包括持续学习、域适应和少样本学习。

二、核心技术方向
5. 神经渲染与场景理解 (Neural Rendering & Scene Understanding)：神经辐射场、动态场景表示、隐式几何、场景分解、时空预测、新视角合成
   定义：研究基于神经网络的场景表示和理解方法，包括NeRF、3D高斯散射和隐式几何等。

6. 感知与识别 (Perception & Recognition)：目标检测、图像分割、图像分类、视频理解、图像增强、密集预测
   定义：研究计算机视觉的基础任务和技术，如检测、分割、分类和视频理解等。

7. 具身智能 (Embodied Intelligence)：导航与规划、交互感知、物理属性感知、强化学习、协作系统、操作控制
   定义：研究视觉系统与物理世界交互的能力，包括视觉引导的动作规划和物理感知等。

8. 高效计算 (Efficient Computing)：模型压缩、稀疏计算、硬件协同、低功耗系统、边缘计算、架构搜索
   定义：研究提高视觉系统效率的方法，包括模型压缩、硬件优化和边缘计算等。

9. 可信系统 (Trustworthy Systems)：鲁棒性、可解释性、不确定性、公平与伦理、隐私保护、安全性
   定义：研究视觉系统的可靠性、安全性和伦理问题，包括鲁棒性、可解释性和隐私保护等。

三、应用方向
10. 领域应用 (Domain Applications)：医学影像与分析、自动驾驶、机器人技术、增强与虚拟现实、工业检测、农业与环境
    定义：研究视觉技术在特定领域的应用，如医疗、自动驾驶、机器人和工业等。

11. 其他 (Others)：不属于以上类别的论文
    定义：其他不能明确归类到上述类别的视觉研究工作。

分类指南：
1. 首先分析论文的核心技术贡献和主要研究目标
2. 考虑论文的方法、实验和应用场景
3. 如果论文涉及多个类别，请选择最核心、最具创新性的方向
4. 优先考虑技术本质而非应用领域（除非应用创新是论文的主要贡献）
5. 只有在确实无法归类到前10个类别时，才选择"其他"类别

边界案例处理：
- 如果论文同时涉及"生成式建模"和"多模态学习"，但核心是文本引导的图像生成，应归类为"生成式建模"
- 如果论文研究NeRF技术，即使应用于医疗领域，也应优先归类为"神经渲染与场景理解"而非"领域应用"
- 如果论文提出新的视觉基础模型并展示了涌现能力，应归类为"基础智能与认知"而非"感知与识别"

请分析论文的核心技术和主要贡献，选择最合适的一个类别。只返回类别名称，不要有任何解释或额外文本。
"""

# 类别关键词配置
CATEGORY_KEYWORDS = {
    # 一、前沿研究方向（优先展示）
    "基础智能与认知 (Foundation Intelligence & Cognition)": {
        "keywords": [
            # 涌现能力（高权重）
            ("emergent capabilities", 2.0),              # 涌现能力
            ("emergent property", 1.8),            # 涌现特性
            ("scaling law", 1.8),                  # 缩放定律
            ("visual reasoning", 1.8),             # 视觉推理
            
            # 自监督学习（高权重）
            ("self-supervised", 2.0),              # 自监督
            ("self-improving", 1.8),               # 自我提升
            ("autonomous visual", 1.8),            # 自主视觉
            ("self-supervised vision", 1.8),       # 自监督视觉
            ("autonomous adaptation", 1.8),        # 自主适应
            
            # 神经符号推理（高权重）
            ("neuro-symbolic", 2.0),               # 神经符号
            ("neural-symbolic reasoning", 2.0),       # 神经符号推理
            ("symbolic reasoning", 1.8),           # 符号推理
            ("visual concept", 1.8),               # 视觉概念
            ("concept learning", 1.8),             # 概念学习
            
            # 认知计算（高权重）
            ("cognitive computing", 1.8),          # 认知计算
            ("brain-inspired vision", 1.8),        # 脑启发视觉
            ("visual cognition", 1.8),             # 视觉认知
            ("cognitive architecture", 1.8),       # 认知架构
            
            # 大规模预训练（高权重）
            ("large-scale pretraining", 2.0),           # 大规模预训练
            ("foundation model", 2.0),             # 基础模型
            ("vision transformer", 1.8),           # 视觉transformer
            ("visual foundation model", 2.0),      # 视觉基础模型
        ],
        "negative_keywords": [
            ("specific application", 1.0),         # 特定应用
            ("downstream task", 0.8),              # 下游任务
        ]
    },
    
    "生成式建模 (Generative Modeling)": {
        "keywords": [
            # 扩散模型（高权重）
            ("diffusion model", 2.0),              # 扩散模型
            ("latent diffusion", 2.0),             # 潜在扩散
            ("denoising diffusion", 2.0),          # 去噪扩散
            ("score-based", 1.8),                  # 基于分数
            ("generative model", 1.8),             # 生成模型
            
            # 视频生成（高权重）
            ("video generation", 2.0),             # 视频生成
            ("video synthesis", 2.0),              # 视频合成
            ("text-to-video", 2.0),                # 文本到视频
            ("motion synthesis", 1.8),             # 运动合成
            ("temporal consistency", 1.8),         # 时间一致性
            
            # 3D生成建模（高权重）
            ("3D generation", 2.0),                # 3D生成
            ("3D synthesis", 2.0),                 # 3D合成
            ("text-to-3D", 2.0),                   # 文本到3D
            ("3D generative", 2.0),                # 3D生成式
            ("shape generation", 1.8),             # 形状生成
            
            # 可控生成（高权重）
            ("controllable generation", 2.0),      # 可控生成
            ("controlled synthesis", 2.0),         # 受控合成
            ("attribute editing", 1.8),            # 属性编辑
            ("semantic editing", 1.8),             # 语义编辑
            ("style transfer", 1.8),               # 风格迁移
            
            # 物理约束生成（高权重）
            ("physics-constrained", 2.0),          # 物理约束
            ("physically plausible", 2.0),         # 物理合理
            ("physics-guided", 1.8),               # 物理引导
            ("physical consistency", 1.8),         # 物理一致性
            ("dynamics-aware", 1.8),               # 动力学感知
        ],
        "negative_keywords": [
            ("recognition", 1.0),                  # 识别
            ("classification", 1.0),               # 分类
        ]
    },
    
    "多模态学习 (Multimodal Learning)": {
        "keywords": [
            # 视觉语言模型（高权重）
            ("vision-language model", 2.0),        # 视觉-语言模型
            ("visual-language pretraining", 2.0),  # 视觉-语言预训练
            ("multimodal foundation model", 2.0),  # 多模态基础模型
            ("CLIP", 1.8),                         # CLIP
            ("visual-text", 1.8),                  # 视觉-文本
            
            # 跨模态对齐（高权重）
            ("cross-modal alignment", 2.0),        # 跨模态对齐
            ("multimodal alignment", 2.0),         # 多模态对齐
            ("vision-language alignment", 2.0),    # 视觉-语言对齐
            ("joint embedding", 1.8),              # 联合嵌入
            ("contrastive learning", 1.8),         # 对比学习
            
            # 多模态推理（高权重）
            ("multimodal reasoning", 2.0),         # 多模态推理
            ("cross-modal reasoning", 2.0),        # 跨模态推理
            ("visual question answering", 1.8),    # 视觉问答
            ("visual reasoning", 1.8),             # 视觉推理
            ("multimodal inference", 1.8),         # 多模态推断
            
            # 多模态表征（高权重）
            ("multimodal representation", 2.0),    # 多模态表征
            ("joint representation", 2.0),         # 联合表征
            ("multimodal embedding", 1.8),         # 多模态嵌入
            ("cross-modal feature", 1.8),          # 跨模态特征
            ("multimodal fusion", 1.8),            # 多模态融合
            
            # 多感知融合（高权重）
            ("multi-sensory fusion", 2.0),         # 多感知融合
            ("audio-visual", 2.0),                 # 音视频
            ("tactile-visual", 2.0),               # 触觉-视觉
            ("multimodal perception", 1.8),        # 多模态感知
            ("cross-sensory", 1.8),                # 跨感官
        ],
        "negative_keywords": [
            ("unimodal", 1.0),                     # 单模态
            ("vision-only", 1.0),                  # 仅视觉
        ]
    },
    
    # 二、核心技术方向
    "感知与识别 (Perception & Recognition)": {
        "keywords": [
            # 目标检测（高权重）
            ("object detection", 2.0),             # 目标检测
            ("instance recognition", 2.0),         # 实例识别
            ("object recognition", 2.0),           # 目标识别
            ("face recognition", 1.8),             # 人脸识别
            ("pedestrian detection", 1.8),         # 行人检测
            
            # 图像分割（高权重）
            ("image segmentation", 2.0),           # 图像分割
            ("semantic segmentation", 2.0),        # 语义分割
            ("instance segmentation", 2.0),        # 实例分割
            ("panoptic segmentation", 2.0),        # 全景分割
            ("medical segmentation", 1.8),         # 医学分割
            
            # 图像分类（高权重）
            ("image classification", 2.0),         # 图像分类
            ("fine-grained classification", 2.0),  # 细粒度分类
            ("visual recognition", 1.8),           # 视觉识别
            ("hierarchical classification", 1.8),  # 层次分类
            ("multi-label classification", 1.8),   # 多标签分类
            
            # 视频理解（高权重）
            ("video understanding", 2.0),          # 视频理解
            ("action recognition", 2.0),           # 动作识别
            ("temporal modeling", 2.0),            # 时序建模
            ("activity recognition", 1.8),         # 活动识别
            ("video classification", 1.8),         # 视频分类
            
            # 图像增强（高权重）
            ("image enhancement", 2.0),            # 图像增强
            ("super-resolution", 2.0),             # 超分辨率
            ("image denoising", 2.0),              # 图像去噪
            ("image restoration", 2.0),            # 图像恢复
            ("image inpainting", 1.8),             # 图像修复
        ],
        "negative_keywords": [
            ("generative", 1.0),                   # 生成式
            ("multimodal", 1.0),                   # 多模态
        ]
    },
    
    "神经渲染与场景理解 (Neural Rendering & Scene Understanding)": {
        "keywords": [
            # 神经辐射场（高权重）
            ("neural radiance field", 2.0),        # 神经辐射场
            ("NeRF", 2.0),                         # NeRF
            ("neural rendering", 2.0),             # 神经渲染
            ("volumetric rendering", 1.8),         # 体积渲染
            ("3D gaussian splatting", 2.0),        # 3D高斯散射
            
            # 动态场景表示（高权重）
            ("dynamic scene representation", 2.0), # 动态场景表示
            ("4D representation", 2.0),            # 4D表示
            ("deformable", 1.8),                   # 可变形
            ("non-rigid", 1.8),                    # 非刚性
            ("dynamic NeRF", 2.0),                 # 动态NeRF
            
            # 隐式几何（高权重）
            ("implicit geometric", 2.0),           # 隐式几何
            ("implicit representation", 2.0),      # 隐式表示
            ("signed distance function", 2.0),     # 符号距离函数
            ("occupancy network", 1.8),            # 占用网络
            ("neural implicit surface", 1.8),      # 神经隐式表面
            
            # 场景分解（高权重）
            ("scene decomposition", 2.0),          # 场景分解
            ("scene reconstruction", 2.0),         # 场景重建
            ("semantic scene", 2.0),               # 语义场景
            ("object-centric", 1.8),               # 以物体为中心
            ("compositional scene", 1.8),          # 组合场景
            
            # 时空预测（高权重）
            ("spatio-temporal prediction", 2.0),   # 时空预测
            ("scene forecasting", 2.0),            # 场景预测
            ("future prediction", 1.8),            # 未来预测
            ("dynamic scene understanding", 1.8),  # 动态场景理解
            ("temporal coherence", 1.8),           # 时间一致性
        ],
        "negative_keywords": [
            ("2D only", 1.0),                      # 仅2D
            ("classification", 1.0),               # 分类
        ]
    },
    
    "具身智能 (Embodied Intelligence)": {
        "keywords": [
            # 导航与规划（高权重）
            ("visual navigation", 2.0),            # 视觉导航
            ("vision-based planning", 2.0),        # 基于视觉的规划
            ("visual path planning", 2.0),         # 视觉路径规划
            ("embodied navigation", 1.8),          # 具身导航
            ("visual exploration", 1.8),           # 视觉探索
            
            # 交互感知（高权重）
            ("interactive perception", 2.0),       # 交互式感知
            ("active perception", 2.0),            # 主动感知
            ("interactive scene understanding", 2.0), # 交互式场景理解
            ("embodied interaction", 1.8),         # 具身交互
            ("manipulation perception", 1.8),      # 操作感知
            
            # 物理属性感知（高权重）
            ("physical property perception", 2.0), # 物理属性感知
            ("physics perception", 2.0),           # 物理感知
            ("material perception", 1.8),          # 材料感知
            ("dynamics estimation", 1.8),          # 动力学估计
            ("physical understanding", 1.8),       # 物理理解
            
            # 强化学习（高权重）
            ("visual reinforcement learning", 2.0), # 视觉强化学习
            ("vision-based RL", 2.0),              # 基于视觉的强化学习
            ("embodied RL", 2.0),                  # 具身强化学习
            ("visual policy learning", 1.8),       # 视觉策略学习
            ("visual behavior learning", 1.8),     # 视觉行为学习
            
            # 协作系统（高权重）
            ("collaborative vision", 2.0),         # 协作视觉
            ("multi-agent vision", 2.0),           # 多智能体视觉
            ("distributed perception", 1.8),       # 分布式感知
            ("swarm perception", 1.8),             # 群体感知
            ("cooperative visual learning", 1.8),  # 合作视觉学习
        ],
        "negative_keywords": [
            ("passive perception", 1.0),           # 被动感知
            ("static image", 1.0),                 # 静态图像
        ]
    },
    
    "高效计算 (Efficient Computing)": {
        "keywords": [
            # 模型压缩（高权重）
            ("model compression", 2.0),            # 模型压缩
            ("model acceleration", 2.0),           # 模型加速
            ("knowledge distillation", 2.0),       # 知识蒸馏
            ("network pruning", 1.8),              # 网络剪枝
            ("quantization", 1.8),                 # 量化
            
            # 稀疏计算（高权重）
            ("sparse computation", 2.0),           # 稀疏计算
            ("sparse activation", 2.0),            # 稀疏激活
            ("conditional computation", 1.8),      # 条件计算
            ("dynamic execution", 1.8),            # 动态执行
            ("sparse convolution", 1.8),           # 稀疏卷积
            
            # 硬件协同（高权重）
            ("hardware-aware", 2.0),               # 硬件感知
            ("hardware-algorithm co-design", 2.0), # 硬件算法协同设计
            ("FPGA", 1.8),                         # FPGA
            ("ASIC", 1.8),                         # ASIC
            ("accelerator", 1.8),                  # 加速器
            
            # 低功耗系统（高权重）
            ("low-power systems", 2.0),             # 低功耗系统
            ("energy-efficient", 2.0),             # 能效
            ("power optimization", 1.8),           # 功耗优化
            ("energy-aware", 1.8),                 # 能源感知
            ("green AI", 1.8),                     # 绿色AI
            
            # 边缘计算（高权重）
            ("edge computing", 2.0),        # 边缘计算
            ("on-device vision", 2.0),             # 设备上视觉
            ("mobile vision", 2.0),                # 移动视觉
            ("embedded vision", 1.8),              # 嵌入式视觉
            ("IoT vision", 1.8),                   # 物联网视觉
        ],
        "negative_keywords": [
            ("large model", 1.0),                  # 大模型
            ("high computational cost", 1.0),      # 高计算成本
        ]
    },
    
    "可信系统 (Trustworthy Systems)": {
        "keywords": [
            # 鲁棒性（高权重）
            ("robustness", 2.0),                # 鲁棒性
            ("adversarial robustness", 2.0),       # 对抗鲁棒性
            ("out-of-distribution", 2.0),          # 分布外
            ("domain robustness", 1.8),            # 域鲁棒性
            ("robust perception", 1.8),            # 鲁棒感知
            
            # 可解释性（高权重）
            ("explainable vision", 2.0),           # 可解释视觉
            ("interpretable vision", 2.0),         # 可解释视觉
            ("visual explanation", 2.0),           # 视觉解释
            ("attribution method", 1.8),           # 归因方法
            ("saliency map", 1.8),                 # 显著图
            
            # 不确定性（高权重）
            ("uncertainty modeling", 2.0),         # 不确定性建模
            ("uncertainty quantification", 2.0),   # 不确定性量化
            ("bayesian vision", 2.0),              # 贝叶斯视觉
            ("probabilistic vision", 1.8),         # 概率视觉
            ("confidence calibration", 1.8),       # 置信度校准
            
            # 公平与伦理（高权重）
            ("fairness", 2.0),                     # 公平性
            ("ethical vision", 2.0),               # 伦理视觉
            ("bias mitigation", 2.0),              # 偏见缓解
            ("algorithmic fairness", 1.8),         # 算法公平性
            ("responsible vision", 1.8),           # 负责任视觉
            
            # 隐私保护（高权重）
            ("privacy-preserving vision", 2.0),    # 隐私保护视觉
            ("federated learning", 2.0),           # 联邦学习
            ("differential privacy", 2.0),         # 差分隐私
            ("secure vision", 1.8),                # 安全视觉
            ("anonymization", 1.8),                # 匿名化
        ],
        "negative_keywords": [
            ("performance only", 1.0),             # 仅性能
            ("accuracy focused", 1.0),             # 专注于准确性
        ]
    },
    
    # 三、应用方向（后展示）
    "领域应用 (Domain Applications)": {
        "keywords": [
            # 医学影像与分析（高权重）
            ("medical imaging", 2.0),              # 医学影像
            ("healthcare monitoring", 2.0),        # 健康监测
            ("clinical vision", 1.8),              # 临床视觉
            ("biomedical imaging", 1.8),           # 生物医学影像
            
            # 自动驾驶（高权重）
            ("autonomous driving", 2.0),    # 自动驾驶
            ("self-driving perception", 2.0),      # 自动驾驶感知
            ("vehicle perception", 2.0),           # 车辆感知
            ("traffic scene understanding", 1.8),  # 交通场景理解
            ("driving scene analysis", 1.8),       # 驾驶场景分析
            
            # 机器人技术（高权重）
            ("robotic vision", 2.0),               # 机器人视觉
            ("robot perception", 2.0),             # 机器人感知
            ("visual servoing", 2.0),              # 视觉伺服
            ("manipulation vision", 1.8),          # 操作视觉
            ("grasping perception", 1.8),          # 抓取感知
            
            # AR/VR（高权重）
            ("AR vision", 2.0),                    # AR视觉
            ("VR vision", 2.0),                    # VR视觉
            ("mixed reality", 2.0),                # 混合现实
            ("augmented reality perception", 1.8), # 增强现实感知
            ("virtual environment", 1.8),          # 虚拟环境
            
            # 工业检测（高权重）
            ("manufacturing inspection", 2.0),     # 制造检测
            ("quality control vision", 2.0),       # 质量控制视觉
            ("industrial automation", 1.8),        # 工业自动化
            ("defect detection", 1.8),             # 缺陷检测
        ],
        "negative_keywords": [
            ("theoretical", 1.0),                  # 理论
            ("fundamental research", 1.0),         # 基础研究
        ]
    },
    
    # 其他类别
    "其他 (Others)": {
        "keywords": [
            ("miscellaneous", 1.0),                # 杂项
            ("other", 1.0),                        # 其他
        ],
        "negative_keywords": []
    }
}