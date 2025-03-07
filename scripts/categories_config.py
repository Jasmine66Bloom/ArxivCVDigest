"""
计算机视觉论文分类配置文件 (2025前瞻版)
包含一级分类和二级分类的层次结构
"""

# 类别阈值配置（越大越严格）
CATEGORY_THRESHOLDS = {
    # 一、前沿研究方向（优先展示）
    "智能视觉基础": {
        "threshold": 1.2,
        "subcategories": {
            "涌现视觉能力": 1.3,    # 大规模视觉模型中的涌现能力
            "自主视觉系统": 1.3,    # 自主学习和适应的视觉系统
            "神经符号视觉": 1.3,    # 结合神经网络和符号推理的视觉
            "视觉认知计算": 1.3,    # 模拟人类认知的视觉计算
            "大规模视觉预训练": 1.3, # 超大规模视觉基础模型
        },
        "priority": 5,  # 最高优先级
    },
    
    "生成式视觉": {
        "threshold": 1.2,
        "subcategories": {
            "高保真图像生成": 1.3,   # 超高质量图像生成
            "长时视频合成": 1.3,     # 长时间连贯视频生成
            "3D内容生成": 1.3,       # 3D模型和场景生成
            "可控内容创建": 1.3,     # 精确可控的生成内容
            "物理约束生成": 1.3,     # 符合物理规律的生成
        },
        "priority": 4,  # 高优先级
    },
    
    "自适应视觉": {
        "threshold": 1.2,
        "subcategories": {
            "持续学习视觉": 1.3,     # 持续学习和知识积累
            "开放世界理解": 1.3,     # 开放集和未知类别识别
            "少样本视觉学习": 1.3,   # 少样本和零样本学习
            "域适应与泛化": 1.3,     # 跨域适应和泛化
            "自校准视觉系统": 1.3,   # 自动校准和自我修正
        },
        "priority": 4,  # 高优先级
    },
    
    "多模态智能": {
        "threshold": 1.2,
        "subcategories": {
            "视觉-语言大模型": 1.3,  # 视觉语言大规模模型
            "多模态对齐与融合": 1.3, # 跨模态对齐和信息融合
            "跨模态推理": 1.3,       # 利用多模态信息进行推理
            "多模态世界模型": 1.3,   # 多模态环境建模
            "多感官协同": 1.3,       # 多种感知信号协同处理
        },
        "priority": 4,  # 高优先级
    },
    
    # 二、核心技术方向
    "基础视觉任务": {
        "threshold": 1.2,
        "subcategories": {
            "目标检测与识别": 1.3,     # 目标检测、实例识别等
            "图像分割": 1.3,          # 语义分割、实例分割、全景分割等
            "图像分类": 1.3,          # 图像分类、细粒度分类等
            "视频理解": 1.3,          # 视频分析、动作识别、时序建模等
            "图像处理增强": 1.3,      # 超分辨率、去噪、图像恢复等
        },
        "priority": 3,  # 中等优先级
    },
    
    "神经场景理解": {
        "threshold": 1.2,
        "subcategories": {
            "新一代神经辐射场": 1.3, # 高效精确的NeRF变体
            "动态场景表示": 1.3,     # 动态和可变形场景表示
            "隐式几何建模": 1.3,     # 隐式函数表示的几何
            "语义场景分解": 1.3,     # 场景的语义理解和分解
            "时空场景预测": 1.3,     # 场景随时间变化的预测
        },
        "priority": 3,  # 中等优先级
    },
    
    "具身视觉智能": {
        "threshold": 1.2,
        "subcategories": {
            "视觉-运动规划": 1.3,    # 基于视觉的动作规划
            "交互式场景理解": 1.3,   # 通过交互理解场景
            "物理动力学感知": 1.3,   # 物理属性和动力学感知
            "视觉强化学习": 1.3,     # 视觉导向的强化学习
            "多智能体视觉协作": 1.3, # 多智能体系统的视觉协作
        },
        "priority": 3,  # 中等优先级
    },
    
    "高效视觉系统": {
        "threshold": 1.2,
        "subcategories": {
            "视觉模型压缩": 1.3,     # 模型压缩和知识蒸馏
            "稀疏计算框架": 1.3,     # 稀疏激活和计算
            "硬件协同设计": 1.3,     # 算法-硬件协同优化
            "能效优化视觉": 1.3,     # 低能耗视觉算法
            "边缘视觉计算": 1.3,     # 边缘设备上的视觉计算
        },
        "priority": 3,  # 中等优先级
    },
    
    "可信视觉": {
        "threshold": 1.2,
        "subcategories": {
            "鲁棒视觉感知": 1.3,     # 对抗和分布外鲁棒性
            "可解释视觉模型": 1.3,   # 可解释和可理解的模型
            "不确定性量化": 1.3,     # 预测不确定性的量化
            "公平性与伦理": 1.3,     # 公平和伦理的视觉系统
            "隐私保护视觉": 1.3,     # 保护隐私的视觉技术
        },
        "priority": 3,  # 中等优先级
    },
    
    # 三、应用方向（后展示）
    "垂直领域视觉": {
        "threshold": 1.2,
        "subcategories": {
            "医疗健康视觉": 1.3,     # 医学影像和健康监测
            "智能驾驶感知": 1.3,     # 自动驾驶视觉感知
            "机器人视觉": 1.3,       # 机器人视觉感知和控制
            "元宇宙视觉": 1.3,       # 虚拟现实和增强现实
            "科学与工业视觉": 1.3,   # 科学研究和工业应用
        },
        "priority": 2,  # 较低优先级
    },
    
    # 其他类别
    "其他": {
        "threshold": 1.8,           # 提高阈值，更难匹配
        "subcategories": {},
        "priority": 1,  # 最低优先级
    }
}

# 类别显示顺序配置（从上到下）
CATEGORY_DISPLAY_ORDER = [
    # 一、前沿研究方向（优先展示）
    "智能视觉基础",     # 涌现视觉能力、自主视觉系统、神经符号视觉、视觉认知计算、大规模视觉预训练
    "生成式视觉",       # 高保真图像生成、长时视频合成、3D内容生成、可控内容创建、物理约束生成
    "自适应视觉",       # 持续学习视觉、开放世界理解、少样本视觉学习、域适应与泛化、自校准视觉系统
    "多模态智能",       # 视觉-语言大模型、多模态对齐与融合、跨模态推理、多模态世界模型、多感官协同
    
    # 二、核心技术方向
    "基础视觉任务",     # 目标检测与识别、图像分割、图像分类、视频理解、图像处理增强
    "神经场景理解",     # 新一代神经辐射场、动态场景表示、隐式几何建模、语义场景分解、时空场景预测
    "具身视觉智能",     # 视觉-运动规划、交互式场景理解、物理动力学感知、视觉强化学习、多智能体视觉协作
    "高效视觉系统",     # 视觉模型压缩、稀疏计算框架、硬件协同设计、能效优化视觉、边缘视觉计算
    "可信视觉",         # 鲁棒视觉感知、可解释视觉模型、不确定性量化、公平性与伦理、隐私保护视觉
    
    # 三、应用方向（后展示）
    "垂直领域视觉",     # 医疗健康视觉、智能驾驶感知、机器人视觉、元宇宙视觉、科学与工业视觉
    
    # 其他类别（永远放在最后）
    "其他"              # 其他类别
]

# 分类提示词
CATEGORY_PROMPT = """
请将以下计算机视觉论文分类到最合适的类别中。

2025年计算机视觉分类体系：

一、前沿研究方向
1. 智能视觉基础：涌现视觉能力、自主视觉系统、神经符号视觉、视觉认知计算、大规模视觉预训练
   定义：研究视觉智能的基础理论、大规模视觉模型、涌现能力和认知计算等前沿课题。

2. 生成式视觉：高保真图像生成、长时视频合成、3D内容生成、可控内容创建、物理约束生成
   定义：研究生成高质量视觉内容的方法，包括图像、视频和3D内容的生成与编辑。

3. 自适应视觉：持续学习视觉、开放世界理解、少样本视觉学习、域适应与泛化、自校准视觉系统
   定义：研究视觉系统在新环境、新任务中的适应能力，包括持续学习、域适应和少样本学习。

4. 多模态智能：视觉-语言大模型、多模态对齐与融合、跨模态推理、多模态世界模型、多感官协同
   定义：研究视觉与其他模态（如语言、音频）的协同理解和交互，包括视觉-语言模型和跨模态推理。

二、核心技术方向
5. 基础视觉任务：目标检测与识别、图像分割、图像分类、视频理解、图像处理增强
   定义：研究计算机视觉的基础任务和技术，如检测、分割、分类和视频理解等。

6. 神经场景理解：新一代神经辐射场、动态场景表示、隐式几何建模、语义场景分解、时空场景预测
   定义：研究基于神经网络的场景表示和理解方法，包括NeRF、3D高斯散射和隐式几何等。

7. 具身视觉智能：视觉-运动规划、交互式场景理解、物理动力学感知、视觉强化学习、多智能体视觉协作
   定义：研究视觉系统与物理世界交互的能力，包括视觉引导的动作规划和物理感知等。

8. 高效视觉系统：视觉模型压缩、稀疏计算框架、硬件协同设计、能效优化视觉、边缘视觉计算
   定义：研究提高视觉系统效率的方法，包括模型压缩、硬件优化和边缘计算等。

9. 可信视觉：鲁棒视觉感知、可解释视觉模型、不确定性量化、公平性与伦理、隐私保护视觉
   定义：研究视觉系统的可靠性、安全性和伦理问题，包括鲁棒性、可解释性和隐私保护等。

三、应用方向
10. 垂直领域视觉：医疗健康视觉、智能驾驶感知、机器人视觉、元宇宙视觉、科学与工业视觉
    定义：研究视觉技术在特定领域的应用，如医疗、自动驾驶、机器人和工业等。

11. 其他：不属于以上类别的论文
    定义：其他不能明确归类到上述类别的视觉研究工作。

分类指南：
1. 首先分析论文的核心技术贡献和主要研究目标
2. 考虑论文的方法、实验和应用场景
3. 如果论文涉及多个类别，请选择最核心、最具创新性的方向
4. 优先考虑技术本质而非应用领域（除非应用创新是论文的主要贡献）
5. 只有在确实无法归类到前10个类别时，才选择"其他"类别

边界案例处理：
- 如果论文同时涉及"生成式视觉"和"多模态智能"，但核心是文本引导的图像生成，应归类为"生成式视觉"
- 如果论文研究NeRF技术，即使应用于医疗领域，也应优先归类为"神经场景理解"而非"垂直领域视觉"
- 如果论文提出新的视觉基础模型并展示了涌现能力，应归类为"智能视觉基础"而非"基础视觉任务"

请分析论文的核心技术和主要贡献，选择最合适的一个类别。只返回类别名称，不要有任何解释或额外文本。
"""

# 类别关键词配置
CATEGORY_KEYWORDS = {
    # 一、前沿研究方向（优先展示）
    "智能视觉基础": {
        "keywords": [
            # 涌现视觉能力（高权重）
            ("emergent visual", 2.0),              # 涌现视觉
            ("emergent ability", 1.8),             # 涌现能力
            ("emergent property", 1.8),            # 涌现特性
            ("scaling law", 1.8),                  # 缩放定律
            ("visual reasoning", 1.8),             # 视觉推理
            
            # 自主视觉系统（高权重）
            ("autonomous vision", 2.0),            # 自主视觉
            ("self-improving", 1.8),               # 自我提升
            ("autonomous visual system", 2.0),     # 自主视觉系统
            ("self-supervised vision", 1.8),       # 自监督视觉
            ("autonomous adaptation", 1.8),        # 自主适应
            
            # 神经符号视觉（高权重）
            ("neuro-symbolic", 2.0),               # 神经符号
            ("neural-symbolic vision", 2.0),       # 神经符号视觉
            ("symbolic reasoning", 1.8),           # 符号推理
            ("visual concept", 1.8),               # 视觉概念
            ("concept learning", 1.8),             # 概念学习
            
            # 视觉认知计算（高权重）
            ("cognitive vision", 2.0),             # 认知视觉
            ("cognitive computing", 1.8),          # 认知计算
            ("brain-inspired vision", 1.8),        # 脑启发视觉
            ("visual cognition", 1.8),             # 视觉认知
            ("cognitive architecture", 1.8),       # 认知架构
            
            # 大规模视觉预训练（高权重）
            ("large vision model", 2.0),           # 大型视觉模型
            ("foundation vision model", 2.0),      # 基础视觉模型
            ("vision foundation model", 2.0),      # 视觉基础模型
            ("large-scale pretraining", 1.8),      # 大规模预训练
            ("billion-parameter", 1.8),            # 十亿参数
            ("trillion-parameter", 2.0),           # 万亿参数
            ("visual pretraining", 1.8),           # 视觉预训练
            
            # 通用视觉模型（中等权重）
            ("general vision model", 1.6),         # 通用视觉模型
            ("universal visual representation", 1.6), # 通用视觉表示
            ("segment anything", 1.6),             # 分割一切
            ("detect anything", 1.6),              # 检测一切
            ("understand anything", 1.6),          # 理解一切
            
            # 视觉架构（中等权重）
            ("vision transformer", 1.5),           # 视觉Transformer
            ("hierarchical vision", 1.5),          # 层次视觉
            ("scalable architecture", 1.5),        # 可扩展架构
            ("efficient attention", 1.5),          # 高效注意力
            ("state space model", 1.6),            # 状态空间模型
            ("mamba vision", 1.6),                 # Mamba视觉
        ],
        "negative_keywords": [
            # 避免与其他类别混淆
            ("medical image", -1.0),               # 医学图像（属于垂直领域）
            ("autonomous driving", -1.0),          # 自动驾驶（属于垂直领域）
            ("video generation", -1.0),            # 视频生成（属于生成式视觉）
            ("image generation", -1.0),            # 图像生成（属于生成式视觉）
            ("3d generation", -1.0),               # 3D生成（属于生成式视觉）
        ],
        "priority": 5,  # 最高优先级
    },
    
    "生成式视觉": {
        "keywords": [
            # 高保真图像生成（高权重）
            ("high-fidelity image generation", 2.0),  # 高保真图像生成
            ("photorealistic image", 1.8),           # 逼真图像
            ("diffusion model", 1.8),                # 扩散模型
            ("stable diffusion", 1.8),               # Stable Diffusion
            ("text-to-image", 1.8),                  # 文本到图像
            ("image synthesis", 1.8),                # 图像合成
            ("generative adversarial", 1.8),         # 生成对抗
            ("gan", 1.6),                            # GAN
            
            # 长时视频合成（高权重）
            ("video generation", 2.0),               # 视频生成
            ("long video synthesis", 2.0),           # 长视频合成
            ("text-to-video", 1.8),                  # 文本到视频
            ("video diffusion", 1.8),                # 视频扩散
            ("consistent video", 1.8),               # 一致性视频
            ("temporal consistency", 1.8),           # 时序一致性
            ("long-form video", 1.8),                # 长格式视频
            
            # 3D内容生成（高权重）
            ("3d content generation", 2.0),          # 3D内容生成
            ("text-to-3d", 1.8),                     # 文本到3D
            ("3d generation", 1.8),                  # 3D生成
            ("3d diffusion", 1.8),                   # 3D扩散
            ("3d synthesis", 1.8),                   # 3D合成
            ("dreamfusion", 1.6),                    # DreamFusion
            ("3d gan", 1.6),                         # 3D GAN
            
            # 可控内容创建（高权重）
            ("controllable generation", 2.0),        # 可控生成
            ("controlled synthesis", 1.8),           # 受控合成
            ("image editing", 1.8),                  # 图像编辑
            ("controlnet", 1.8),                     # ControlNet
            ("inpainting", 1.6),                     # 图像修复
            ("outpainting", 1.6),                    # 图像扩展
            ("style transfer", 1.6),                 # 风格迁移
            
            # 物理约束生成（高权重）
            ("physics-constrained generation", 2.0), # 物理约束生成
            ("physically plausible", 1.8),           # 物理上合理
            ("physics-guided", 1.8),                 # 物理引导
            ("physics-based generation", 1.8),       # 基于物理的生成
            ("physical consistency", 1.8),           # 物理一致性
            ("physically accurate", 1.8),            # 物理准确
            
            # 生成模型架构（中等权重）
            ("latent diffusion", 1.6),               # 潜在扩散
            ("flow matching", 1.6),                  # 流匹配
            ("score distillation", 1.6),             # 分数蒸馏
            ("consistency model", 1.6),              # 一致性模型
            ("generative prior", 1.6),               # 生成先验
            
            # 生成评估（中等权重）
            ("generation quality", 1.5),             # 生成质量
            ("fidelity metric", 1.5),                # 保真度指标
            ("perceptual quality", 1.5),             # 感知质量
            ("generation diversity", 1.5),           # 生成多样性
            ("generation evaluation", 1.5),          # 生成评估
        ],
        "negative_keywords": [
            # 避免与其他类别混淆
            ("recognition", -1.0),                   # 识别（属于视觉识别）
            ("detection", -1.0),                     # 检测（属于视觉识别）
            ("segmentation", -1.0),                  # 分割（属于视觉识别）
            ("classification", -1.0),                # 分类（属于视觉识别）
            ("tracking", -1.0),                      # 跟踪（属于视频理解）
            ("pose estimation", -1.0),               # 姿态估计（属于人体分析）
            ("medical image", -1.0),                 # 医学图像（属于垂直领域）
        ],
        "priority": 4,  # 高优先级
    },
    
    "自适应视觉": {
        "keywords": [
            # 持续学习视觉（高权重）
            ("continual learning", 2.0),             # 持续学习
            ("lifelong learning", 1.8),              # 终身学习
            ("incremental learning", 1.8),           # 增量学习
            ("knowledge accumulation", 1.8),         # 知识积累
            ("catastrophic forgetting", 1.8),        # 灾难性遗忘
            ("memory replay", 1.6),                  # 记忆重放
            
            # 开放世界理解（高权重）
            ("open world recognition", 2.0),         # 开放世界识别
            ("open set recognition", 1.8),           # 开集识别
            ("unknown detection", 1.8),              # 未知检测
            ("novel class discovery", 1.8),          # 新类发现
            ("out-of-distribution", 1.8),            # 分布外
            ("open vocabulary", 1.8),                # 开放词汇
            
            # 少样本视觉学习（高权重）
            ("few-shot learning", 2.0),              # 少样本学习
            ("zero-shot learning", 1.8),             # 零样本学习
            ("one-shot learning", 1.8),              # 单样本学习
            ("meta-learning", 1.8),                  # 元学习
            ("prompt learning", 1.8),                # 提示学习
            ("in-context learning", 1.8),            # 上下文学习
            
            # 域适应与泛化（高权重）
            ("domain adaptation", 2.0),              # 域适应
            ("domain generalization", 1.8),          # 域泛化
            ("transfer learning", 1.8),              # 迁移学习
            ("cross-domain", 1.8),                   # 跨域
            ("style transfer learning", 1.6),        # 风格迁移学习
            ("unsupervised adaptation", 1.6),        # 无监督适应
            
            # 自校准视觉系统（高权重）
            ("self-calibration", 2.0),               # 自校准
            ("self-correction", 1.8),                # 自修正
            ("self-supervised calibration", 1.8),    # 自监督校准
            ("adaptive calibration", 1.8),           # 自适应校准
            ("uncertainty calibration", 1.8),        # 不确定性校准
            ("confidence calibration", 1.8),         # 置信度校准
            
            # 自适应学习（中等权重）
            ("adaptive learning", 1.6),              # 自适应学习
            ("curriculum learning", 1.6),            # 课程学习
            ("active learning", 1.6),                # 主动学习
            ("self-paced learning", 1.6),            # 自步调学习
            ("online learning", 1.6),                # 在线学习
            
            # 长尾分布学习（中等权重）
            ("long-tail learning", 1.6),             # 长尾学习
            ("imbalanced learning", 1.6),            # 不平衡学习
            ("class imbalance", 1.6),                # 类别不平衡
            ("rare category", 1.6),                  # 稀有类别
            ("minority class", 1.6),                 # 少数类
        ],
        "negative_keywords": [
            # 避免与其他类别混淆
            ("image generation", -1.0),              # 图像生成（属于生成式视觉）
            ("video generation", -1.0),              # 视频生成（属于生成式视觉）
            ("3d generation", -1.0),                 # 3D生成（属于生成式视觉）
            ("multimodal", -1.0),                    # 多模态（属于多模态智能）
            ("vision-language", -1.0),               # 视觉-语言（属于多模态智能）
            ("embodied", -1.0),                      # 具身（属于具身视觉智能）
            ("robot", -1.0),                         # 机器人（属于具身视觉智能）
        ],
        "priority": 4,  # 高优先级
    },
    
    "多模态智能": {
        "keywords": [
            # 视觉-语言大模型（高权重）
            ("vision-language model", 2.0),          # 视觉-语言模型
            ("multimodal large language model", 2.0), # 多模态大语言模型
            ("visual language model", 2.0),          # 视觉语言模型
            ("multimodal llm", 1.8),                 # 多模态LLM
            ("mllm", 1.8),                           # MLLM
            ("llava", 1.8),                          # LLaVA
            ("minigpt", 1.8),                        # MiniGPT
            ("visual instruction tuning", 1.8),      # 视觉指令微调
            
            # 多模态对齐与融合（高权重）
            ("multimodal alignment", 2.0),           # 多模态对齐
            ("cross-modal alignment", 1.8),          # 跨模态对齐
            ("modality fusion", 1.8),                # 模态融合
            ("multimodal fusion", 1.8),              # 多模态融合
            ("joint embedding", 1.8),                # 联合嵌入
            ("clip", 1.8),                           # CLIP
            ("align", 1.8),                          # ALIGN
            
            # 跨模态推理（高权重）
            ("cross-modal reasoning", 2.0),          # 跨模态推理
            ("multimodal reasoning", 1.8),           # 多模态推理
            ("visual reasoning", 1.8),               # 视觉推理
            ("multimodal inference", 1.8),           # 多模态推断
            ("visual question answering", 1.8),      # 视觉问答
            ("vqa", 1.8),                            # VQA
            ("visual dialog", 1.8),                  # 视觉对话
            
            # 多模态世界模型（高权重）
            ("multimodal world model", 2.0),         # 多模态世界模型
            ("multimodal environment model", 1.8),   # 多模态环境模型
            ("visual world model", 1.8),             # 视觉世界模型
            ("multimodal predictive model", 1.8),    # 多模态预测模型
            ("multimodal simulation", 1.8),          # 多模态模拟
            ("multimodal dynamics", 1.8),            # 多模态动力学
            
            # 多感官协同（高权重）
            ("multi-sensory", 2.0),                  # 多感官
            ("audio-visual", 1.8),                   # 视听
            ("tactile-visual", 1.8),                 # 触视
            ("cross-sensory", 1.8),                  # 跨感官
            ("sensory fusion", 1.8),                 # 感官融合
            ("multimodal perception", 1.8),          # 多模态感知
            
            # 多模态理解（中等权重）
            ("multimodal understanding", 1.6),       # 多模态理解
            ("cross-modal understanding", 1.6),      # 跨模态理解
            ("multimodal comprehension", 1.6),       # 多模态理解
            ("multimodal representation", 1.6),      # 多模态表示
            ("multimodal embedding", 1.6),           # 多模态嵌入
            
            # 多模态生成（中等权重）
            ("multimodal generation", 1.6),          # 多模态生成
            ("cross-modal generation", 1.6),         # 跨模态生成
            ("multimodal synthesis", 1.6),           # 多模态合成
            ("text-to-image-to-text", 1.6),          # 文本到图像到文本
            ("image captioning", 1.6),               # 图像描述
        ],
        "negative_keywords": [
            # 避免与其他类别混淆
            ("image generation", -1.0),              # 图像生成（属于生成式视觉）
            ("video generation", -1.0),              # 视频生成（属于生成式视觉）
            ("3d generation", -1.0),                 # 3D生成（属于生成式视觉）
            ("embodied", -1.0),                      # 具身（属于具身视觉智能）
            ("robot", -1.0),                         # 机器人（属于具身视觉智能）
            ("manipulation", -1.0),                  # 操作（属于具身视觉智能）
            ("medical image", -1.0),                 # 医学图像（属于垂直领域）
        ],
        "priority": 4,  # 高优先级
    },
    
    # 二、核心技术方向
    "基础视觉任务": {
        "keywords": [
            # 目标检测与识别（高权重）
            ("object detection", 2.0),                # 目标检测
            ("detection transformer", 1.8),           # 检测Transformer
            ("yolo", 1.8),                            # YOLO
            ("faster rcnn", 1.8),                     # Faster R-CNN
            ("object recognition", 1.8),              # 目标识别
            ("detection model", 1.8),                 # 检测模型
            ("anchor based", 1.6),                    # 基于锚框
            ("anchor free", 1.6),                     # 无锚框
            ("proposal generation", 1.6),             # 候选框生成
            ("bounding box", 1.6),                    # 边界框
            
            # 图像分割（高权重）
            ("image segmentation", 2.0),              # 图像分割
            ("semantic segmentation", 1.8),           # 语义分割
            ("instance segmentation", 1.8),           # 实例分割
            ("panoptic segmentation", 1.8),           # 全景分割
            ("mask2former", 1.6),                     # Mask2Former
            ("segment anything", 1.8),                # SAM
            ("sam", 1.8),                             # SAM
            ("oneformer", 1.6),                       # OneFormer
            ("segmentation model", 1.8),              # 分割模型
            
            # 图像分类（高权重）
            ("image classification", 2.0),            # 图像分类
            ("visual recognition", 1.8),              # 视觉识别
            ("image recognition", 1.8),               # 图像识别
            ("fine-grained classification", 1.8),     # 细粒度分类
            ("classification model", 1.8),            # 分类模型
            ("convnext", 1.6),                        # ConvNeXt
            ("efficientnet", 1.6),                    # EfficientNet
            ("vision transformer", 1.6),              # Vision Transformer
            
            # 视频理解（高权重）
            ("video understanding", 2.0),             # 视频理解
            ("action recognition", 1.8),              # 动作识别
            ("activity recognition", 1.8),            # 活动识别
            ("temporal modeling", 1.8),               # 时序建模
            ("video classification", 1.8),            # 视频分类
            ("video object tracking", 1.8),           # 视频目标跟踪
            ("multi-object tracking", 1.8),           # 多目标跟踪
            ("spatio-temporal", 1.6),                 # 时空
            ("video transformer", 1.6),               # 视频Transformer
            
            # 图像处理增强（高权重）
            ("image enhancement", 2.0),               # 图像增强
            ("super resolution", 1.8),                # 超分辨率
            ("image restoration", 1.8),               # 图像恢复
            ("image denoising", 1.8),                 # 图像去噪
            ("image deblurring", 1.8),                # 图像去模糊
            ("image inpainting", 1.8),                # 图像修复
            ("low-light enhancement", 1.8),           # 低光照增强
            ("image quality", 1.6),                   # 图像质量
            ("image compression", 1.6),               # 图像压缩
        ],
        "negative_keywords": [
            # 避免与其他类别混淆
            ("image generation", -1.0),               # 图像生成（属于生成式视觉）
            ("video generation", -1.0),               # 视频生成（属于生成式视觉）
            ("text-to-image", -1.0),                  # 文本到图像（属于生成式视觉）
            ("nerf", -1.0),                           # NeRF（属于神经场景理解）
            ("gaussian splatting", -1.0),             # 高斯散射（属于神经场景理解）
            ("multimodal", -1.0),                     # 多模态（属于多模态智能）
            ("vision-language", -1.0),                # 视觉-语言（属于多模态智能）
            ("medical image", -1.0),                  # 医学图像（属于垂直领域）
        ],
        "priority": 3,  # 中等优先级
    },
    
    "神经场景理解": {
        "keywords": [
            # 新一代神经辐射场（高权重）
            ("neural radiance field", 2.0),          # 神经辐射场
            ("nerf", 2.0),                           # NeRF
            ("instant-ngp", 1.8),                    # Instant-NGP
            ("mip-nerf", 1.8),                       # Mip-NeRF
            ("neus", 1.8),                           # NeuS
            ("3d gaussian splatting", 2.0),          # 3D高斯散射
            ("gaussian splatting", 2.0),             # 高斯散射
            ("3d-gs", 1.8),                          # 3D-GS
            
            # 动态场景表示（高权重）
            ("dynamic scene representation", 2.0),    # 动态场景表示
            ("dynamic nerf", 1.8),                   # 动态NeRF
            ("deformable nerf", 1.8),                # 可变形NeRF
            ("4d neural field", 1.8),                # 4D神经场
            ("neural scene flow", 1.8),              # 神经场景流
            ("dynamic 3d reconstruction", 1.8),      # 动态3D重建
            
            # 隐式几何建模（高权重）
            ("implicit geometric representation", 2.0), # 隐式几何表示
            ("implicit neural representation", 1.8),  # 隐式神经表示
            ("neural implicit surface", 1.8),         # 神经隐式表面
            ("signed distance function", 1.8),        # 符号距离函数
            ("sdf", 1.8),                            # SDF
            ("occupancy field", 1.8),                # 占用场
            
            # 语义场景分解（高权重）
            ("semantic scene decomposition", 2.0),    # 语义场景分解
            ("neural scene decomposition", 1.8),      # 神经场景分解
            ("object-centric scene", 1.8),           # 以物体为中心的场景
            ("compositional scene", 1.8),            # 组合式场景
            ("scene factorization", 1.8),            # 场景分解
            ("neural scene graph", 1.8),             # 神经场景图
            
            # 时空场景预测（高权重）
            ("spatio-temporal prediction", 2.0),      # 时空预测
            ("future scene prediction", 1.8),         # 未来场景预测
            ("neural scene forecasting", 1.8),        # 神经场景预测
            ("4d scene prediction", 1.8),            # 4D场景预测
            ("dynamic scene understanding", 1.8),     # 动态场景理解
            ("temporal scene modeling", 1.8),         # 时序场景建模
            
            # 新视角合成（中等权重）
            ("novel view synthesis", 1.6),           # 新视角合成
            ("view synthesis", 1.6),                 # 视角合成
            ("free viewpoint", 1.6),                 # 自由视角
            ("multi-view reconstruction", 1.6),      # 多视图重建
            ("neural rendering", 1.6),               # 神经渲染
            
            # 3D重建（中等权重）
            ("3d reconstruction", 1.6),              # 3D重建
            ("structure from motion", 1.6),          # 运动恢复结构
            ("multi-view stereo", 1.6),              # 多视图立体视觉
            ("visual slam", 1.6),                    # 视觉SLAM
            ("point cloud reconstruction", 1.6),     # 点云重建
        ],
        "negative_keywords": [
            # 避免与其他类别混淆
            ("image generation", -1.0),              # 图像生成（属于生成式视觉）
            ("video generation", -1.0),              # 视频生成（属于生成式视觉）
            ("text-to-3d", -1.0),                    # 文本到3D（属于生成式视觉）
            ("multimodal", -1.0),                    # 多模态（属于多模态智能）
            ("vision-language", -1.0),               # 视觉-语言（属于多模态智能）
            ("embodied", -1.0),                      # 具身（属于具身视觉智能）
            ("robot", -1.0),                         # 机器人（属于具身视觉智能）
        ],
        "priority": 3,  # 中等优先级
    },
    
    "具身视觉智能": {
        "keywords": [
            # 视觉-运动规划（高权重）
            ("vision-based planning", 2.0),           # 基于视觉的规划
            ("visual planning", 2.0),                 # 视觉规划
            ("vision-based manipulation", 1.8),       # 基于视觉的操作
            ("visual action planning", 1.8),          # 视觉动作规划
            ("visual trajectory planning", 1.8),      # 视觉轨迹规划
            ("rt-1", 1.8),                            # RT-1
            ("rt-2", 1.8),                            # RT-2
            
            # 交互式场景理解（高权重）
            ("interactive scene understanding", 2.0),  # 交互式场景理解
            ("interaction-based perception", 1.8),     # 基于交互的感知
            ("active scene exploration", 1.8),         # 主动场景探索
            ("interactive perception", 1.8),           # 交互式感知
            ("hand-object interaction", 1.8),          # 手物交互
            ("human-object interaction", 1.8),         # 人物交互
            
            # 物理动力学感知（高权重）
            ("physics dynamics perception", 2.0),      # 物理动力学感知
            ("physical property estimation", 1.8),     # 物理属性估计
            ("visual physics", 1.8),                   # 视觉物理
            ("physical scene understanding", 1.8),     # 物理场景理解
            ("visual dynamics prediction", 1.8),       # 视觉动力学预测
            ("affordance learning", 1.8),              # 可供性学习
            
            # 视觉强化学习（高权重）
            ("visual reinforcement learning", 2.0),    # 视觉强化学习
            ("vision-based reinforcement learning", 1.8), # 基于视觉的强化学习
            ("visual rl", 1.8),                        # 视觉RL
            ("image-based rl", 1.8),                   # 基于图像的RL
            ("visual representation for rl", 1.8),      # 用于RL的视觉表示
            ("world model", 1.8),                      # 世界模型
            
            # 多智能体视觉协作（高权重）
            ("multi-agent visual collaboration", 2.0), # 多智能体视觉协作
            ("collaborative perception", 1.8),         # 协作感知
            ("distributed visual perception", 1.8),    # 分布式视觉感知
            ("swarm perception", 1.8),                 # 群体感知
            ("multi-robot vision", 1.8),               # 多机器人视觉
            ("cooperative visual learning", 1.8),      # 协作视觉学习
            
            # 机器人视觉（中等权重）
            ("robot vision", 1.6),                     # 机器人视觉
            ("robotic perception", 1.6),               # 机器人感知
            ("robotic manipulation", 1.6),             # 机器人操作
            ("robot learning", 1.6),                   # 机器人学习
            ("visual servoing", 1.6),                  # 视觉伺服
            ("hand-eye coordination", 1.6),            # 手眼协调
            
            # 视觉导航（中等权重）
            ("visual navigation", 1.6),                # 视觉导航
            ("vision-based navigation", 1.6),          # 基于视觉的导航
            ("vision-language navigation", 1.6),       # 视觉语言导航
            ("embodied navigation", 1.6),              # 具身导航
            ("visual localization", 1.6),              # 视觉定位
            ("visual exploration", 1.6),               # 视觉探索
        ],
        "negative_keywords": [
            # 避免与其他类别混淆
            ("image generation", -1.0),                # 图像生成（属于生成式视觉）
            ("video generation", -1.0),                # 视频生成（属于生成式视觉）
            ("3d generation", -1.0),                   # 3D生成（属于生成式视觉）
            ("nerf", -1.0),                            # NeRF（属于神经场景理解）
            ("gaussian splatting", -1.0),              # 高斯散射（属于神经场景理解）
            ("medical image", -1.0),                   # 医学图像（属于垂直领域）
            ("autonomous driving", -1.0),              # 自动驾驶（属于垂直领域）
        ],
        "priority": 3,  # 中等优先级
    },
    
    "高效视觉系统": {
        "keywords": [
            # 视觉模型压缩（高权重）
            ("model compression", 2.0),                # 模型压缩
            ("network pruning", 1.8),                 # 网络剪枝
            ("model quantization", 1.8),              # 模型量化
            ("knowledge distillation", 1.8),          # 知识蒸馏
            ("parameter efficient", 1.8),             # 参数高效
            ("model distillation", 1.8),              # 模型蒸馏
            
            # 稀疏计算框架（高权重）
            ("sparse computation", 2.0),              # 稀疏计算
            ("sparse activation", 1.8),               # 稀疏激活
            ("sparse attention", 1.8),                # 稀疏注意力
            ("sparse convolution", 1.8),              # 稀疏卷积
            ("dynamic computation", 1.8),             # 动态计算
            ("conditional computation", 1.8),         # 条件计算
            
            # 硬件协同设计（高权重）
            ("hardware-aware design", 2.0),           # 硬件感知设计
            ("hardware-software co-design", 1.8),     # 硬件软件协同设计
            ("accelerator design", 1.8),              # 加速器设计
            ("fpga implementation", 1.8),             # FPGA实现
            ("asic design", 1.8),                     # ASIC设计
            ("hardware optimization", 1.8),           # 硬件优化
            
            # 能效优化视觉（高权重）
            ("energy efficient vision", 2.0),         # 能效优化视觉
            ("low-power vision", 1.8),                # 低功耗视觉
            ("energy-aware", 1.8),                    # 能源感知
            ("green ai", 1.8),                        # 绿色AI
            ("sustainable vision", 1.8),              # 可持续视觉
            ("power efficient", 1.8),                 # 功率高效
            
            # 边缘视觉计算（高权重）
            ("edge vision computing", 2.0),           # 边缘视觉计算
            ("on-device vision", 1.8),                # 设备上视觉
            ("mobile vision", 1.8),                   # 移动视觉
            ("embedded vision", 1.8),                 # 嵌入式视觉
            ("tinyml vision", 1.8),                   # 微型机器学习视觉
            ("lightweight vision", 1.8),              # 轻量级视觉
            
            # 模型加速（中等权重）
            ("inference acceleration", 1.6),          # 推理加速
            ("model acceleration", 1.6),              # 模型加速
            ("efficient inference", 1.6),             # 高效推理
            ("fast inference", 1.6),                  # 快速推理
            ("real-time vision", 1.6),                # 实时视觉
            
            # 神经架构搜索（中等权重）
            ("neural architecture search", 1.6),      # 神经架构搜索
            ("nas", 1.6),                             # NAS
            ("efficient architecture", 1.6),          # 高效架构
            ("architecture optimization", 1.6),       # 架构优化
            ("automated design", 1.6),                # 自动化设计
        ],
        "negative_keywords": [
            # 避免与其他类别混淆
            ("image generation", -1.0),               # 图像生成（属于生成式视觉）
            ("video generation", -1.0),               # 视频生成（属于生成式视觉）
            ("3d generation", -1.0),                  # 3D生成（属于生成式视觉）
            ("nerf", -1.0),                           # NeRF（属于神经场景理解）
            ("gaussian splatting", -1.0),              # 高斯散射（属于神经场景理解）
            ("robot", -1.0),                          # 机器人（属于具身视觉智能）
            ("embodied", -1.0),                       # 具身（属于具身视觉智能）
        ],
        "priority": 3,  # 中等优先级
    },
    
    "可信视觉": {
        "keywords": [
            # 鲁棒视觉感知（高权重）
            ("robust vision", 2.0),                   # 鲁棒视觉
            ("adversarial robustness", 1.8),          # 对抗鲁棒性
            ("out-of-distribution robustness", 1.8),  # 分布外鲁棒性
            ("distribution shift", 1.8),              # 分布偏移
            ("domain robustness", 1.8),               # 域鲁棒性
            ("adversarial attack", 1.8),              # 对抗攻击
            ("adversarial defense", 1.8),             # 对抗防御
            
            # 可解释视觉模型（高权重）
            ("explainable vision", 2.0),              # 可解释视觉
            ("interpretable vision", 1.8),            # 可解释视觉
            ("visual explanation", 1.8),              # 视觉解释
            ("model interpretation", 1.8),            # 模型解释
            ("attribution method", 1.8),              # 归因方法
            ("concept-based explanation", 1.8),       # 基于概念的解释
            
            # 不确定性量化（高权重）
            ("uncertainty quantification", 2.0),      # 不确定性量化
            ("uncertainty estimation", 1.8),          # 不确定性估计
            ("predictive uncertainty", 1.8),          # 预测不确定性
            ("bayesian deep learning", 1.8),          # 贝叶斯深度学习
            ("ensemble uncertainty", 1.8),            # 集成不确定性
            ("calibration", 1.8),                     # 校准
            
            # 公平性与伦理（高权重）
            ("fairness in vision", 2.0),              # 视觉中的公平性
            ("ethical vision", 1.8),                  # 伦理视觉
            ("bias mitigation", 1.8),                 # 偏见缓解
            ("algorithmic fairness", 1.8),            # 算法公平性
            ("responsible vision", 1.8),              # 负责任的视觉
            ("fair representation", 1.8),             # 公平表示
            
            # 隐私保护视觉（高权重）
            ("privacy-preserving vision", 2.0),       # 隐私保护视觉
            ("federated learning vision", 1.8),       # 联邦学习视觉
            ("differential privacy", 1.8),            # 差分隐私
            ("anonymization", 1.8),                   # 匿名化
            ("secure visual computing", 1.8),         # 安全视觉计算
            ("encrypted inference", 1.8),             # 加密推理
            
            # 安全视觉（中等权重）
            ("secure vision", 1.6),                   # 安全视觉
            ("deepfake detection", 1.6),              # 深度伪造检测
            ("visual forensics", 1.6),                # 视觉取证
            ("manipulation detection", 1.6),          # 篡改检测
            ("content authenticity", 1.6),            # 内容真实性
            
            # 可验证视觉（中等权重）
            ("verifiable vision", 1.6),               # 可验证视觉
            ("certified robustness", 1.6),            # 认证鲁棒性
            ("formal verification", 1.6),             # 形式化验证
            ("safety guarantee", 1.6),                # 安全保证
            ("provable defense", 1.6),                # 可证明防御
        ],
        "negative_keywords": [
            # 避免与其他类别混淆
            ("image generation", -1.0),               # 图像生成（属于生成式视觉）
            ("video generation", -1.0),               # 视频生成（属于生成式视觉）
            ("3d generation", -1.0),                  # 3D生成（属于生成式视觉）
            ("nerf", -1.0),                           # NeRF（属于神经场景理解）
            ("gaussian splatting", -1.0),              # 高斯散射（属于神经场景理解）
            ("robot", -1.0),                          # 机器人（属于具身视觉智能）
            ("embodied", -1.0),                       # 具身（属于具身视觉智能）
        ],
        "priority": 3,  # 中等优先级
    },
    
    # 三、应用方向（后展示）
    "垂直领域视觉": {
        "keywords": [
            # 医疗健康视觉（高权重）
            ("medical vision", 2.0),                  # 医疗视觉
            ("healthcare vision", 2.0),               # 医疗保健视觉
            ("medical image", 1.8),                   # 医学图像
            ("medical imaging", 1.8),                 # 医学成像
            ("healthcare imaging", 1.8),              # 医疗保健成像
            ("clinical vision", 1.8),                 # 临床视觉
            ("pathology image", 1.8),                 # 病理图像
            ("radiology", 1.8),                       # 放射学
            ("ct scan", 1.8),                         # CT扫描
            ("mri", 1.8),                             # MRI
            ("ultrasound", 1.8),                      # 超声波
            ("x-ray", 1.8),                           # X光
            
            # 智能驾驶感知（高权重）
            ("autonomous driving perception", 2.0),   # 自动驾驶感知
            ("self-driving perception", 2.0),         # 自动驾驶感知
            ("autonomous vehicle", 1.8),              # 自动驾驶车辆
            ("driving scene understanding", 1.8),     # 驾驶场景理解
            ("traffic scene", 1.8),                   # 交通场景
            ("vehicle detection", 1.8),               # 车辆检测
            ("lane detection", 1.8),                  # 车道检测
            ("road segmentation", 1.8),               # 道路分割
            
            # 机器人视觉（高权重）
            ("robot vision", 2.0),                    # 机器人视觉
            ("robotic vision", 2.0),                  # 机器人视觉
            ("industrial robot vision", 1.8),         # 工业机器人视觉
            ("service robot vision", 1.8),            # 服务机器人视觉
            ("drone vision", 1.8),                    # 无人机视觉
            ("uav vision", 1.8),                      # 无人机视觉
            ("agricultural robot", 1.8),              # 农业机器人
            
            # 元宇宙视觉（高权重）
            ("metaverse vision", 2.0),                # 元宇宙视觉
            ("augmented reality vision", 1.8),        # 增强现实视觉
            ("virtual reality vision", 1.8),          # 虚拟现实视觉
            ("mixed reality", 1.8),                   # 混合现实
            ("ar vision", 1.8),                       # AR视觉
            ("vr vision", 1.8),                       # VR视觉
            ("xr vision", 1.8),                       # XR视觉
            
            # 科学与工业视觉（高权重）
            ("scientific vision", 2.0),               # 科学视觉
            ("industrial vision", 2.0),               # 工业视觉
            ("manufacturing vision", 1.8),            # 制造业视觉
            ("quality inspection", 1.8),              # 质量检测
            ("defect detection", 1.8),                # 缺陷检测
            ("satellite vision", 1.8),                # 卫星视觉
            ("remote sensing", 1.8),                  # 遥感
            ("microscopy vision", 1.8),               # 显微视觉
            
            # 特定领域应用（中等权重）
            ("retail vision", 1.6),                   # 零售视觉
            ("security vision", 1.6),                 # 安防视觉
            ("surveillance vision", 1.6),             # 监控视觉
            ("sports vision", 1.6),                   # 体育视觉
            ("agricultural vision", 1.6),             # 农业视觉
            ("fashion vision", 1.6),                  # 时尚视觉
            ("education vision", 1.6),                # 教育视觉
        ],
        "negative_keywords": [
            # 避免与其他类别混淆
            ("image generation", -1.0),               # 图像生成（属于生成式视觉）
            ("video generation", -1.0),               # 视频生成（属于生成式视觉）
            ("3d generation", -1.0),                  # 3D生成（属于生成式视觉）
            ("nerf", -1.0),                           # NeRF（属于神经场景理解）
            ("gaussian splatting", -1.0),             # 高斯散射（属于神经场景理解）
        ],
        "priority": 2,  # 较低优先级
    },
    
    # 其他类别
    "其他": {
        "keywords": [
            # 通用关键词（低权重）
            ("computer vision", 1.0),                 # 计算机视觉
            ("deep learning", 1.0),                   # 深度学习
            ("neural network", 1.0),                  # 神经网络
            ("artificial intelligence", 1.0),         # 人工智能
            ("machine learning", 1.0),                # 机器学习
            
            # 未分类技术（低权重）
            ("visual representation", 1.0),           # 视觉表示
            ("feature extraction", 1.0),              # 特征提取
            ("image processing", 1.0),                # 图像处理
            ("video processing", 1.0),                # 视频处理
            ("visual analysis", 1.0),                 # 视觉分析
        ],
        "negative_keywords": [],  # 无负面关键词，作为兜底分类
        "priority": 1,  # 最低优先级
    },
}