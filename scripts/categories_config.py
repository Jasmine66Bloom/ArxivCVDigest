"""
计算机视觉论文分类配置文件 (2025前瞻版)
包含一级分类和二级分类的层次结构，以及中英文对照和元数据
"""

# 类别阈值配置（越大越严格）
CATEGORY_THRESHOLDS = {
    # 一、表示学习方向
    "表示学习 (Representation Learning)": {
        "threshold": 1.15,
        "subcategories": {
            "基础模型 (Foundation Models)": 1.25,
            "预训练模型 (Pretrained Models)": 1.25,
            "视觉Transformer (Vision Transformers)": 1.25,
        },
        "priority": 5
    },
    
    "生成建模 (Generative Modeling)": {
        "threshold": 1.15,
        "subcategories": {
            "扩散模型 (Diffusion Models)": 1.25,
            "生成对抗网络 (GANs)": 1.25,
            "自回归模型 (Autoregressive Models)": 1.25,
        },
        "priority": 4.8
    },
    
    "多模态学习 (Multimodal Learning)": {
        "threshold": 1.15,
        "subcategories": {
            "视觉语言模型 (Vision-Language Models)": 1.25,
            "跨模态对齐 (Cross-modal Alignment)": 1.25,
            "多模态融合 (Multimodal Fusion)": 1.25,
        },
        "priority": 4.7
    },
    
    # 二、视觉感知方向
    "目标检测识别 (Object Detection & Recognition)": {
        "threshold": 1.2,
        "subcategories": {
            "二维检测 (2D Detection)": 1.3,
            "三维检测 (3D Detection)": 1.3,
            "多目标跟踪 (Multi-object Tracking)": 1.3,
        },
        "priority": 4.5
    },
    
    "场景理解 (Scene Understanding)": {
        "threshold": 1.2,
        "subcategories": {
            "语义分割 (Semantic Segmentation)": 1.3,
            "实例分割 (Instance Segmentation)": 1.3,
            "全景分割 (Panoptic Segmentation)": 1.3,
        },
        "priority": 4.4
    },
    
    "时序理解 (Temporal Understanding)": {
        "threshold": 1.2,
        "subcategories": {
            "动作识别 (Action Recognition)": 1.3,
            "时序分析 (Temporal Analysis)": 1.3,
            "视频预测 (Video Prediction)": 1.3,
        },
        "priority": 4.3
    },
    
    # 三、几何视觉方向
    "三维重建 (3D Reconstruction)": {
        "threshold": 1.2,
        "subcategories": {
            "单目重建 (Monocular Reconstruction)": 1.3,
            "多视图重建 (Multi-view Reconstruction)": 1.3,
            "神经隐式重建 (Neural Implicit Reconstruction)": 1.3,
        },
        "priority": 4.4
    },
    
    "神经渲染 (Neural Rendering)": {
        "threshold": 1.2,
        "subcategories": {
            "神经辐射场 (Neural Radiance Fields)": 1.3,
            "可控渲染 (Controllable Rendering)": 1.3,
            "场景编辑 (Scene Editing)": 1.3,
        },
        "priority": 4.3
    },
    
    "定位与映射 (Localization & Mapping)": {
        "threshold": 1.2,
        "subcategories": {
            "视觉SLAM (Visual SLAM)": 1.3,
            "位姿估计 (Pose Estimation)": 1.3,
            "语义建图 (Semantic Mapping)": 1.3,
        },
        "priority": 4.2
    },
    
    # 四、学习算法方向
    "自监督学习 (Self-supervised Learning)": {
        "threshold": 1.15,
        "subcategories": {
            "对比学习 (Contrastive Learning)": 1.25,
            "掩码自编码 (Masked Autoencoding)": 1.25,
            "一致性学习 (Consistency Learning)": 1.25,
        },
        "priority": 4.6
    },
    
    "迁移与适应 (Transfer & Adaptation)": {
        "threshold": 1.2,
        "subcategories": {
            "元学习 (Meta Learning)": 1.3,
            "域适应 (Domain Adaptation)": 1.3,
            "增量学习 (Incremental Learning)": 1.3,
        },
        "priority": 4.2
    },
    
    "鲁棒学习 (Robust Learning)": {
        "threshold": 1.2,
        "subcategories": {
            "对抗攻击 (Adversarial Attack)": 1.3,
            "对抗防御 (Adversarial Defense)": 1.3,
            "对抗训练 (Adversarial Training)": 1.3,
        },
        "priority": 4.1
    },
    
    # 五、模型优化方向
    "模型压缩加速 (Model Compression & Acceleration)": {
        "threshold": 1.2,
        "subcategories": {
            "知识蒸馏 (Knowledge Distillation)": 1.3,
            "网络剪枝 (Network Pruning)": 1.3,
            "量化优化 (Quantization)": 1.3,
        },
        "priority": 4.0
    },
    
    "泛化与鲁棒性 (Generalization & Robustness)": {
        "threshold": 1.2,
        "subcategories": {
            "域泛化 (Domain Generalization)": 1.3,
            "分布鲁棒性 (Distribution Robustness)": 1.3,
            "不确定性建模 (Uncertainty Modeling)": 1.3,
        },
        "priority": 4.0
    },
    
    "可解释性 (Interpretability)": {
        "threshold": 1.2,
        "subcategories": {
            "可视化解释 (Visual Explanation)": 1.3,
            "归因分析 (Attribution Analysis)": 1.3,
            "概念解释 (Concept Explanation)": 1.3,
        },
        "priority": 3.9
    },
    
    # 六、应用方向
    "医学影像分析 (Medical Image Analysis)": {
        "threshold": 1.2,
        "subcategories": {
            "疾病诊断 (Disease Diagnosis)": 1.3,
            "医学分割 (Medical Segmentation)": 1.3,
            "影像重建 (Image Reconstruction)": 1.3,
        },
        "priority": 4.0
    },
    
    "智能驾驶 (Intelligent Driving)": {
        "threshold": 1.2,
        "subcategories": {
            "环境感知 (Environment Perception)": 1.3,
            "轨迹预测 (Trajectory Prediction)": 1.3,
            "决策规划 (Decision Planning)": 1.3,
        },
        "priority": 4.0
    },
    
    "工业视觉 (Industrial Vision)": {
        "threshold": 1.2,
        "subcategories": {
            "缺陷检测 (Defect Detection)": 1.3,
            "质量控制 (Quality Control)": 1.3,
            "工业测量 (Industrial Measurement)": 1.3,
        },
        "priority": 3.8
    },
    
    # 其他类别
    "其他 (Others)": {
        "threshold": 1.7,
        "subcategories": {},
        "priority": 1
    }
}

# 类别显示顺序配置
CATEGORY_DISPLAY_ORDER = [
    # 一、表示学习方向
    "表示学习 (Representation Learning)",
    "生成建模 (Generative Modeling)",
    "多模态学习 (Multimodal Learning)",
    
    # 二、视觉感知方向
    "目标检测识别 (Object Detection & Recognition)",
    "场景理解 (Scene Understanding)",
    "时序理解 (Temporal Understanding)",
    
    # 三、几何视觉方向
    "三维重建 (3D Reconstruction)",
    "神经渲染 (Neural Rendering)",
    "定位与映射 (Localization & Mapping)",
    
    # 四、学习算法方向
    "自监督学习 (Self-supervised Learning)",
    "迁移与适应 (Transfer & Adaptation)",
    "鲁棒学习 (Robust Learning)",
    
    # 五、模型优化方向
    "模型压缩加速 (Model Compression & Acceleration)",
    "泛化与鲁棒性 (Generalization & Robustness)",
    "可解释性 (Interpretability)",
    
    # 六、应用方向
    "医学影像分析 (Medical Image Analysis)",
    "智能驾驶 (Intelligent Driving)",
    "工业视觉 (Industrial Vision)",
    
    # 其他类别
    "其他 (Others)"
]

# 分类提示词
CATEGORY_PROMPT = """
请将以下计算机视觉论文分类到最合适的类别中。

2025年计算机视觉分类体系：

一、表示学习方向
1. 表示学习 (Representation Learning)：基础模型、预训练模型、视觉Transformer
   定义：研究视觉表示学习的基础模型、多模态预训练和视觉Transformer等前沿课题。

2. 生成建模 (Generative Modeling)：扩散模型、生成对抗网络、自回归模型
   定义：研究生成高质量视觉内容的方法，包括图像、视频和3D内容的生成与编辑。

3. 多模态学习 (Multimodal Learning)：视觉语言模型、跨模态对齐、多模态融合
   定义：研究视觉与其他模态（如语言、音频）的协同理解和交互，包括视觉-语言模型和跨模态推理。

二、视觉感知方向
4. 目标检测识别 (Object Detection & Recognition)：二维检测、三维检测、多目标跟踪
   定义：研究计算机视觉的基础任务和技术，如检测、分割、分类和视频理解等。

5. 场景理解 (Scene Understanding)：语义分割、实例分割、全景分割
   定义：研究视觉场景解析的任务和技术，如语义分割、实例分割和全景分割等。

6. 时序理解 (Temporal Understanding)：动作识别、时序分析、视频预测
   定义：研究视觉时序建模的任务和技术，如动作识别、时序分析和视频预测等。

三、几何视觉方向
7. 三维重建 (3D Reconstruction)：单目重建、多视图重建、神经隐式重建
   定义：研究视觉几何重建的任务和技术，如单目重建、多视图重建和神经隐式重建等。

8. 神经渲染 (Neural Rendering)：神经辐射场、可控渲染、场景编辑
   定义：研究视觉场景建模的方法，包括神经辐射场、可控渲染和场景编辑等。

9. 定位与映射 (Localization & Mapping)：视觉SLAM、位姿估计、语义建图
   定义：研究视觉定位的任务和技术，如视觉SLAM、位姿估计和语义建图等。

四、学习算法方向
10. 自监督学习 (Self-supervised Learning)：对比学习、掩码自编码、一致性学习
   定义：研究自监督学习的方法和技术，如对比学习、掩码自编码和一致性学习等。

11. 迁移与适应 (Transfer & Adaptation)：元学习、域适应、增量学习
   定义：研究迁移学习的方法和技术，如元学习、域适应和增量学习等。

12. 鲁棒学习 (Robust Learning)：对抗攻击、对抗防御、对抗训练
   定义：研究对抗鲁棒性的方法和技术，如对抗攻击、对抗防御和对抗训练等。

五、模型优化方向
13. 模型压缩加速 (Model Compression & Acceleration)：知识蒸馏、网络剪枝、量化优化
   定义：研究模型优化和推理效率的方法和技术，如知识蒸馏、网络剪枝和量化优化等。

14. 泛化与鲁棒性 (Generalization & Robustness)：域泛化、分布鲁棒性、不确定性建模
   定义：研究模型泛化能力和鲁棒性的方法和技术，如域泛化、分布鲁棒性和不确定性建模等。

15. 可解释性 (Interpretability)：可视化解释、归因分析、概念解释
   定义：研究模型可解释性和概念理解的方法和技术，如可视化解释、归因分析和概念解释等。

六、应用方向
16. 医学影像分析 (Medical Image Analysis)：疾病诊断、医学分割、影像重建
   定义：研究视觉技术在医学领域的应用，如疾病诊断、医学分割和影像重建等。

17. 智能驾驶 (Intelligent Driving)：环境感知、轨迹预测、决策规划
   定义：研究视觉技术在自动驾驶领域的应用，如环境感知、轨迹预测和决策规划等。

18. 工业视觉 (Industrial Vision)：缺陷检测、质量控制、工业测量
   定义：研究视觉技术在工业领域的应用，如缺陷检测、质量控制和工业测量等。

19. 其他 (Others)：不属于以上类别的论文
    定义：其他不能明确归类到上述类别的视觉研究工作。

分类指南：
1. 首先分析论文的核心技术贡献和主要研究目标
2. 考虑论文的方法、实验和应用场景
3. 如果论文涉及多个类别，请选择最核心、最具创新性的方向
4. 优先考虑技术本质而非应用领域（除非应用创新是论文的主要贡献）
5. 只有在确实无法归类到前18个类别时，才选择"其他"类别

边界案例处理：
- 如果论文同时涉及"生成建模"和"多模态学习"，但核心是文本引导的图像生成，应归类为"生成建模"
- 如果论文研究NeRF技术，即使应用于医疗领域，也应优先归类为"神经渲染"而非"医学影像分析"
- 如果论文提出新的视觉基础模型并展示了涌现能力，应归类为"表示学习"而非"目标检测识别"

请分析论文的核心技术和主要贡献，选择最合适的一个类别。只返回类别名称，不要有任何解释或额外文本。
"""

# 类别关键词配置
CATEGORY_KEYWORDS = {
    # 一、表示学习方向
    "表示学习 (Representation Learning)": {
        "keywords": [
            # 视觉基础模型（高权重）
            ("vision foundation model", 2.0),      # 视觉基础模型
            ("vision transformer", 1.8),           # 视觉transformer
            
            # 多模态预训练（高权重）
            ("multimodal pretraining", 2.0),        # 多模态预训练
            ("multimodal foundation model", 2.0),  # 多模态基础模型
        ],
        "negative_keywords": [
            ("specific application", 1.0),         # 特定应用
            ("downstream task", 0.8),              # 下游任务
        ]
    },
    
    "生成建模 (Generative Modeling)": {
        "keywords": [
            # 扩散模型（高权重）
            ("diffusion model", 2.0),              # 扩散模型
            ("latent diffusion", 2.0),             # 潜在扩散
            ("denoising diffusion", 2.0),          # 去噪扩散
            ("score-based", 1.8),                  # 基于分数
            ("generative model", 1.8),             # 生成模型
            
            # 生成对抗网络（高权重）
            ("GANs", 2.0),                         # GANs
            ("generative adversarial networks", 2.0), # 生成对抗网络
            
            # 自回归模型（高权重）
            ("autoregressive model", 2.0),          # 自回归模型
            ("autoregressive networks", 2.0),        # 自回归网络
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
            
            # 多模态融合（高权重）
            ("multimodal fusion", 2.0),            # 多模态融合
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
    
    # 二、视觉感知方向
    "目标检测识别 (Object Detection & Recognition)": {
        "keywords": [
            # 二维检测（高权重）
            ("2D detection", 2.0),             # 二维检测
            ("instance recognition", 2.0),         # 实例识别
            ("object recognition", 2.0),           # 目标识别
            ("face recognition", 1.8),             # 人脸识别
            ("pedestrian detection", 1.8),         # 行人检测
            
            # 三维检测（高权重）
            ("3D detection", 2.0),                # 三维检测
            ("3D generative", 2.0),                # 3D生成式
            ("shape generation", 1.8),             # 形状生成
            
            # 多目标跟踪（高权重）
            ("multi-object tracking", 2.0),        # 多目标跟踪
            ("temporal modeling", 2.0),            # 时序建模
            ("activity recognition", 1.8),         # 活动识别
            ("video classification", 1.8),         # 视频分类
        ],
        "negative_keywords": [
            ("multimodal", 1.0),                   # 多模态
        ]
    },
    
    "场景理解 (Scene Understanding)": {
        "keywords": [
            # 语义分割（高权重）
            ("semantic segmentation", 2.0),        # 语义分割
            ("instance segmentation", 2.0),        # 实例分割
            ("panoptic segmentation", 2.0),        # 全景分割
            ("medical segmentation", 1.8),         # 医学分割
            
            # 多标签分类（高权重）
            ("multi-label classification", 2.0),   # 多标签分类
        ],
        "negative_keywords": [
            ("recognition", 1.0),                  # 识别
            ("classification", 1.0),               # 分类
        ]
    },
    
    "时序理解 (Temporal Understanding)": {
        "keywords": [
            # 动作识别（高权重）
            ("action recognition", 2.0),           # 动作识别
            ("temporal modeling", 2.0),            # 时序建模
            ("activity recognition", 2.0),         # 活动识别
            ("video classification", 1.8),         # 视频分类
        ],
        "negative_keywords": [
            ("recognition", 1.0),                  # 识别
            ("classification", 1.0),               # 分类
        ]
    },
    
    # 三、几何视觉方向
    "三维重建 (3D Reconstruction)": {
        "keywords": [
            # 单目重建（高权重）
            ("monocular reconstruction", 2.0),        # 单目重建
            ("single-view reconstruction", 2.0),      # 单视角重建
            
            # 多视图重建（高权重）
            ("multi-view reconstruction", 2.0),       # 多视角重建
            ("multi-view stereo", 2.0),              # 多视角立体
            ("structure from motion", 2.0),          # 运动结构恢复
            
            # 神经隐式重建（高权重）
            ("neural implicit reconstruction", 2.0),      # 神经隐式重建
            ("neural implicit surface", 2.0),      # 神经隐式表面
            ("implicit geometric", 2.0),           # 隐式几何
            ("implicit representation", 2.0),      # 隐式表示
            ("signed distance function", 2.0),     # 符号距离函数
            ("occupancy network", 1.8),            # 占用网络
        ],
        "negative_keywords": [
            ("2D only", 1.0),                      # 仅2D
            ("3D only", 1.0),                      # 仅3D
            ("multimodal", 1.0),                   # 多模态
        ]
    },
    
    "神经渲染 (Neural Rendering)": {
        "keywords": [
            # 神经辐射场（高权重）
            ("neural radiance field", 2.0),        # 神经辐射场
            ("NeRF", 2.0),                         # NeRF
            ("neural rendering", 2.0),             # 神经渲染
            ("volumetric rendering", 1.8),         # 体积渲染
            ("3D gaussian splatting", 2.0),        # 3D高斯散射
            
            # 可控渲染（高权重）
            ("controllable rendering", 2.0),         # 可控渲染
            ("style transfer", 1.8),               # 风格迁移
            ("physical consistency", 1.8),         # 物理一致性
            ("dynamics-aware", 1.8),               # 动力学感知
        ],
        "negative_keywords": [
            ("2D only", 1.0),                      # 仅2D
            ("3D only", 1.0),                      # 仅3D
            ("multimodal", 1.0),                   # 多模态
        ]
    },
    
    "定位与映射 (Localization & Mapping)": {
        "keywords": [
            # 视觉SLAM（高权重）
            ("visual SLAM", 2.0),                # 视觉SLAM
            ("structure from motion", 2.0),          # 运动结构恢复
            
            # 位姿估计（高权重）
            ("pose estimation", 2.0),             # 位姿估计
            ("camera pose estimation", 2.0),       # 相机位姿估计
            ("object pose estimation", 1.8),       # 物体位姿估计
            
            # 语义建图（高权重）
            ("semantic mapping", 2.0),             # 语义建图
            ("place recognition", 2.0),          # 地点识别
            ("scene understanding", 1.8),          # 场景理解
        ],
        "negative_keywords": [
            ("2D only", 1.0),                      # 仅2D
            ("3D only", 1.0),                      # 仅3D
            ("multimodal", 1.0),                   # 多模态
        ]
    },
    
    # 四、学习算法方向
    "自监督学习 (Self-supervised Learning)": {
        "keywords": [
            # 对比学习（高权重）
            ("contrastive learning", 2.0),         # 对比学习
            ("contrastive representation", 2.0),    # 对比表征
            ("contrastive self-supervised", 2.0),     # 对比自监督
            
            # 掩码自编码（高权重）
            ("masked autoencoding", 2.0),           # 掩码自编码
            ("masked reconstruction", 2.0),           # 掩码重建
            ("masked denoising", 2.0),                # 掩码去噪
            
            # 一致性学习（高权重）
            ("consistency learning", 2.0),          # 一致性学习
            ("temporal consistency", 2.0),            # 时间一致性
            ("domain consistency", 1.8),             # 域一致性
        ],
        "negative_keywords": [
            ("recognition", 1.0),                  # 识别
            ("classification", 1.0),               # 分类
        ]
    },
    
    "迁移与适应 (Transfer & Adaptation)": {
        "keywords": [
            # 元学习（高权重）
            ("meta learning", 2.0),                # 元学习
            ("meta-learning", 2.0),                  # 元学习
            ("learning to learn", 2.0),                # 学习如何学习
            
            # 域适应（高权重）
            ("domain adaptation", 2.0),                # 域适应
            ("domain generalization", 2.0),              # 域泛化
            ("cross-domain", 2.0),                     # 跨域
            ("out-of-distribution", 2.0),             # 分布外
            
            # 增量学习（高权重）
            ("incremental learning", 2.0),            # 增量学习
            ("continual learning", 2.0),                # 持续学习
            ("class incremental", 1.8),                  # 类别增量
            ("task incremental", 1.8),                  # 任务增量
        ],
        "negative_keywords": [
            ("recognition", 1.0),                  # 识别
            ("classification", 1.0),               # 分类
        ]
    },
    
    "鲁棒学习 (Robust Learning)": {
        "keywords": [
            # 对抗攻击（高权重）
            ("adversarial attack", 2.0),                # 对抗攻击
            ("adversarial example", 2.0),                # 对抗示例
            ("adversarial training", 2.0),                # 对抗训练
            
            # 对抗防御（高权重）
            ("adversarial defense", 2.0),                # 对抗防御
            ("robust perception", 2.0),                # 鲁棒感知
            ("robustness", 2.0),                # 鲁棒性
            
            # 对抗鲁棒性（高权重）
            ("adversarial robustness", 2.0),                # 对抗鲁棒性
            ("out-of-distribution", 2.0),                # 分布外
            ("domain robustness", 1.8),                # 域鲁棒性
            
            # 对抗训练（高权重）
            ("adversarial training", 2.0),                # 对抗训练
            ("confidence calibration", 1.8),                # 置信度校准
            ("probabilistic vision", 1.8),                # 概率视觉
        ],
        "negative_keywords": [
            ("recognition", 1.0),                  # 识别
            ("classification", 1.0),               # 分类
        ]
    },
    
    # 五、模型优化方向
    "模型压缩加速 (Model Compression & Acceleration)": {
        "keywords": [
            # 知识蒸馏（高权重）
            ("knowledge distillation", 2.0),       # 知识蒸馏
            ("distillation", 2.0),                # 蒸馏
            ("knowledge transfer", 2.0),                # 知识转移
            
            # 网络剪枝（高权重）
            ("network pruning", 2.0),                # 网络剪枝
            ("sparse activation", 2.0),                # 稀疏激活
            ("conditional computation", 1.8),                # 条件计算
            
            # 量化优化（高权重）
            ("quantization", 2.0),                # 量化
            ("quantization-aware", 2.0),                # 量化感知
            ("quantization-aware training", 2.0),                # 量化感知训练
        ],
        "negative_keywords": [
            ("recognition", 1.0),                  # 识别
            ("classification", 1.0),               # 分类
        ]
    },
    
    "泛化与鲁棒性 (Generalization & Robustness)": {
        "keywords": [
            # 域泛化（高权重）
            ("domain generalization", 2.0),                # 域泛化
            ("out-of-distribution", 2.0),                # 分布外
            ("cross-domain", 2.0),                # 跨域
            
            # 分布鲁棒性（高权重）
            ("distribution robustness", 2.0),                # 分布鲁棒性
            ("robustness", 2.0),                # 鲁棒性
            
            # 不确定性建模（高权重）
            ("uncertainty modeling", 2.0),                # 不确定性建模
            ("uncertainty quantification", 2.0),                # 不确定性量化
            ("bayesian vision", 2.0),                # 贝叶斯视觉
            ("probabilistic vision", 1.8),                # 概率视觉
        ],
        "negative_keywords": [
            ("recognition", 1.0),                  # 识别
            ("classification", 1.0),               # 分类
        ]
    },
    
    "可解释性 (Interpretability)": {
        "keywords": [
            # 可视化解释（高权重）
            ("visual explanation", 2.0),                # 视觉解释
            ("attribution method", 2.0),                # 归因方法
            ("saliency map", 2.0),                # 显著图
            
            # 归因分析（高权重）
            ("attribution analysis", 2.0),                # 归因分析
            ("concept explanation", 2.0),                # 概念解释
            ("concept learning", 1.8),                # 概念学习
        ],
        "negative_keywords": [
            ("recognition", 1.0),                  # 识别
            ("classification", 1.0),               # 分类
        ]
    },
    
    # 六、应用方向
    "医学影像分析 (Medical Image Analysis)": {
        "keywords": [
            # 疾病诊断（高权重）
            ("disease diagnosis", 2.0),                # 疾病诊断
            ("healthcare monitoring", 2.0),                # 健康监测
            ("clinical vision", 1.8),                # 临床视觉
            ("biomedical imaging", 1.8),                # 生物医学影像
            
            # 医学分割（高权重）
            ("medical segmentation", 2.0),                # 医学分割
            ("semantic segmentation", 2.0),                # 语义分割
            ("instance segmentation", 2.0),                # 实例分割
            ("panoptic segmentation", 2.0),                # 全景分割
            ("medical imaging", 2.0),                # 医学影像
            
            # 影像重建（高权重）
            ("image reconstruction", 2.0),                # 图像重建
            ("healthcare imaging", 1.8),                # 健康影像
            ("clinical imaging", 1.8),                # 临床影像
            ("biomedical imaging", 1.8),                # 生物医学影像
        ],
        "negative_keywords": [
            ("recognition", 1.0),                  # 识别
            ("classification", 1.0),               # 分类
        ]
    },
    
    "智能驾驶 (Intelligent Driving)": {
        "keywords": [
            # 环境感知（高权重）
            ("environment perception", 2.0),                # 环境感知
            ("self-driving perception", 2.0),                # 自动驾驶感知
            ("vehicle perception", 2.0),                # 车辆感知
            ("traffic scene understanding", 1.8),                # 交通场景理解
            ("driving scene analysis", 1.8),                # 驾驶场景分析
            
            # 轨迹预测（高权重）
            ("trajectory prediction", 2.0),                # 轨迹预测
            ("motion prediction", 2.0),                # 运动预测
            ("future prediction", 1.8),                # 未来预测
            
            # 决策规划（高权重）
            ("decision planning", 2.0),                # 决策规划
            ("decision making", 2.0),                # 决策制定
            ("decision-making", 2.0),                # 决策制定
            ("decision-making process", 1.8),                # 决策制定过程
        ],
        "negative_keywords": [
            ("recognition", 1.0),                  # 识别
            ("classification", 1.0),               # 分类
        ]
    },
    
    "工业视觉 (Industrial Vision)": {
        "keywords": [
            # 缺陷检测（高权重）
            ("defect detection", 2.0),                # 缺陷检测
            ("quality control", 2.0),                # 质量控制
            ("industrial measurement", 2.0),                # 工业测量
            
            # 工业自动化（高权重）
            ("industrial automation", 2.0),                # 工业自动化
            ("manufacturing inspection", 2.0),                # 制造检测
            ("quality control vision", 2.0),                # 质量控制视觉
            
            # 工业测量（高权重）
            ("industrial measurement", 2.0),                # 工业测量
            ("manufacturing measurement", 1.8),                # 制造测量
            ("quality measurement", 1.8),                # 质量测量
        ],
        "negative_keywords": [
            ("recognition", 1.0),                  # 识别
            ("classification", 1.0),               # 分类
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