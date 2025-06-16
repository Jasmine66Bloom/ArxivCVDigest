"""
计算机视觉论文分类配置文件
包含一级分类和二级分类的层次结构，以及中英文对照和元数据
"""

# 类别阈值配置（越大越严格）
CATEGORY_THRESHOLDS = {
    # 应用类别 - 优先级保持高，但阈值提高以减少过多分类
    # 1. 生物医学影像计算 - 显著提高阈值以减少过多分类
    "生物医学影像计算 (Biomedical Image Computing)": {
        "threshold": 3.5,
        "subcategories": {
            "医学图像分析与分割 (Medical Image Analysis & Segmentation)": 3.6,
            "计算病理学与诊断 (Computational Pathology & Diagnostics)": 3.6,
            "医学AI辅助诊断 (Medical AI-assisted Diagnosis)": 3.7,
        },
        "priority": 2.0
    },
    
    # 2. 智能交通与自主系统
    "智能交通与自主系统 (Intelligent Transportation & Autonomous Systems)": {
        "threshold": 3.0,
        "subcategories": {
            "自主驾驶感知与决策 (Autonomous Driving Perception & Decision)": 3.1,
            "端到端自动驾驶系统 (End-to-end Autonomous Driving Systems)": 3.2,
            "智能交通系统与城市计算 (Intelligent Transportation Systems & Urban Computing)": 3.3,
        },
        "priority": 2.0
    },
    
    # 3. 工业视觉与遥感感知
    "工业视觉与遥感感知 (Industrial Vision & Remote Sensing)": {
        "threshold": 3.0,
        "subcategories": {
            "工业缺陷检测与质量控制 (Industrial Defect Detection & Quality Control)": 3.1,
            "遥感大模型与地理分析 (Remote Sensing Foundation Models & Geospatial Analytics)": 3.2,
            "环境监测与精准农业 (Environmental Monitoring & Precision Agriculture)": 3.3,
        },
        "priority": 2.08
    },
    
    # 4. 交互媒体与虚拟现实技术
    "交互媒体与虚拟现实技术 (Interactive Media & Extended Reality)": {
        "threshold": 3.0,  # 提高阈值以与其他应用类别一致
        "subcategories": {
            "沉浸式虚拟现实与AR/VR (Immersive XR & AR/VR)": 3.1,
            "数字人与虚拟人物 (Digital Humans & Virtual Avatars)": 3.2,
            "交互式媒体与元宇宙 (Interactive Media & Metaverse)": 3.3,
        },
        "priority": 2.0
    },
    
    # 基础研究类别
    # 5. 视觉表征学习与大模型
    "视觉表征学习与大模型 (Visual Representation Learning & Foundation Models)": {
        "threshold": 1.15,
        "subcategories": {
            "视觉大模型与基础模型 (Vision Foundation Models)": 1.2,
            "视觉Transformer架构 (Vision Transformer Architectures)": 1.25,
            "自监督与对比学习 (Self-supervised & Contrastive Learning)": 1.3,
        },
        "priority": 8.0
    },
    
    # 6. 静态图像理解与语义解析
    "静态图像理解与语义解析 (Static Image Understanding & Semantic Analysis)": {
        "threshold": 1.15,
        "subcategories": {
            "目标检测与定位 (Object Detection & Localization)": 1.2,
            "分割与实例识别 (Segmentation & Instance Recognition)": 1.25,
            "场景理解与视觉推理 (Scene Understanding & Visual Reasoning)": 1.3,
        },
        "priority": 8.0
    },
    
    # 7. 生成式视觉与内容创建
    "生成式视觉与内容创建 (Generative Vision & Content Creation)": {
        "threshold": 1.15,
        "subcategories": {
            "文本到图像生成 (Text-to-Image Generation)": 1.2,
            "图像编辑与操作 (Image Editing & Manipulation)": 1.25,
            "深度生成模型与扮演学习 (Deep Generative Models & GANs)": 1.3,
        },
        "priority": 8.0
    },
    
    # 8. 三维重建与几何感知
    "三维重建与几何感知 (3D Reconstruction & Geometric Perception)": {
        "threshold": 1.15,
        "subcategories": {
            "神经辐射场与隐式表示 (Neural Radiance Fields & Implicit Representations)": 1.2,
            "三维重建与深度估计 (3D Reconstruction & Depth Estimation)": 1.25,
            "新视角合成与生成 (Novel View Synthesis & Generation)": 1.3,
        },
        "priority": 8.0
    },
    
    # 9. 动态视觉与时序建模
    "动态视觉与时序建模 (Dynamic Vision & Temporal Modeling)": {
        "threshold": 1.15,
        "subcategories": {
            "视频理解与时序建模 (Video Understanding & Temporal Modeling)": 1.2,
            "动作识别与人体姿态 (Action Recognition & Human Pose)": 1.25,
            "多目标跟踪与运动预测 (Multi-object Tracking & Motion Prediction)": 1.3,
        },
        "priority": 8.0
    },
    
    # 10. 多模态视觉与跨模态学习
    "多模态视觉与跨模态学习 (Multimodal Vision & Cross-modal Learning)": {
        "threshold": 1.15,
        "subcategories": {
            "视觉-语言对齐与理解 (Vision-Language Alignment & Understanding)": 1.2,
            "多模态大模型 (Multimodal Large Language Models)": 1.25,
            "视觉问答与描述 (Visual QA & Description)": 1.3,
        },
        "priority": 8.0
    },
    
    # 11. 模型优化与系统鲁棒性
    "模型优化与系统鲁棒性 (Model Optimization & System Robustness)": {
        "threshold": 1.15,
        "subcategories": {
            "模型压缩与量化 (Model Compression & Quantization)": 1.2,
            "鲁棒性与对抗样本 (Robustness & Adversarial Examples)": 1.25,
            "可解释性与公平性 (Explainability & Fairness)": 1.3,
        },
        "priority": 8.0
    },
    
    # 12. 效率学习与适应性智能
    "效率学习与适应性智能 (Efficient Learning & Adaptive Intelligence)": {
        "threshold": 0.5,
        "subcategories": {
            "小样本与零样本学习 (Few-shot & Zero-shot Learning)": 1.2,
            "迁移学习与域适应 (Transfer Learning & Domain Adaptation)": 1.25,
            "持续学习与元学习 (Continual Learning & Meta-learning)": 1.3,
        },
        "priority": 8.00
    },
    
    # 13. 感知-动作智能与主动视觉
    "感知-动作智能与主动视觉 (Perception-Action Intelligence & Active Vision)": {
        "threshold": 0.5,  # 恢复到原始阈值
        "subcategories": {
            "具身视觉与机器人感知 (Embodied Vision & Robotic Perception)": 1.2,
            "视觉导航与主动探索 (Visual Navigation & Active Exploration)": 1.25,
            "感知-动作循环与交互 (Perception-Action Loop & Interaction)": 1.3,
        },
        "priority": 8.0
    },
    
    # 14. 前沿视觉理论与跨学科融合
    "前沿视觉理论与跨学科融合 (Frontier Vision Theory & Interdisciplinary Integration)": {
        "threshold": 0.5,
        "subcategories": {
            "计算神经科学与认知视觉 (Computational Neuroscience & Cognitive Vision)": 1.2,
            "视觉基础理论与数学模型 (Vision Foundations & Mathematical Models)": 1.25,
            "跨学科融合与新型计算范式 (Interdisciplinary Integration & Novel Computing Paradigms)": 1.3,
        },
        "priority": 8.00
    },
    
    # 其他类别
    "其他 (Others)": {
        "threshold": 0.5,
        "subcategories": {},
        "priority": 1
    }
}

# 类别显示顺序配置
CATEGORY_DISPLAY_ORDER = [
    # 基础研究类别
    "视觉表征学习与大模型 (Visual Representation Learning & Foundation Models)",
    "静态图像理解与语义解析 (Static Image Understanding & Semantic Analysis)",
    "生成式视觉与内容创建 (Generative Vision & Content Creation)",
    "三维重建与几何感知 (3D Reconstruction & Geometric Perception)",
    "动态视觉与时序建模 (Dynamic Vision & Temporal Modeling)",
    "多模态视觉与跨模态学习 (Multimodal Vision & Cross-modal Learning)",
    "模型优化与系统鲁棒性 (Model Optimization & System Robustness)",
    "效率学习与适应性智能 (Efficient Learning & Adaptive Intelligence)",
    "感知-动作智能与主动视觉 (Perception-Action Intelligence & Active Vision)",
    "前沿视觉理论与跨学科融合 (Frontier Vision Theory & Interdisciplinary Integration)",
    # 应用类别优先显示
    "生物医学影像计算 (Biomedical Image Computing)",
    "智能交通与自主系统 (Intelligent Transportation & Autonomous Systems)",
    "工业视觉与遥感感知 (Industrial Vision & Remote Sensing)",
    "交互媒体与虚拟现实技术 (Interactive Media & Extended Reality)",
    # 其他类别
    "其他 (Others)"
]

# 分类提示词
CATEGORY_PROMPT = """
你是一个计算机视觉领域的专家，请帮我将下面的论文分类到以下类别中的一个：

# 应用类别
1. 生物医学影像计算 (Biomedical Image Computing)
   定义：研究医学图像分析与分割、计算病理学、计算机辅助诊断、医学图像重建与配准等生物医学领域的视觉计算技术。

2. 智能交通与自主系统 (Intelligent Transportation & Autonomous Systems)
   定义：研究自主驾驶感知与决策、多传感器融合与引擎、端到端自动驾驶系统等智能交通领域的视觉感知技术。

3. 工业视觉与遥感感知 (Industrial Vision & Remote Sensing)
   定义：研究工业缺陷检测与质量控制、遥感大模型与地理分析、环境监测与精准农业等工业和遥感领域的计算机视觉技术。

4. 交互媒体与虚拟现实技术 (Interactive Media & Extended Reality)
   定义：研究沉浸式虚拟现实与AR/VR、数字人与虚拟人物、交互式媒体与元宇宙等交互媒体领域的视觉计算技术。

# 基础研究类别（优先考虑）
5. 视觉表征学习与大模型 (Visual Representation Learning & Foundation Models)
   定义：研究视觉大模型与基础模型、视觉Transformer架构和自监督与对比学习等前沿技术。

6. 静态图像理解与语义解析 (Static Image Understanding & Semantic Analysis)
   定义：研究目标检测与定位、分割与实例识别、场景理解与视觉推理等视觉识别任务。

7. 生成式视觉与内容创建 (Generative Vision & Content Creation)
   定义：研究扬散模型与文生图、视频生成与编辑、控制生成与定制化等生成式视觉技术。

8. 三维重建与几何感知 (3D Reconstruction & Geometric Perception)
   定义：研究神经辐射场与隐式表示、三维重建与深度估计、新视角合成与生成等三维视觉技术。

9. 动态视觉与时序建模 (Dynamic Vision & Temporal Modeling)
   定义：研究视频理解与时序建模、动作识别与人体姿态、多目标跟踪与运动预测等时序视觉技术。

10. 多模态视觉与跨模态学习 (Multimodal Vision & Cross-modal Learning)
    定义：研究视觉-语言对齐与理解、多模态大模型、视觉问答与描述等视觉-语言协同技术。

11. 模型优化与系统鲁棒性 (Model Optimization & System Robustness)
    定义：研究模型压缩与量化、鲁棒性与对抗样本、可解释性与公平性等计算效率技术。

12. 效率学习与适应性智能 (Efficient Learning & Adaptive Intelligence)
     定义：研究小样本与零样本学习、迁移学习与域适应、持续学习与元学习等低资源条件下的高效学习技术。

13. 感知-动作智能与主动视觉 (Perception-Action Intelligence & Active Vision)
     定义：研究具身视觉与机器人感知、视觉导航与主动探索、感知-动作循环与交互等技术。

14. 前沿视觉理论与跨学科融合 (Frontier Vision Theory & Interdisciplinary Integration)
     定义：研究神经-符号视觉与因果推理、生物启发视觉与认知科学、跨学科视觉应用等前沿视觉理论。

15. 其他 (Others)
    定义：不属于以上类别的其他计算机视觉研究。

分类指南：
1. 首先分析论文的核心技术贡献和主要研究目标
2. 考虑论文的方法、实验和应用场景
3. 如果论文涉及多个类别，请选择最核心、最具创新性的方向
4. 优先考虑技术本质而非应用领域（除非应用创新是论文的主要贡献）
5. 只有在确实无法归类到前14个类别时，才选择“其他”类别

边界案例处理：
- 如果论文同时涉及“生成式视觉与内容创建”和“多模态视觉与跨模态学习”，但核心是文本引导的图像生成，应归类为“生成式视觉与内容创建”
- 如果论文研究NeRF技术，即使应用于医疗领域，也应优先归类为“三维重建与几何感知”而非“生物医学影像计算”
- 如果论文提出新的视觉基础模型并展示了涌现能力，应归类为“视觉表征学习与大模型”而非“静态图像理解与语义解析”
- 如果论文提出的方法同时适用于多个应用领域，应归类为其所属的基础研究类别，而非具体应用类别
- 如果论文研究多模态大模型，但重点是视觉语言对齐而非生成能力，应归类为“多模态视觉与跨模态学习”
- 如果论文研究机器人视觉导航，应归类为“感知-动作智能与主动视觉”而非“智能交通与自主系统”（除非特别强调交通场景）
- 如果论文研究小样本学习或迁移学习，但应用于特定领域（如医学影像），应根据技术创新点判断：如果创新在学习方法，归为“效率学习与适应性智能”；如果创新在应用领域，则归为相应应用类别
- 如果论文研究视频生成或编辑，应归类为“生成式视觉与内容创建”而非“动态视觉与时序建模”
- 如果论文提出的方法主要针对模型压缩或加速，即使应用于特定类型模型，也应归类为“模型优化与系统鲁棒性”

请分析论文的核心技术和主要贡献，选择最合适的一个类别。只返回类别名称，不要有任何解释或额外文本。
"""

# 类别关键词配置
CATEGORY_KEYWORDS = {
    # 应用类别关键词
    # 1. 生物医学影像计算 - 增强关键词权重并增加标题必要条件
    "生物医学影像计算 (Biomedical Image Computing)": {
        "keywords": [
            # 医学图像分析（超高权重）- 标题必包含关键词
            ("medical image", 3.5),               # 医学图像
            ("medical imaging", 3.5),             # 医学影像
            ("biomedical image", 3.5),            # 生物医学图像
            ("healthcare imaging", 3.5),          # 医疗保健影像
            ("clinical imaging", 3.5),            # 临床影像
            
            # 医学成像技术（高权重）- 标题必包含关键词
            ("radiology", 3.3),                   # 放射学
            ("CT scan", 3.3),                     # CT扫描
            ("MRI scan", 3.3),                    # 核磁共振扫描
            ("X-ray imaging", 3.3),                # X射线成像
            ("ultrasound imaging", 3.3),           # 超声成像
            
            # 医学分析与诊断（超高权重）- 标题必包含关键词
            ("medical segmentation", 3.5),        # 医学分割
            ("lesion detection", 3.5),            # 病变检测
            ("tumor segmentation", 3.5),          # 肿瘤分割
            ("pathology analysis", 3.3),           # 病理分析
            ("medical diagnosis", 3.3),            # 医学诊断
            ("tumor detection", 3.3),              # 肿瘤检测
            ("cancer diagnosis", 3.3),             # 癌症诊断
            ("clinical diagnosis", 3.3),           # 临床诊断
            ("healthcare analytics", 3.3),         # 医疗保健分析
            
            # 应用研究特征关键词（超高权重）- 标题必包含关键词
            ("medical application", 3.8),         # 医学应用
            ("clinical application", 3.8),        # 临床应用
            ("healthcare application", 3.8),      # 医疗保健应用
            ("medical domain", 3.8),              # 医学领域
            
            # 中文关键词（超高权重）- 标题必包含关键词
            ("医学影像", 3.5),                  # 医学影像
            ("医学成像", 3.5),                  # 医学成像
            ("医疗影像", 3.5),                  # 医疗影像
            ("病理分析", 3.5),                  # 病理分析
            ("医学诊断", 3.5),                  # 医学诊断
            ("医学分割", 3.5),                  # 医学分割
            ("病变检测", 3.5),                 # 病变检测
            ("肿瘤分析", 3.3),                  # 肿瘤分析
            ("癌症诊断", 3.3),                  # 癌症诊断
            ("医学应用", 3.8),                  # 医学应用
            ("医疗应用", 3.8),                  # 医疗应用
            ("临床应用", 3.8),                  # 临床应用
        ],
        "negative_keywords": [
            ("general vision", 2.0),              # 通用视觉
            ("foundation model", 2.0),            # 基础模型
            ("general purpose", 2.0),             # 通用目的
            ("non-medical", 2.0),                 # 非医学
            ("autonomous", 2.0),                  # 自主系统
            ("robotics", 2.0),                    # 机器人
            ("industrial", 2.0),                  # 工业
            ("remote sensing", 2.0),              # 遥感
            ("virtual reality", 2.0),             # 虚拟现实
            ("augmented reality", 2.0),           # 增强现实
            ("natural language", 2.0),            # 自然语言
            ("video generation", 2.0),            # 视频生成
            ("image generation", 2.0),            # 图像生成
        ]
    },
    
    # 2. 智能交通与自主系统 - 增强关键词权重
    "智能交通与自主系统 (Intelligent Transportation & Autonomous Systems)": {
        "keywords": [
            # 自主驾驶感知与决策（超高权重）
            ("autonomous driving", 3.3),           # 自动驾驶
            ("self-driving", 3.3),                # 自动驾驶
            ("driving perception", 3.3),          # 驾驶感知
            ("driving decision", 3.3),            # 驾驶决策
            ("driver assistance", 3.3),           # 驾驶辅助
            
            # 交通监控与分析（高权重）
            ("traffic monitoring", 3.1),           # 交通监控
            ("traffic analysis", 3.1),             # 交通分析
            ("intelligent transportation", 3.1),   # 智能交通
            ("traffic scene", 3.1),                # 交通场景
            
            # 传感器融合与感知（高权重）
            ("sensor fusion", 3.1),                # 传感器融合
            ("lidar", 3.1),                        # 激光雷达
            ("radar", 3.1),                        # 雷达
            ("vehicle detection", 3.1),            # 车辆检测
            ("vehicle tracking", 3.1),             # 车辆跟踪
            
            # 应用研究特征关键词（超高权重）
            ("transportation application", 3.3),    # 交通应用
            ("automotive application", 3.3),       # 汽车应用
            ("in autonomous driving", 3.3),        # 在自动驾驶中
            ("for autonomous driving", 3.3),       # 用于自动驾驶
            ("in intelligent transportation", 3.3), # 在智能交通中
            ("for intelligent transportation", 3.3), # 用于智能交通
            ("in vehicle", 3.3),                   # 在车辆中
            ("for vehicle", 3.3),                  # 用于车辆
            
            # 中文关键词（超高权重）
            ("自动驾驶", 3.3),                    # 自动驾驶
            ("智能交通", 3.3),                    # 智能交通
            ("车辆检测", 3.3),                    # 车辆检测
            ("车辆跟踪", 3.3),                    # 车辆跟踪
            ("驾驶辅助", 3.3),                    # 驾驶辅助
            ("交通", 3.0),                        # 交通
            ("车辆", 3.0),                        # 车辆
            ("道路", 3.0),                        # 道路
            ("传感器融合", 3.0),                  # 传感器融合
            ("端到端", 3.0),                      # 端到端
            ("交通应用", 3.3),                  # 交通应用
            ("汽车应用", 3.3),                  # 汽车应用
            ("驾驶应用", 3.3),                  # 驾驶应用
        ],
        "negative_keywords": [
            ("general vision", 1.5),              # 通用视觉
            ("foundation model", 1.2),            # 基础模型
            ("robot navigation", 1.5),            # 机器人导航
            ("embodied AI", 1.5),                 # 具身人工智能
            ("general purpose", 1.5),             # 通用目的
        ]
    },
    
    # 3. 工业视觉与遥感感知 - 增强关键词权重
    "工业视觉与遥感感知 (Industrial Vision & Remote Sensing)": {
        "keywords": [
            # 智能制造与质量检测（超高权重）
            ("industrial inspection", 3.2),        # 工业检测
            ("quality control", 3.2),             # 质量控制
            ("defect detection", 3.2),            # 缺陷检测
            ("manufacturing", 3.0),               # 制造
            ("industrial vision", 3.2),           # 工业视觉
            ("smart manufacturing", 3.2),         # 智能制造
            ("factory automation", 3.2),          # 工厂自动化
            ("industrial quality", 3.2),          # 工业质量
            
            # 遥感大模型与分析（超高权重）
            ("remote sensing foundation model", 3.2), # 遥感基础模型
            ("satellite vision model", 3.2),      # 卫星视觉模型
            ("remote sensing analysis", 3.2),     # 遥感分析
            ("earth observation", 3.2),           # 地球观测
            
            # 环境监测与地理信息（超高权重）
            ("environmental monitoring", 3.2),    # 环境监测
            ("remote sensing", 3.2),              # 遥感
            ("satellite image", 3.2),             # 卫星图像
            ("aerial image", 3.2),                # 航空图像
            ("GIS", 3.2),                         # 地理信息系统
            ("geospatial", 3.2),                  # 地理空间
            ("land cover", 3.0),                  # 土地覆盖
            
            # 应用研究特征关键词（超高权重）
            ("industrial application", 3.2),       # 工业应用
            ("manufacturing application", 3.2),    # 制造应用
            ("remote sensing application", 3.2),   # 遥感应用
            ("in industrial", 3.2),                # 在工业中
            ("for industrial", 3.2),               # 用于工业
            ("in manufacturing", 3.2),             # 在制造中
            ("for manufacturing", 3.2),            # 用于制造
            ("in remote sensing", 3.2),            # 在遥感中
            ("for remote sensing", 3.2),           # 用于遥感
            
            # 中文关键词（超高权重）
            ("工业", 3.2),                      # 工业
            ("检测", 3.0),                      # 检测
            ("质量", 3.0),                      # 质量
            ("遥感", 3.2),                      # 遥感
            ("工业视觉", 3.2),                  # 工业视觉
            ("卫星图像", 3.2),                 # 卫星图像
            ("缺陷检测", 3.2),                 # 缺陷检测
            ("地理信息", 3.2),                 # 地理信息
            ("卫星", 3.0),                      # 卫星
            ("智能制造", 3.0),                  # 智能制造
            ("环境监测", 3.0),                  # 环境监测
            ("工业应用", 3.2),                  # 工业应用
            ("制造应用", 3.2),                  # 制造应用
            ("遥感应用", 3.2),                  # 遥感应用
        ],
        "negative_keywords": [
            ("general vision", 1.0),              # 通用视觉
            ("foundation model", 0.8),            # 基础模型
        ]
    },
    
    # 4. 交互媒体与虚拟现实技术 - 增强关键词权重
    "交互媒体与虚拟现实技术 (Interactive Media & Extended Reality)": {
        "keywords": [
            # 虚拟现实与增强现实（超高权重）
            ("virtual reality", 3.0),             # 虚拟现实
            ("augmented reality", 3.0),           # 增强现实
            ("mixed reality", 3.0),               # 混合现实
            ("extended reality", 3.0),            # 扩展现实
            ("XR", 3.0),                          # XR数字孤生
            ("virtual character", 3.0),           # 虚拟角色
            ("digital human", 3.0),               # 数字人
            ("avatar", 3.0),                      # 虚拟形象
            ("metaverse", 3.0),                   # 元宇宙
            
            # 交互体验与游戏技术（超高权重）
            ("interactive experience", 3.0),      # 交互体验
            ("game technology", 3.0),             # 游戏技术
            ("immersive experience", 3.0),        # 沉浸式体验
            ("3D interaction", 3.0),              # 3D交互
            ("human-computer interaction", 3.0),  # 人机交互
            ("HCI", 3.0),                         # 人机交互
            
            # 其他相关技术
            ("AR", 3.0),                          # AR
            ("VR", 3.0),                          # VR
            
            # 中文关键词（超高权重）
            ("虚拟现实", 3.0),                  # 虚拟现实
            ("增强现实", 3.0),                  # 增强现实
            ("数字人", 3.0),                    # 数字人
            ("元宇宙", 3.0),                    # 元宇宙
            ("交互媒体", 3.0),                  # 交互媒体
            ("沉浸式", 3.0),                    # 沉浸式
            ("creative content", 3.0),            # 创意内容
            ("digital art", 3.0),                 # 数字艺术
            ("media generation", 3.0),            # 媒体生成
            ("entertainment", 3.0),               # 娱乐
            ("创意", 3.0),                      # 创意
            ("娱乐", 3.0),                      # 娱乐
            ("游戏", 3.0),                      # 游戏
            ("交互应用", 3.0),                # 交互应用
            ("虚拟现实应用", 3.0),            # 虚拟现实应用
            ("娱乐应用", 3.0),                # 娱乐应用
        ],
        "negative_keywords": [
            ("general vision", 1.0),              # 通用视觉
            ("foundation model", 0.8),            # 基础模型
        ]
    },
    
    # 基础研究类别关键词
    # 5. 视觉表征学习与大模型
    "视觉表征学习与大模型 (Visual Representation Learning & Foundation Models)": {
        "keywords": [
            # 视觉大模型与涌现能力（高权重）
            ("vision foundation model", 2.8),      # 视觉基础模型
            ("large-scale pretrained model", 2.8), # 大规模预训练模型
            ("foundation model", 2.8),             # 基础模型
            ("emergent ability", 2.8),            # 涌现能力
            ("scaling law", 2.8),                 # 缩放定律
            ("large vision model", 2.8),          # 大视觉模型
            
            # 视觉表征与自监督学习（高权重）
            ("visual representation", 2.8),       # 视觉表征
            ("self-supervised learning", 2.8),     # 自监督学习
            ("contrastive learning", 2.8),        # 对比学习
            ("masked autoencoder", 2.8),          # 掉码自编码器
            ("representation learning", 2.8),      # 表征学习
            
            # 视觉架构与混合专家（高权重）
            ("vision transformer", 2.8),           # 视觉transformer
            ("transformer architecture", 2.8),      # transformer架构
            ("mixture of experts", 2.8),          # 混合专家
            ("MoE", 2.8),                         # MoE
            ("visual architecture", 2.8),          # 视觉架构
            
            # 基础研究特征关键词（高权重）
            ("proposed method", 5.0),              # 提出的方法
            ("novel approach", 5.0),               # 新型方法
            ("proposed framework", 5.0),           # 提出的框架
            ("can be applied to", 5.0),            # 可应用于
            ("applicable to", 5.0),                # 适用于
            ("general purpose", 5.0),              # 通用目的
            ("general method", 5.0),               # 通用方法
            ("generic approach", 5.0),             # 通用方法
            
            # 中文关键词（高权重）
            ("大模型", 2.0),                      # 大模型
            ("基础模型", 2.0),                  # 基础模型
            ("自监督", 2.0),                    # 自监督
            ("表征学习", 2.0),                  # 表征学习
            ("涌现能力", 2.0),                  # 涌现能力
            ("提出方法", 2.5),                  # 提出方法
            ("新型方法", 2.5),                  # 新型方法
            ("通用方法", 2.5),                  # 通用方法
            ("可应用于", 2.5),                  # 可应用于
        ],
        "negative_keywords": [
            ("specific application", 2.0),         # 特定应用
            ("downstream task", 2.0),              # 下游任务
            ("in medical", 2.0),                   # 在医疗中
            ("in industrial", 2.0),                # 在工业中
            ("in transportation", 2.0),            # 在交通中
            ("in remote sensing", 2.0),            # 在遥感中
            ("application scenario", 2.0),         # 应用场景
            ("application case", 2.0),             # 应用案例
            ("medical application", 2.0),          # 医疗应用
            ("industrial application", 2.0),       # 工业应用
            ("transportation application", 2.0),   # 交通应用
            ("remote sensing application", 2.0),   # 遥感应用
            ("应用场景", 2.0),                  # 应用场景
            ("医疗应用", 2.0),                  # 医疗应用
            ("工业应用", 2.0),                  # 工业应用
            ("交通应用", 2.0),                  # 交通应用
            ("遥感应用", 2.0),                  # 遥感应用
        ]
    },
    
    # 6. 静态图像理解与语义解析
    "静态图像理解与语义解析 (Static Image Understanding & Semantic Analysis)": {
        "keywords": [
            # 目标检测与定位（高权重）
            ("object detection", 2.5),             # 目标检测
            ("object localization", 2.5),          # 目标定位
            ("detection transformer", 2.5),        # 检测 transformer
            
            # 图像分类与识别（高权重）
            ("image classification", 2.5),         # 图像分类
            ("image recognition", 2.5),            # 图像识别
            
            # 语义/实例分割（高权重）
            ("semantic segmentation", 2.5),        # 语义分割
            ("instance segmentation", 2.5),        # 实例分割
            ("panoptic segmentation", 2.5),        # 全景分割
            ("segmentation transformer", 2.5),     # 分割 transformer
            
            # 关键点检测与姿态估计（高权重）
            ("keypoint detection", 2.5),           # 关键点检测
            ("pose estimation", 2.5),              # 姿态估计
            
            # 基础研究特征关键词（高权重）
            ("proposed method", 5.0),              # 提出的方法
            ("novel approach", 5.0),               # 新型方法
            ("proposed framework", 5.0),           # 提出的框架
            ("can be applied to", 5.0),            # 可应用于
            ("applicable to", 5.0),                # 适用于
            ("general purpose", 5.0),              # 通用目的
            ("general method", 5.0),               # 通用方法
            ("generic approach", 5.0),             # 通用方法
            
            # 中文基础研究关键词（高权重）
            ("目标检测", 2.5),                  # 目标检测
            ("图像分类", 2.5),                  # 图像分类
            ("语义分割", 2.5),                  # 语义分割
            ("实例分割", 2.5),                  # 实例分割
            ("提出方法", 3.0),                  # 提出方法
            ("VQA", 2.5),                           # VQA
            ("image captioning", 2.5),              # 图像描述
            ("visual reasoning", 2.5),              # 视觉推理
            ("visual grounding", 2.5),              # 视觉定位
            
            # 其他前沿视觉理论
            ("vision theory", 2.0),                  # 视觉理论
            ("computational vision", 2.0),           # 计算视觉
            ("visual science", 2.0),                 # 视觉科学
            ("vision science", 2.0),                 # 视觉科学
            ("visual cognition", 2.0),               # 视觉认知
            ("visual perception", 2.0),              # 视觉感知
            ("视觉理论", 2.0),                  # 视觉理论
            ("计算视觉", 2.0),                  # 计算视觉
            ("视觉科学", 2.0),                  # 视觉科学
            ("视觉认知", 2.0),                  # 视觉认知
            ("跨学科融合", 2.0),                # 跨学科融合
            
            # 基础研究特征关键词（高权重）
            ("proposed method", 5.0),              # 提出的方法
            ("novel approach", 5.0),               # 新型方法
            ("proposed framework", 5.0),           # 提出的框架
            ("can be applied to", 5.0),            # 可应用于
            ("applicable to", 5.0),                # 适用于
            ("general purpose", 5.0),              # 通用目的
            ("general method", 5.0),               # 通用方法
            ("generic approach", 5.0),             # 通用方法
            
            # 中文基础研究关键词（高权重）
            ("CLIP", 2.0),                         # CLIP
            ("多模态模型", 2.0),                # 多模态模型
            ("视觉语言模型", 2.0),              # 视觉语言模型
            ("提出方法", 2.5),                  # 提出方法
            ("新型方法", 2.5),                  # 新型方法
            ("通用方法", 2.5),                  # 通用方法
            ("可应用于", 2.5),                  # 可应用于
        ],
        "negative_keywords": [
            ("unimodal", 2.0),                      # 单模态（提升惩罚）
            ("single modality", 2.0),               # 单一模态
            ("pure vision", 2.0),                   # 纯视觉
            ("specific application", 2.0),         # 特定应用
            ("downstream task", 2.0),              # 下游任务
            ("in medical", 2.0),                   # 在医疗中
            ("in industrial", 2.0),                # 在工业中
            ("in transportation", 2.0),            # 在交通中
            ("in remote sensing", 2.0),            # 在遥感中
            ("application scenario", 2.0),         # 应用场景
            ("application case", 2.0),             # 应用案例
            ("medical application", 2.0),          # 医疗应用
            ("industrial application", 2.0),       # 工业应用
            ("transportation application", 2.0),   # 交通应用
            ("remote sensing application", 2.0),   # 遥感应用
            ("应用场景", 2.0),                  # 应用场景
            ("医疗应用", 2.0),                  # 医疗应用
            ("工业应用", 2.0),                  # 工业应用
            ("交通应用", 2.0),                  # 交通应用
            ("遥感应用", 2.0),                  # 遥感应用
            ("text-to-image generation", 1.0),      # 文本生成图像（应归为生成式视觉）
        ]
    },
    
    # 11. 模型优化与系统鲁棒性
    "模型优化与系统鲁棒性 (Model Optimization & System Robustness)": {
        "keywords": [
            # 模型压缩与加速（高权重）
            ("model compression", 2.0),             # 模型压缩
            ("model acceleration", 2.0),            # 模型加速
            ("knowledge distillation", 1.8),        # 知识蒸馏
            
            # 神经架构优化（高权重）
            ("neural architecture", 2.0),           # 神经架构
            ("architecture optimization", 2.0),      # 架构优化
            ("NAS", 1.8),                           # 神经架构搜索
            
            # 资源受限视觉计算（高权重）
            ("resource-constrained", 2.0),          # 资源受限
            ("efficient inference", 2.0),           # 高效推理
            
            # 推理优化（高权重）
            ("inference optimization", 2.0),         # 推理优化
            ("quantization", 2.0),                  # 量化
            ("pruning", 2.0),                       # 剪枝
            
            # 对抗鲁棒性（高权重）
            ("adversarial robustness", 2.0),        # 对抗鲁棒性
            ("adversarial attack", 2.0),            # 对抗攻击
            ("adversarial defense", 2.0),           # 对抗防御
            
            # 分布外泛化（高权重）
            ("out-of-distribution", 2.0),          # 分布外
            ("distribution generalization", 2.0),    # 分布泛化
            ("domain generalization", 1.8),         # 域泛化
            
            # 不确定性量化（高权重）
            ("uncertainty quantification", 2.0),     # 不确定性量化
            ("uncertainty estimation", 2.0),         # 不确定性估计
            
            # 视觉安全与隐私（高权重）
            ("visual security", 2.0),               # 视觉安全
            ("visual privacy", 2.0),                # 视觉隐私
        ],
        "negative_keywords": [
            ("accuracy only", 1.0),                 # 仅精度
            ("theoretical", 0.8),                   # 理论的
        ]
    },
    
    # 12. 效率学习与适应性智能
    "效率学习与适应性智能 (Efficient Learning & Adaptive Intelligence)": {
        "keywords": [
            # 小样本与零样本学习（高权重）
            ("few-shot learning", 2.5),              # 小样本学习
            ("one-shot learning", 2.5),              # 单样本学习
            ("zero-shot learning", 2.5),             # 零样本学习
            ("low-shot learning", 2.5),              # 低样本学习
            ("sample-efficient learning", 2.5),      # 样本高效学习
            
            # 迁移学习与域适应（高权重）
            ("transfer learning", 2.5),              # 迁移学习
            ("domain adaptation", 2.5),              # 域适应
            ("domain generalization", 2.5),          # 域泛化
            ("cross-domain", 2.5),                   # 跨域
            ("unsupervised domain adaptation", 2.5),  # 无监督域适应
            
            # 持续学习与元学习（高权重）
            ("continual learning", 2.0),             # 持续学习
            ("lifelong learning", 2.0),              # 终身学习
            ("incremental learning", 2.0),           # 增量学习
            ("meta-learning", 2.0),                  # 元学习
            ("learning to learn", 2.0),              # 学会学习
            
            # 其他效率学习技术
            ("semi-supervised learning", 2.0),       # 半监督学习
            ("weakly-supervised learning", 2.0),     # 弱监督学习
            ("label-efficient learning", 2.0),       # 标签高效学习
            ("active learning", 2.0),                # 主动学习
            ("小样本", 2.0),                    # 小样本
            ("零样本", 2.0),                    # 零样本
            ("迁移学习", 2.0),                  # 迁移学习
            ("域适应", 2.0),                    # 域适应
            ("持续学习", 2.0),                  # 持续学习
            ("元学习", 2.0),                    # 元学习
            
            # 基础研究特征关键词（高权重）
            ("proposed method", 5.0),              # 提出的方法
            ("novel approach", 5.0),               # 新型方法
            ("proposed framework", 5.0),           # 提出的框架
            ("can be applied to", 5.0),            # 可应用于
            ("applicable to", 5.0),                # 适用于
            ("general purpose", 5.0),              # 通用目的
            ("general method", 5.0),               # 通用方法
            ("generic approach", 5.0),             # 通用方法
            
            # 中文基础研究关键词（高权重）
            ("小样本学习", 2.5),                # 小样本学习
            ("迁移学习方法", 2.5),              # 迁移学习方法
            ("元学习框架", 2.5),                # 元学习框架
            ("提出方法", 3.0),                  # 提出方法
            ("新型方法", 3.0),                  # 新型方法
            ("通用方法", 3.0),                  # 通用方法
            ("可应用于", 3.0),                  # 可应用于
            ("理论框架", 3.0),                  # 理论框架
            ("理论分析", 3.0),                  # 理论分析
            ("我们提出", 3.0),                  # 我们提出
            ("算法设计", 2.5),                  # 算法设计
            ("基础研究", 2.5),                  # 基础研究
            ("基本原理", 2.5),                  # 基本原理
        ],
        "negative_keywords": [
            ("fully supervised", 1.2),               # 全监督（提升惩罚）
            ("large dataset", 1.0),                  # 大数据集
            ("massive training data", 1.0),          # 大量训练数据
            ("specific application", 1.0),         # 特定应用
            ("downstream task", 1.0),              # 下游任务
            ("in medical", 1.0),                   # 在医疗中
            ("in industrial", 1.0),                # 在工业中
            ("in transportation", 1.0),            # 在交通中
            ("in remote sensing", 1.0),            # 在遥感中
            ("application scenario", 1.0),         # 应用场景
            ("application case", 1.0),             # 应用案例
            ("medical application", 1.2),          # 医疗应用
            ("industrial application", 1.2),       # 工业应用
            ("transportation application", 1.2),   # 交通应用
            ("remote sensing application", 1.2),   # 遥感应用
            ("应用场景", 1.0),                  # 应用场景
            ("医疗应用", 1.2),                  # 医疗应用
            ("工业应用", 1.2),                  # 工业应用
            ("交通应用", 1.2),                  # 交通应用
            ("遥感应用", 1.2),                  # 遥感应用
        ]
    },
    
    # 13. 感知-动作智能与主动视觉
    "感知-动作智能与主动视觉 (Perception-Action Intelligence & Active Vision)": {
        "keywords": [
            # 具身视觉与机器人感知（高权重）
            ("embodied vision", 2.0),                # 具身视觉
            ("embodied AI", 2.0),                    # 具身人工智能
            ("robotic vision", 2.0),                 # 机器人视觉
            ("robot perception", 2.0),               # 机器人感知
            ("robot manipulation", 2.0),             # 机器人操作
            
            # 视觉导航与主动探索（高权重）
            ("visual navigation", 2.0),              # 视觉导航
            ("active exploration", 2.0),             # 主动探索
            ("active vision", 2.0),                  # 主动视觉
            ("visual attention", 2.0),               # 视觉注意力
            ("gaze control", 2.0),                   # 视线控制
            
            # 感知-动作循环与交互（高权重）
            ("perception-action loop", 2.0),         # 感知-动作循环
            ("sensorimotor learning", 2.0),          # 感知运动学习
            ("interactive perception", 2.0),         # 交互式感知
            ("visual control", 2.0),                 # 视觉控制
            ("vision-based control", 2.0),           # 基于视觉的控制
            ("visual servoing", 2.0),                # 视觉伴随控制
            ("具身视觉", 2.0),                  # 具身视觉
            ("机器人视觉", 2.0),                # 机器人视觉
            ("视觉导航", 2.0),                  # 视觉导航
            ("主动视觉", 2.0),                  # 主动视觉
            ("感知动作", 2.0),                  # 感知动作
            
            # 基础研究特征关键词（高权重）
            ("proposed method", 3.0),              # 提出的方法
            ("novel approach", 3.0),               # 新型方法
            ("proposed framework", 3.0),           # 提出的框架
            ("can be applied to", 3.0),            # 可应用于
            ("applicable to", 3.0),                # 适用于
            ("general purpose", 3.0),              # 通用目的
            ("general method", 3.0),               # 通用方法
            ("generic approach", 3.0),             # 通用方法
            ("theoretical", 3.0),                  # 理论的
            ("theoretical framework", 3.0),        # 理论框架
            ("theoretical analysis", 3.0),         # 理论分析
            ("algorithm", 2.5),                    # 算法
            ("methodology", 2.5),                  # 方法论
            ("we propose", 3.0),                   # 我们提出
            ("we present", 3.0),                   # 我们呈现
            ("we introduce", 3.0),                 # 我们引入
            ("foundation", 2.5),                   # 基础
            ("foundational", 2.5),                 # 基础的
            ("fundamental", 2.5),                  # 基本的
            ("generalizable", 3.0),                # 可泛化的
            ("universal", 2.5),                    # 通用的
            
            # 中文基础研究关键词（高权重）
            ("具身视觉方法", 2.5),              # 具身视觉方法
            ("主动视觉框架", 2.5),              # 主动视觉框架
            ("感知动作循环", 2.5),              # 感知动作循环
            ("提出方法", 3.0),                  # 提出方法
            ("新型方法", 3.0),                  # 新型方法
            ("通用方法", 3.0),                  # 通用方法
            ("可应用于", 3.0),                  # 可应用于
            ("理论框架", 3.0),                  # 理论框架
            ("理论分析", 3.0),                  # 理论分析
            ("我们提出", 3.0),                  # 我们提出
            ("算法设计", 2.5),                  # 算法设计
            ("基础研究", 2.5),                  # 基础研究
            ("基本原理", 2.5),                  # 基本原理
        ],
        "negative_keywords": [
            ("passive vision", 1.2),                 # 被动视觉（提升惩罚）
            ("static analysis", 1.2),                # 静态分析
            ("pure recognition", 1.2),               # 纯识别
            ("autonomous driving", 1.2),             # 自动驾驶（应归为智能交通）
            
            ("specific application", 1.0),         # 特定应用
            ("downstream task", 1.0),              # 下游任务
            ("in medical", 1.0),                   # 在医疗中
            ("in industrial", 1.0),                # 在工业中
            ("in transportation", 1.0),            # 在交通中
            ("in remote sensing", 1.0),            # 在遥感中
            ("application scenario", 1.0),         # 应用场景
            ("application case", 1.0),             # 应用案例
            ("medical application", 1.2),          # 医疗应用
            ("industrial application", 1.2),       # 工业应用
            ("transportation application", 1.2),   # 交通应用
            ("remote sensing application", 1.2),   # 遥感应用
            ("应用场景", 1.0),                  # 应用场景
            ("医疗应用", 1.2),                  # 医疗应用
            ("工业应用", 1.2),                  # 工业应用
            ("交通应用", 1.2),                  # 交通应用
            ("遥感应用", 1.2),                  # 遥感应用
        ]
    },
    
    # 14. 前沿视觉理论与跨学科融合
    "前沿视觉理论与跨学科融合 (Frontier Vision Theory & Interdisciplinary Integration)": {
        "keywords": [
            # 神经-符号视觉（高权重）
            ("neuro-symbolic vision", 2.0),           # 神经-符号视觉
            ("symbolic reasoning", 2.0),              # 符号推理
            ("neural-symbolic", 1.8),                 # 神经-符号
            
            # 视觉认知计算（高权重）
            ("visual cognitive computing", 2.0),      # 视觉认知计算
            ("cognitive vision", 2.0),                # 认知视觉
            ("brain-inspired vision", 1.8),           # 脑启发视觉
            
            # 量子视觉算法（高权重）
            ("quantum visual algorithm", 2.0),        # 量子视觉算法
            ("quantum vision", 2.0),                  # 量子视觉
            
            # 可解释视觉智能（高权重）
            ("explainable visual intelligence", 2.0), # 可解释视觉智能
            ("explainable vision", 2.0),              # 可解释视觉
            ("interpretable vision", 1.8),            # 可解释视觉
            
            # 其他前沿视觉理论
            ("vision theory", 2.0),                  # 视觉理论
            ("computational vision", 2.0),           # 计算视觉
            ("visual science", 2.0),                 # 视觉科学
            ("vision science", 2.0),                 # 视觉科学
            ("visual cognition", 2.0),               # 视觉认知
            ("visual perception", 2.0),              # 视觉感知
            ("视觉理论", 2.0),                  # 视觉理论
            ("计算视觉", 2.0),                  # 计算视觉
            ("视觉科学", 2.0),                  # 视觉科学
            ("视觉认知", 2.0),                  # 视觉认知
            ("跨学科融合", 2.0),                # 跨学科融合
            
            # 基础研究特征关键词（高权重）
            ("proposed method", 2.5),              # 提出的方法
            ("novel approach", 2.5),               # 新型方法
            ("proposed framework", 2.5),           # 提出的框架
            ("can be applied to", 2.5),            # 可应用于
            ("applicable to", 2.5),                # 适用于
            ("general purpose", 2.5),              # 通用目的
            ("general method", 2.5),               # 通用方法
            ("generic approach", 2.5),             # 通用方法
            
            # 中文基础研究关键词（高权重）
            ("理论模型", 2.0),                  # 理论模型
            ("计算视觉框架", 2.0),              # 计算视觉框架
            ("视觉科学方法", 2.0),              # 视觉科学方法
            ("提出方法", 2.5),                  # 提出方法
            ("新型方法", 2.5),                  # 新型方法
            ("通用方法", 2.5),                  # 通用方法
            ("可应用于", 2.5),                  # 可应用于
        ],
        "negative_keywords": [
            ("traditional vision", 1.2),               # 传统视觉
            ("standard approach", 1.2),               # 标准方法
            ("specific application", 1.0),            # 特定应用
            ("downstream task", 1.0),              # 下游任务
            ("in medical", 1.0),                   # 在医疗中
            ("in industrial", 1.0),                # 在工业中
            ("in transportation", 1.0),            # 在交通中
            ("in remote sensing", 1.0),            # 在遥感中
            ("application scenario", 1.0),         # 应用场景
            ("application case", 1.0),             # 应用案例
            ("medical application", 1.2),          # 医疗应用
            ("industrial application", 1.2),       # 工业应用
            ("transportation application", 1.2),   # 交通应用
            ("remote sensing application", 1.2),   # 遥感应用
            ("应用场景", 1.0),                  # 应用场景
            ("医疗应用", 1.2),                  # 医疗应用
            ("工业应用", 1.2),                  # 工业应用
            ("交通应用", 1.2),                  # 交通应用
            ("遥感应用", 1.2),                  # 遥感应用
        ]
    },
    
    # 其他类别 - 保持原始阈值
    "其他 (Others)": {
        "threshold": 0.5,  # 恢复到原始阈值
        "subcategories": {},
        "priority": 0.5,
        "keywords": [
            # 不属于上述类别的论文（进一步降低权重）
            ("miscellaneous", 0.6),                # 杂项（显著降低权重）
            ("other", 0.6),                        # 其他（显著降低权重）
            ("novel approach", 0.6),               # 新型方法（显著降低权重）
        ],
        "negative_keywords": [
            # 显著增强所有主要类别的关键词的负面权重
            ("foundation model", 2.0),             # 基础模型
            ("detection", 2.0),                    # 检测
            ("segmentation", 2.0),                 # 分割
            ("generative", 2.0),                   # 生成式
            ("3D", 2.0),                           # 3D
            ("temporal", 2.0),                     # 时序
            ("self-supervised", 2.0),              # 自监督
            ("efficiency", 2.0),                   # 效率
            ("robustness", 2.0),                   # 鲁棒性
            ("few-shot", 2.0),                     # 小样本
            ("embodied", 2.0),                     # 具身
            ("vision-language", 2.0),              # 视觉-语言
            ("application", 2.0),                  # 应用
            ("interdisciplinary", 2.0),            # 跨学科
            ("medical", 2.0),                      # 医学
            ("autonomous", 2.0),                   # 自主
            ("industrial", 2.0),                   # 工业
            ("AR", 2.0),                           # AR
            ("VR", 2.0),                           # VR
            ("proposed for", 2.0),                 # 提出用于（基础研究特征）
            ("can be applied", 2.0),               # 可应用于（基础研究特征）
        ]
    }
}