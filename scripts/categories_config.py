"""
计算机视觉论文分类配置文件
"""

# 类别阈值配置
CATEGORY_THRESHOLDS = {
    "3D场景": 1.1,      # 降低阈值，更容易匹配
    "神经渲染": 1.1,    # 降低阈值，更容易匹配
    "3DGS": 1.1,       # 降低阈值，更容易匹配
    "生成模型": 1.1,    # 降低阈值，更容易匹配
    "多模态": 1.1,      # 降低阈值，更容易匹配
    "检测分割": 1.0,    # 保持较低阈值
    "图像理解": 1.0,    # 保持较低阈值
    "视频理解": 1.0,    # 保持较低阈值
    "图像处理": 1.0,    # 保持较低阈值
    "人体分析": 1.2,    # 稍微降低阈值
    "人脸技术": 1.2,    # 稍微降低阈值
    "数字人": 1.2,      # 稍微降低阈值
    "模型优化": 1.0,    # 降低阈值，更容易匹配
    "具身智能": 1.0,    # 降低阈值，更容易匹配
    "扩散桥": 1.0,      # 降低阈值，更容易匹配
    "流模型": 1.0,      # 降低阈值，更容易匹配
    "其他": 1.8,       # 提高阈值，更难匹配
}

# 类别关键词配置
CATEGORY_KEYWORDS = {
    # 3D场景（传统3D重建、SLAM、多视图几何）
    "3D场景": {
        "keywords": [
            # 3D重建核心技术（高权重）
            ("3d reconstruction", 1.8),           # 3D重建（核心）
            ("structure from motion", 1.8),       # 运动恢复结构（核心）
            ("multi view stereo", 1.8),          # 多视图立体视觉（核心）
            ("visual slam", 1.8),                # 视觉SLAM（核心）
            ("simultaneous localization and mapping", 1.8),  # SLAM（核心）
            
            # 几何重建技术（中等权重）
            ("point cloud reconstruction", 1.6),  # 点云重建
            ("mesh reconstruction", 1.6),         # 网格重建
            ("surface reconstruction", 1.6),      # 表面重建
            ("dense reconstruction", 1.6),        # 稠密重建
            ("geometric reconstruction", 1.6),    # 几何重建
            
            # 相机定位和跟踪（中等权重）
            ("camera pose estimation", 1.6),      # 相机姿态估计
            ("visual odometry", 1.6),            # 视觉里程计
            ("camera tracking", 1.6),            # 相机跟踪
            ("visual localization", 1.6),        # 视觉定位
            ("bundle adjustment", 1.6),          # 光束法平差
            
            # 3D表示和处理（较低权重）
            ("point cloud processing", 1.4),      # 点云处理
            ("mesh processing", 1.4),             # 网格处理
            ("3d feature", 1.4),                 # 3D特征
            ("depth estimation", 1.4),           # 深度估计
            ("3d registration", 1.4),            # 3D配准
        ],
        "negative_keywords": [
            # 神经渲染（严格排除）
            "neural radiance fields",
            "nerf",
            "neural rendering",
            "implicit neural representation",
            "neural implicit surface",
            
            # 3DGS（严格排除）
            "gaussian splatting",
            "3d gaussian",
            "gaussian representation",
            "splat rendering",
            "gaussian primitive",
            
            # 生成模型（避免混淆）
            "text to 3d",
            "3d generation",
            "3d synthesis",
            "3d gan",
            "3d diffusion",
            
            # 数字人（避免混淆）
            "digital human",
            "virtual human",
            "human avatar",
            "character animation",
            "human modeling",
            
            # 通用视觉任务（避免过度匹配）
            "object detection",
            "semantic segmentation",
            "instance segmentation",
            "scene understanding",
            "image classification"
        ]
    },

    # 神经渲染（NeRF及神经隐式表示）
    "神经渲染": {
        "keywords": [
            # NeRF核心技术（高权重）
            ("neural radiance fields", 1.8),      # 神经辐射场（核心）
            ("nerf", 1.8),                        # NeRF（核心）
            ("neural rendering", 1.8),            # 神经渲染（核心）
            ("neural field", 1.8),                # 神经场（核心）
            ("implicit neural representation", 1.8), # 神经隐式表示（核心）
            
            # 神经表示变体（中等权重）
            ("neural surface", 1.6),              # 神经表面
            ("neural volume", 1.6),               # 神经体积
            ("neural implicit", 1.6),             # 神经隐式
            ("neural scene representation", 1.6),  # 神经场景表示
            ("neural geometry", 1.6),             # 神经几何
            
            # 渲染和优化（中等权重）
            ("volume rendering", 1.6),            # 体渲染
            ("ray marching", 1.6),                # 光线步进
            ("radiance field", 1.6),              # 辐射场
            ("neural reconstruction", 1.6),        # 神经重建
            ("view synthesis", 1.6),              # 视图合成
            
            # 特定应用（较低权重）
            ("dynamic nerf", 1.4),                # 动态NeRF
            ("deformable nerf", 1.4),             # 可变形NeRF
            ("instant ngp", 1.4),                 # 即时NGP
            ("neural light field", 1.4),          # 神经光场
            ("neural reflectance", 1.4),          # 神经反射
        ],
        "negative_keywords": [
            # 传统3D（严格排除）
            "structure from motion",
            "multi view stereo",
            "visual slam",
            "point cloud reconstruction",
            "mesh reconstruction",
            
            # 3DGS（严格排除）
            "gaussian splatting",
            "3d gaussian",
            "gaussian representation",
            "splat rendering",
            "gaussian primitive",
            
            # 生成模型（避免混淆）
            "text to 3d",
            "3d generation",
            "3d synthesis",
            "3d gan",
            "3d diffusion",
            
            # 通用渲染（避免混淆）
            "rasterization",
            "ray tracing",
            "path tracing",
            "photon mapping",
            "global illumination",
            
            # 通用视觉任务（避免过度匹配）
            "object detection",
            "semantic segmentation",
            "instance segmentation",
            "scene understanding",
            "image classification"
        ]
    },

    # 3D高斯散射（3D Gaussian Splatting）
    "3DGS": {
        "keywords": [
            # 3DGS核心技术（高权重）
            ("gaussian splatting", 1.8),          # 高斯散射（核心）
            ("3d gaussian", 1.8),                 # 3D高斯（核心）
            ("gaussian representation", 1.8),      # 高斯表示（核心）
            ("gaussian primitive", 1.8),           # 高斯基元（核心）
            ("splat rendering", 1.8),             # 散射渲染（核心）
            
            # 渲染和优化（中等权重）
            ("gaussian rendering", 1.6),          # 高斯渲染
            ("gaussian optimization", 1.6),       # 高斯优化
            ("gaussian field", 1.6),             # 高斯场
            ("gaussian scene", 1.6),             # 高斯场景
            ("point based rendering", 1.6),      # 基于点的渲染
            
            # 特定应用（较低权重）
            ("dynamic gaussian", 1.4),           # 动态高斯
            ("deformable gaussian", 1.4),        # 可变形高斯
            ("gaussian neural", 1.4),            # 高斯神经
            ("gaussian based", 1.4),             # 基于高斯
            ("gaussian model", 1.4),             # 高斯模型
        ],
        "negative_keywords": [
            # 传统3D（严格排除）
            "structure from motion",
            "multi view stereo",
            "visual slam",
            "point cloud reconstruction",
            "mesh reconstruction",
            
            # NeRF（严格排除）
            "neural radiance fields",
            "nerf",
            "neural rendering",
            "implicit neural representation",
            "neural implicit surface",
            
            # 生成模型（避免混淆）
            "text to 3d",
            "3d generation",
            "3d synthesis",
            "3d gan",
            "3d diffusion",
            
            # 统计模型（避免混淆）
            "gaussian mixture model",
            "gaussian process",
            "gaussian distribution",
            "gaussian noise",
            "gaussian filter",
            
            # 通用视觉任务（避免过度匹配）
            "object detection",
            "semantic segmentation",
            "instance segmentation",
            "scene understanding",
            "image classification"
        ]
    },

    # 生成模型（扩散模型、GAN、文生图）
    "生成模型": {
        "keywords": [
            # 核心生成模型（高权重）
            ("diffusion model", 1.8),              # 扩散模型
            ("diffusion bridge", 1.8),             # 扩散桥
            ("diffusion process", 1.8),            # 扩散过程
            ("score matching", 1.8),               # 分数匹配
            ("score based", 1.8),                  # 基于分数
            ("score function", 1.8),               # 分数函数
            ("probability flow", 1.8),             # 概率流
            ("stochastic differential", 1.8),      # 随机微分
            ("sde", 1.8),                         # SDE
            ("langevin dynamics", 1.8),           # 朗之万动力学
            ("stable diffusion", 1.8),             # stable diffusion
            ("latent diffusion", 1.8),             # 潜空间扩散
            ("generative model", 1.8),             # 生成模型
            ("text to image", 1.8),                # 文本到图像
            ("image generation", 1.8),             # 图像生成
            ("gan", 1.8),                          # GAN
            ("generative adversarial", 1.8),       # 生成对抗
            ("vae", 1.8),                          # VAE
            ("variational autoencoder", 1.8),      # 变分自编码器
            
            # 扩散相关技术（中等权重）
            ("noise prediction", 1.5),             # 噪声预测
            ("noise estimation", 1.5),             # 噪声估计
            ("denoising", 1.5),                    # 去噪
            ("sampling strategy", 1.5),            # 采样策略
            ("reverse process", 1.5),              # 逆过程
            ("forward process", 1.5),              # 前向过程
            ("markov chain", 1.5),                # 马尔可夫链
            ("discretization", 1.5),              # 离散化
            ("continuous time", 1.5),             # 连续时间
            
            # 具体生成任务（中等权重）
            ("image synthesis", 1.5),              # 图像合成
            ("image editing", 1.5),                # 图像编辑
            ("text guided", 1.5),                  # 文本引导
            ("style transfer", 1.5),               # 风格迁移
            ("inpainting", 1.5),                   # 图像修复
            ("outpainting", 1.5),                  # 图像扩展
            
            # 生成相关技术（较低权重）
            ("conditional generation", 1.2),        # 条件生成
            ("unconditional generation", 1.2),      # 无条件生成
            ("adversarial training", 1.2),          # 对抗训练
            ("likelihood", 1.2),                    # 似然
            ("probability density", 1.2),           # 概率密度
            ("energy based", 1.2),                  # 基于能量
        ],
        "negative_keywords": [
            # 排除传统视觉任务
            "detection",
            "segmentation",
            "classification",
            "recognition",
            "tracking",
            "estimation",
            
            # 排除3D相关
            "3d reconstruction",
            "nerf",
            "neural rendering",
            "point cloud",
            "mesh",
            "gaussian splatting",
            
            # 排除图像处理
            "super resolution",
            "restoration",
            "enhancement",
            "quality assessment",
            "compression",
            
            # 排除多模态（除非明确是生成任务）
            "multimodal",
            "cross modal",
            "vision language",
            "visual question",
            "image captioning",
            
            # 排除具身智能
            "robot",
            "manipulation",
            "navigation",
            "planning",
            "control",
            
            # 排除通用术语
            "feature extraction",
            "representation learning",
            "self supervised",
            "semi supervised",
            "few shot",
            "zero shot",
            "transfer learning",
            
            # 排除数据处理
            "data augmentation",
            "data processing",
            "dataset",
            "benchmark",
            "evaluation",
            
            # 排除优化和训练
            "optimization",
            "regularization",
            "loss function",
            "training strategy",
            "fine tuning",
            
            # 排除分析和理论
            "analysis",
            "theoretical",
            "empirical",
            "study",
            "survey",
            "review"
        ]
    },

    # 多模态（跨模态学习和对齐）
    "多模态": {
        "keywords": [
            # 多模态学习核心（高权重）
            ("multimodal learning", 1.8),          # 多模态学习（核心）
            ("cross modal learning", 1.8),         # 跨模态学习（核心）
            ("multimodal representation", 1.8),    # 多模态表示（核心）
            ("multimodal alignment", 1.8),         # 多模态对齐（核心）
            ("multimodal fusion", 1.8),            # 多模态融合（核心）
            
            # 模态对齐和匹配（中等权重）
            ("modality alignment", 1.6),           # 模态对齐
            ("cross modal matching", 1.6),         # 跨模态匹配
            ("modality fusion", 1.6),              # 模态融合
            ("joint embedding", 1.6),              # 联合嵌入
            ("semantic alignment", 1.6),           # 语义对齐
            
            # 多模态理解（中等权重）
            ("multimodal understanding", 1.6),     # 多模态理解
            ("vision language", 1.6),              # 视觉语言
            ("visual semantic", 1.6),              # 视觉语义
            ("cross modal retrieval", 1.6),        # 跨模态检索
            ("multimodal embedding", 1.6),         # 多模态嵌入
            
            # 特定应用（较低权重）
            ("image text matching", 1.4),          # 图文匹配
            ("video text matching", 1.4),          # 视频文本匹配
            ("audio visual", 1.4),                 # 视听
            ("visual question answering", 1.4),    # 视觉问答
            ("visual grounding", 1.4),             # 视觉定位
        ],
        "negative_keywords": [
            # 生成模型（严格排除）
            "text to image",
            "image generation",
            "video generation",
            "image synthesis",
            "video synthesis",
            
            # 3D生成（严格排除）
            "text to 3d",
            "3d generation",
            "3d synthesis",
            "mesh generation",
            "point cloud generation",
            
            # 单模态任务（避免混淆）
            "image classification",
            "object detection",
            "semantic segmentation",
            "action recognition",
            "pose estimation",
            
            # 特定领域（避免混淆）
            "neural rendering",
            "neural radiance fields",
            "digital human",
            "virtual human",
            "human avatar",
            
            # 基础模型（避免过度匹配）
            "neural network",
            "deep learning",
            "machine learning",
            "artificial intelligence",
            "backbone network"
        ]
    },

    # 检测分割（目标检测和分割任务）
    "检测分割": {
        "keywords": [
            # 目标检测核心（高权重）
            ("object detection model", 1.8),       # 目标检测模型（核心）
            ("object detector", 1.8),              # 目标检测器（核心）
            ("detection framework", 1.8),          # 检测框架（核心）
            ("detection architecture", 1.8),       # 检测架构（核心）
            ("detection network", 1.8),            # 检测网络（核心）
            
            # 语义分割（中等权重）
            ("semantic segmentation", 1.6),        # 语义分割
            ("instance segmentation", 1.6),        # 实例分割
            ("panoptic segmentation", 1.6),        # 全景分割
            ("segmentation model", 1.6),           # 分割模型
            ("segmentation network", 1.6),         # 分割网络
            
            # 检测技术（中等权重）
            ("anchor based", 1.6),                 # 基于锚框
            ("anchor free", 1.6),                  # 无锚框
            ("proposal generation", 1.6),          # 候选框生成
            ("bounding box", 1.6),                 # 边界框
            ("detection head", 1.6),               # 检测头
            
            # 特定应用（较低权重）
            ("salient object detection", 1.4),     # 显著性目标检测
            ("anomaly detection", 1.4),            # 异常检测
            ("edge detection", 1.4),               # 边缘检测
            ("boundary detection", 1.4),           # 边界检测
            ("keypoint detection", 1.4),           # 关键点检测
        ],
        "negative_keywords": [
            # 生成模型（严格排除）
            "text to image",
            "image generation",
            "video generation",
            "image synthesis",
            "video synthesis",
            
            # 3D任务（严格排除）
            "3d reconstruction",
            "point cloud",
            "neural rendering",
            "nerf",
            "gaussian splatting",
            
            # 人体/人脸（避免混淆）
            "human pose",
            "face recognition",
            "facial expression",
            "body parsing",
            "person reid",
            
            # 多模态（避免混淆）
            "multimodal",
            "cross modal",
            "vision language",
            "visual semantic",
            "joint embedding",
            
            # 图像处理（避免混淆）
            "image enhancement",
            "image restoration",
            "super resolution",
            "denoising",
            "deblurring"
        ]
    },

    # 图像理解（图像分类和场景理解）
    "图像理解": {
        "keywords": [
            # 图像分类核心（高权重）
            ("image classification model", 1.8),    # 图像分类模型（核心）
            ("visual recognition model", 1.8),      # 视觉识别模型（核心）
            ("classification network", 1.8),        # 分类网络（核心）
            ("recognition framework", 1.8),         # 识别框架（核心）
            ("image recognition system", 1.8),      # 图像识别系统（核心）
            
            # 场景理解（中等权重）
            ("scene understanding", 1.6),           # 场景理解
            ("scene recognition", 1.6),             # 场景识别
            ("scene parsing", 1.6),                 # 场景解析
            ("visual understanding", 1.6),          # 视觉理解
            ("scene classification", 1.6),          # 场景分类
            
            # 视觉特征（中等权重）
            ("feature extraction", 1.6),            # 特征提取
            ("visual feature", 1.6),                # 视觉特征
            ("feature representation", 1.6),        # 特征表示
            ("attention mechanism", 1.6),           # 注意力机制
            ("visual attention", 1.6),              # 视觉注意力
            
            # 特定应用（较低权重）
            ("fine grained recognition", 1.4),      # 细粒度识别
            ("zero shot recognition", 1.4),         # 零样本识别
            ("few shot recognition", 1.4),          # 小样本识别
            ("open set recognition", 1.4),          # 开集识别
            ("long tail recognition", 1.4),         # 长尾识别
        ],
        "negative_keywords": [
            # 检测分割（严格排除）
            "object detection",
            "semantic segmentation",
            "instance segmentation",
            "detection network",
            "segmentation model",
            
            # 生成模型（严格排除）
            "image generation",
            "image synthesis",
            "text to image",
            "diffusion model",
            "gan",
            
            # 3D任务（避免混淆）
            "3d reconstruction",
            "point cloud",
            "neural rendering",
            "nerf",
            "gaussian splatting",
            
            # 人体/人脸（避免混淆）
            "human pose",
            "face recognition",
            "facial expression",
            "body parsing",
            "person reid",
            
            # 图像处理（避免混淆）
            "image enhancement",
            "image restoration",
            "super resolution",
            "denoising",
            "deblurring"
        ]
    },

    # 视频理解（视频分析和动作识别）
    "视频理解": {
        "keywords": [
            # 动作识别核心（高权重）
            ("action recognition model", 1.8),      # 动作识别模型（核心）
            ("video action recognition", 1.8),      # 视频动作识别（核心）
            ("temporal action detection", 1.8),     # 时序动作检测（核心）
            ("action localization", 1.8),           # 动作定位（核心）
            ("action understanding", 1.8),          # 动作理解（核心）
            
            # 时序建模（中等权重）
            ("temporal modeling", 1.6),             # 时序建模
            ("temporal relation", 1.6),             # 时序关系
            ("temporal reasoning", 1.6),            # 时序推理
            ("motion pattern", 1.6),                # 运动模式
            ("motion feature", 1.6),                # 运动特征
            
            # 视频分析（中等权重）
            ("video understanding", 1.6),           # 视频理解
            ("video analysis", 1.6),                # 视频分析
            ("video recognition", 1.6),             # 视频识别
            ("video classification", 1.6),          # 视频分类
            ("video representation", 1.6),          # 视频表示
            
            # 特定应用（较低权重）
            ("gesture recognition", 1.4),           # 手势识别
            ("activity recognition", 1.4),          # 活动识别
            ("event detection", 1.4),               # 事件检测
            ("behavior analysis", 1.4),             # 行为分析
            ("interaction recognition", 1.4),       # 交互识别
        ],
        "negative_keywords": [
            # 生成模型（严格排除）
            "video generation",
            "video synthesis",
            "text to video",
            "video diffusion",
            "video gan",
            
            # 人体分析（严格排除）
            "human pose estimation",
            "human motion capture",
            "human action recognition",
            "human behavior analysis",
            "human skeleton tracking",
            
            # 3D任务（避免混淆）
            "3d reconstruction",
            "point cloud",
            "neural rendering",
            "nerf",
            "gaussian splatting",
            
            # 多模态（避免混淆）
            "multimodal",
            "cross modal",
            "vision language",
            "visual semantic",
            "joint embedding",
            
            # 视频处理（避免混淆）
            "video enhancement",
            "video restoration",
            "video super resolution",
            "video denoising",
            "video compression"
        ]
    },

    # 图像处理（图像增强和修复）
    "图像处理": {
        "keywords": [
            # 图像增强核心（高权重）
            ("image enhancement model", 1.8),       # 图像增强模型（核心）
            ("image restoration model", 1.8),       # 图像修复模型（核心）
            ("super resolution model", 1.8),        # 超分辨率模型（核心）
            ("image quality enhancement", 1.8),     # 图像质量增强（核心）
            ("image reconstruction model", 1.8),    # 图像重建模型（核心）
            
            # 图像修复（中等权重）
            ("image denoising", 1.6),              # 图像去噪
            ("image deblurring", 1.6),             # 图像去模糊
            ("image inpainting", 1.6),             # 图像修复
            ("image completion", 1.6),             # 图像补全
            ("image restoration", 1.6),            # 图像恢复
            
            # 图像质量（中等权重）
            ("image quality", 1.6),                # 图像质量
            ("image degradation", 1.6),            # 图像退化
            ("image artifact", 1.6),               # 图像伪影
            ("image distortion", 1.6),             # 图像失真
            ("image enhancement", 1.6),            # 图像增强
            
            # 特定应用（较低权重）
            ("low light enhancement", 1.4),        # 低光照增强
            ("hdr imaging", 1.4),                  # HDR成像
            ("image harmonization", 1.4),          # 图像协调
            ("image retouching", 1.4),             # 图像修饰
            ("image refinement", 1.4),             # 图像细化
        ],
        "negative_keywords": [
            # 生成模型（严格排除）
            "image generation",
            "image synthesis",
            "text to image",
            "diffusion model",
            "gan",
            
            # 检测分割（严格排除）
            "object detection",
            "semantic segmentation",
            "instance segmentation",
            "detection network",
            "segmentation model",
            
            # 3D任务（避免混淆）
            "3d reconstruction",
            "point cloud",
            "neural rendering",
            "nerf",
            "gaussian splatting",
            
            # 多模态（避免混淆）
            "multimodal",
            "cross modal",
            "vision language",
            "visual semantic",
            "joint embedding",
            
            # 图像理解（避免混淆）
            "image classification",
            "scene understanding",
            "visual recognition",
            "feature extraction",
            "attention mechanism"
        ]
    },

    # 人体分析（人体姿态估计、动作分析）
    "人体分析": {
        "keywords": [
            # 人体姿态核心概念（高权重核心词）
            ("human pose estimation", 1.8),      # 人体姿态估计（核心）
            ("3d human pose", 1.8),             # 3D人体姿态（核心）
            ("human motion capture", 1.8),       # 人体动作捕捉（核心）
            ("human mesh recovery", 1.8),        # 人体网格重建（核心）
            ("human motion estimation", 1.8),    # 人体运动估计（核心）
            
            # 人体动作分析（中等权重）
            ("human action recognition", 1.6),   # 人体动作识别
            ("human motion analysis", 1.6),      # 人体运动分析
            ("human activity recognition", 1.6), # 人体活动识别
            ("human behavior analysis", 1.6),    # 人体行为分析
            ("human interaction", 1.6),          # 人体交互
            
            # 人体解析和重建（中等权重）
            ("human body parsing", 1.6),         # 人体解析
            ("human shape estimation", 1.6),     # 人体形状估计
            ("human body reconstruction", 1.6),  # 人体重建
            ("human skeleton estimation", 1.6),  # 人体骨架估计
            ("human body tracking", 1.6),        # 人体跟踪
            
            # 组合关键词（较低权重）
            ("body pose", 1.4),                  # 身体姿态
            ("human skeleton", 1.4),             # 人体骨架
            ("human joints", 1.4),               # 人体关节
            ("human kinematics", 1.4),          # 人体运动学
            
            # 特定应用场景（较低权重）
            ("human gait analysis", 1.4),        # 人体步态分析
            ("sports motion", 1.4),              # 运动动作
            ("dance motion", 1.4),               # 舞蹈动作
            ("human fall detection", 1.4),       # 人体跌倒检测
        ],
        "negative_keywords": [
            # 人脸相关（严格排除）
            "face recognition",
            "facial expression",
            "face detection",
            "face tracking",
            "head pose",
            
            # 数字人相关（严格排除）
            "digital human",
            "virtual human",
            "metahuman",
            "character animation",
            "talking head",
            
            # 生成相关（部分排除）
            "human image generation",
            "human synthesis",
            "human gan",
            "human diffusion",
            
            # 其他人体相关任务（避免混淆）
            "person re-identification",
            "person reid",
            "pedestrian detection",
            "crowd counting",
            "person search",
            
            # 通用视觉任务（避免过度匹配）
            "object detection",
            "instance segmentation",
            "semantic segmentation",
            "scene understanding",
            
            # 特定领域（避免误分类）
            "fashion",
            "clothing",
            "dress",
            "outfit",
            "garment"
        ]
    },

    # 人脸技术（人脸识别、表情分析）
    "人脸技术": {
        "keywords": [
            # 人脸识别核心技术（高权重）
            ("face recognition system", 1.8),    # 人脸识别系统（核心）
            ("facial recognition system", 1.8),  # 面部识别系统（核心）
            ("face verification", 1.8),          # 人脸验证（核心）
            ("face identification", 1.8),        # 人脸识别（核心）
            ("face authentication", 1.8),        # 人脸认证（核心）
            
            # 表情分析（中等权重）
            ("facial expression recognition", 1.6), # 面部表情识别
            ("facial emotion recognition", 1.6),    # 面部情绪识别
            ("micro expression detection", 1.6),    # 微表情检测
            ("facial affect analysis", 1.6),        # 面部情感分析
            ("facial action unit", 1.6),           # 面部动作单元
            
            # 人脸特征和属性（中等权重）
            ("facial landmark detection", 1.6),   # 面部特征点检测
            ("face alignment", 1.6),              # 人脸对齐
            ("facial attribute analysis", 1.6),   # 面部属性分析
            ("face parsing", 1.6),                # 人脸解析
            ("face tracking", 1.6),               # 人脸跟踪
            
            # 组合关键词（较低权重）
            ("3d face", 1.4),                     # 3D人脸
            ("face detection", 1.4),              # 人脸检测
            ("face segmentation", 1.4),           # 人脸分割
            ("face reconstruction", 1.4),         # 人脸重建
            
            # 特定应用（较低权重）
            ("face anti spoofing", 1.4),          # 人脸反欺骗
            ("face liveness detection", 1.4),     # 人脸活体检测
            ("gaze estimation", 1.4),             # 视线估计
            ("head pose estimation", 1.4),        # 头部姿态估计
        ],
        "negative_keywords": [
            # 人体相关（严格排除）
            "human pose estimation",
            "body pose",
            "human motion",
            "human action",
            "human skeleton",
            
            # 数字人相关（严格排除）
            "digital human",
            "virtual human",
            "metahuman",
            "avatar generation",
            "character animation",
            
            # 生成相关（部分排除）
            "face generation",
            "face synthesis",
            "face gan",
            "face diffusion",
            "deepfake",
            
            # 美颜和编辑（避免混淆）
            "face beautification",
            "face makeup",
            "face editing",
            "face enhancement",
            "face retouching",
            
            # 通用视觉任务（避免过度匹配）
            "object detection",
            "instance segmentation",
            "semantic segmentation",
            "scene understanding",
            
            # 特定应用（避免误分类）
            "face swap",
            "face morphing",
            "face reenactment",
            "face stylization",
            "face cartoonization"
        ]
    },

    # 数字人（数字人生成、动画、驱动）
    "数字人": {
        "keywords": [
            # 数字人核心技术（高权重）
            ("digital human generation", 1.8),    # 数字人生成（核心）
            ("digital human synthesis", 1.8),     # 数字人合成（核心）
            ("virtual human creation", 1.8),      # 虚拟人创建（核心）
            ("digital avatar generation", 1.8),   # 数字化身生成（核心）
            ("metahuman creation", 1.8),          # 元人类创建（核心）
            
            # 数字人动画和驱动（高权重）
            ("digital human animation", 1.8),     # 数字人动画（核心）
            ("virtual human driving", 1.8),       # 虚拟人驱动（核心）
            ("digital human motion", 1.8),        # 数字人运动（核心）
            ("avatar control", 1.8),              # 数字化身控制（核心）
            ("talking head synthesis", 1.8),      # 说话头生成（核心）
            
            # 数字人渲染和外观（中等权重）
            ("digital human rendering", 1.6),     # 数字人渲染
            ("virtual human appearance", 1.6),    # 虚拟人外观
            ("digital human texture", 1.6),       # 数字人纹理
            ("avatar stylization", 1.6),          # 数字化身风格化
            ("digital human relighting", 1.6),    # 数字人重光照
            
            # 数字人交互和表现（中等权重）
            ("digital human interaction", 1.6),   # 数字人交互
            ("virtual human behavior", 1.6),      # 虚拟人行为
            ("digital human expression", 1.6),    # 数字人表情
            ("avatar gesture synthesis", 1.6),    # 数字化身手势
            ("digital human performance", 1.6),   # 数字人表演
            
            # 特定应用场景（较低权重）
            ("digital presenter", 1.4),           # 数字主播
            ("virtual idol", 1.4),                # 虚拟偶像
            ("digital twin human", 1.4),          # 数字孪生人
            ("virtual try on", 1.4),              # 虚拟试穿
            
            # 组合关键词（较低权重）
            ("3d avatar", 1.4),                   # 3D化身
            ("character rigging", 1.4),           # 角色绑定
            ("motion retargeting", 1.4),          # 动作重定向
            ("facial reenactment", 1.4),          # 人脸重演
        ],
        "negative_keywords": [
            # 传统人体分析（严格排除）
            "human pose estimation",
            "human motion capture",
            "human action recognition",
            "human behavior analysis",
            "human skeleton tracking",
            
            # 传统人脸技术（严格排除）
            "face recognition",
            "facial expression recognition",
            "face verification",
            "face detection",
            "face alignment",
            
            # 通用生成模型（避免混淆）
            "text to image",
            "image generation",
            "video generation",
            "style transfer",
            "image synthesis",
            
            # 3D场景重建（避免混淆）
            "3d reconstruction",
            "scene reconstruction",
            "point cloud",
            "mesh reconstruction",
            "surface reconstruction",
            
            # 神经渲染（避免混淆）
            "neural rendering",
            "neural radiance fields",
            "nerf",
            "implicit neural representation",
            "volume rendering",
            
            # 特定领域（避免误分类）
            "game character",
            "cartoon character",
            "anime character",
            "non photorealistic",
            "sketch based",
            
            # 通用视觉任务（避免过度匹配）
            "object detection",
            "instance segmentation",
            "semantic segmentation",
            "scene understanding",
            "image classification"
        ]
    },

    # 模型优化（模型压缩、加速、量化）
    "模型优化": {
        "keywords": [
            # 模型压缩核心（高权重）
            ("model compression", 1.8),             # 模型压缩（核心）
            ("network pruning", 1.8),              # 网络剪枝（核心）
            ("model quantization", 1.8),           # 模型量化（核心）
            ("parameter efficient", 1.8),          # 参数高效（核心）
            ("model distillation", 1.8),           # 模型蒸馏（核心）
            
            # 模型加速（中等权重）
            ("inference acceleration", 1.6),        # 推理加速
            ("model acceleration", 1.6),            # 模型加速
            ("efficient inference", 1.6),           # 高效推理
            ("lightweight model", 1.6),             # 轻量级模型
            ("computation efficient", 1.6),         # 计算高效
            
            # 优化技术（中等权重）
            ("knowledge distillation", 1.6),        # 知识蒸馏
            ("neural architecture search", 1.6),    # 神经架构搜索
            ("sparse training", 1.6),               # 稀疏训练
            ("weight sharing", 1.6),                # 权重共享
            ("model compression algorithm", 1.6),   # 模型压缩算法
            
            # 特定应用（较低权重）
            ("binary neural network", 1.4),         # 二值神经网络
            ("low bit quantization", 1.4),          # 低比特量化
            ("dynamic pruning", 1.4),               # 动态剪枝
            ("efficient architecture", 1.4),         # 高效架构
            ("model optimization", 1.4),             # 模型优化
        ],
        "negative_keywords": [
            # 生成模型（严格排除）
            "diffusion model",
            "generative model",
            "gan",
            "text to image",
            "image synthesis",
            
            # 特定领域模型（避免混淆）
            "detection model",
            "segmentation model",
            "recognition model",
            "tracking model",
            "pose estimation model",
            
            # 优化器和训练（避免混淆）
            "optimizer",
            "learning rate",
            "gradient descent",
            "backpropagation",
            "training strategy",
            
            # 通用深度学习（避免过度匹配）
            "deep learning",
            "neural network",
            "machine learning",
            "artificial intelligence",
            "backbone network"
        ]
    },

    # 具身智能（机器人和环境交互）
    "具身智能": {
        "keywords": [
            # 机器人相关（高权重）
            ("robot", 1.8),                        # 机器人（基础）
            ("robotic", 1.8),                      # 机器人的
            ("manipulation", 1.8),                 # 操作
            ("manipulator", 1.8),                  # 机械臂
            ("gripper", 1.8),                      # 夹持器
            ("grasping", 1.8),                     # 抓取
            ("navigation", 1.8),                   # 导航
            
            # 具身智能相关（高权重）
            ("embodied", 1.8),                     # 具身的
            ("interactive", 1.8),                  # 交互的
            ("interaction", 1.8),                  # 交互
            ("physical", 1.8),                     # 物理的
            ("real world", 1.8),                   # 真实世界
            ("real-world", 1.8),                   # 真实世界
            
            # 机器人学习（中等权重）
            ("reinforcement learning", 1.6),       # 强化学习
            ("imitation learning", 1.6),           # 模仿学习
            ("policy learning", 1.6),              # 策略学习
            ("skill learning", 1.6),               # 技能学习
            ("learning from demonstration", 1.6),  # 示教学习
            
            # 任务和动作（中等权重）
            ("motion planning", 1.6),              # 运动规划
            ("path planning", 1.6),                # 路径规划
            ("trajectory", 1.6),                   # 轨迹
            ("control", 1.6),                      # 控制
            ("pick and place", 1.6),              # 抓取放置
            ("obstacle avoidance", 1.6),           # 避障
            
            # 感知和环境（中等权重）
            ("perception", 1.6),                   # 感知
            ("environment", 1.6),                  # 环境
            ("scene", 1.6),                        # 场景
            ("exploration", 1.6),                  # 探索
            ("mapping", 1.6),                      # 建图
            ("localization", 1.6),                 # 定位
            
            # 具体应用（较低权重）
            ("mobile robot", 1.4),                 # 移动机器人
            ("humanoid", 1.4),                     # 人形机器人
            ("autonomous", 1.4),                   # 自主的
            ("dexterous", 1.4),                    # 灵巧的
            ("collaborative", 1.4),                # 协作的
            ("manipulation task", 1.4),            # 操作任务
            ("robotic task", 1.4),                # 机器人任务
        ],
        "negative_keywords": [
            # 纯视觉任务（排除）
            "detection",
            "segmentation",
            "classification",
            "recognition",
            "generation",
            "synthesis",
            
            # 3D相关（排除）
            "3d reconstruction",
            "nerf",
            "gaussian splatting",
            "neural rendering",
            
            # 多模态（排除）
            "multimodal",
            "language",
            "text",
            "audio",
            "speech",
            
            # 图像处理（排除）
            "image enhancement",
            "image restoration",
            "super resolution",
            "image quality",
            
            # 人体相关（排除）
            "pose estimation",
            "action recognition",
            "human motion",
            "skeleton"
        ]
    },

    # 扩散桥（优先级高于生成模型）
    "扩散桥": {
        "keywords": [
            # 核心概念（最高权重）
            ("diffusion bridge", 2.0),             # 扩散桥
            ("probability flow", 2.0),             # 概率流
            ("stochastic bridge", 2.0),            # 随机桥
            ("score matching", 2.0),               # 分数匹配
            ("score based model", 2.0),            # 基于分数的模型
            ("score function", 2.0),               # 分数函数
            ("stochastic differential equation", 2.0), # 随机微分方程
            ("sde", 2.0),                          # SDE
            ("langevin dynamics", 2.0),            # 朗之万动力学
            
            # 理论基础（高权重）
            ("probability flow ode", 1.8),          # 概率流常微分方程
            ("fokker planck", 1.8),                # 福克-普朗克
            ("reversible diffusion", 1.8),         # 可逆扩散
            ("brownian bridge", 1.8),              # 布朗桥
            ("markov chain", 1.8),                 # 马尔可夫链
            ("continuous time", 1.8),              # 连续时间
            ("discretization", 1.8),               # 离散化
            
            # 技术细节（中等权重）
            ("noise prediction", 1.5),             # 噪声预测
            ("noise estimation", 1.5),             # 噪声估计
            ("sampling strategy", 1.5),            # 采样策略
            ("reverse process", 1.5),              # 逆过程
            ("forward process", 1.5),              # 前向过程
            ("likelihood estimation", 1.5),        # 似然估计
            ("probability density", 1.5),          # 概率密度
            ("energy based", 1.5),                 # 基于能量
        ],
        "negative_keywords": [
            # 排除普通生成模型
            "stable diffusion",
            "text to image",
            "image editing",
            "inpainting",
            "style transfer",
            "gan",
            "generative adversarial",
            "vae",
            "variational autoencoder",
            
            # 排除其他任务
            "detection",
            "segmentation",
            "classification",
            "recognition",
            "tracking",
            "3d reconstruction",
            "nerf",
            "neural rendering",
            
            # 排除通用术语
            "transformer",
            "attention mechanism",
            "neural network",
            "deep learning",
            "machine learning",
            "training strategy",
            "optimization"
        ]
    },

    # 流模型（可逆神经网络和概率流）
    "流模型": {
        "keywords": [
            # 流模型核心（高权重）
            ("normalizing flow", 1.8),              # 标准化流（核心）
            ("invertible neural network", 1.8),     # 可逆神经网络（核心）
            ("flow based model", 1.8),              # 基于流的模型（核心）
            ("continuous normalizing flow", 1.8),   # 连续标准化流（核心）
            ("autoregressive flow", 1.8),           # 自回归流（核心）
            
            # 模型架构（中等权重）
            ("coupling layer", 1.6),                # 耦合层
            ("invertible architecture", 1.6),       # 可逆架构
            ("reversible network", 1.6),            # 可逆网络
            ("bijective mapping", 1.6),             # 双射映射
            ("flow architecture", 1.6),             # 流架构
            
            # 优化方法（中等权重）
            ("likelihood estimation", 1.6),         # 似然估计
            ("density estimation", 1.6),            # 密度估计
            ("change of variables", 1.6),           # 变量变换
            ("probability flow", 1.6),              # 概率流
            ("invertible transformation", 1.6),     # 可逆变换
            
            # 特定应用（较低权重）
            ("generative flow", 1.4),               # 生成流
            ("real nvp", 1.4),                      # Real NVP
            ("glow model", 1.4),                    # Glow模型
            ("flow matching", 1.4),                 # 流匹配
            ("continuous time flow", 1.4),          # 连续时间流
        ],
        "negative_keywords": [
            # 扩散模型（严格排除）
            "diffusion model",
            "stable diffusion",
            "latent diffusion",
            "denoising diffusion",
            "guided diffusion",
            
            # 扩散桥（避免混淆）
            "diffusion bridge",
            "stochastic differential equation",
            "martingale",
            "brownian motion",
            "stochastic process",
            
            # 生成模型（避免混淆）
            "text to image",
            "image generation",
            "video generation",
            "gan",
            "vae",
            
            # 通用深度学习（避免过度匹配）
            "deep learning",
            "neural network",
            "machine learning",
            "artificial intelligence",
            "backbone network"
        ]
    },
}

# 所有类别列表（不包括"其他"）
ALL_CATEGORIES = list(CATEGORY_THRESHOLDS.keys()) + ["其他"]

# 分类提示词
CATEGORY_PROMPT = f"""请从以下预定义类别中，选择最合适的1-2个类别：
{', '.join(ALL_CATEGORIES[:-1])}

每个类别的主要研究方向：
1. 3D场景：3D重建、新视角合成、点云处理、深度估计等
2. 神经渲染：NeRF及其变体
3. 3DGS：3D Gaussian Splatting及其变体
4. 生成模型：扩散模型、GAN、文生图等
5. 多模态：跨模态学习和对齐
6. 检测分割：目标检测、实例分割、语义分割等
7. 图像理解：图像分类、场景理解、细粒度识别等
8. 视频理解：动作识别、目标追踪、时序分析等
9. 图像处理：图像增强、超分辨率、图像修复等
10. 人体分析：姿态估计、动作分析、人体重识别等
11. 人脸技术：人脸识别、生成、动画等
12. 数字人：数字人生成、数字孪生、虚拟人等
13. 模型优化：模型压缩、加速、量化等
14. 具身智能：机器人、交互、环境等
15. 扩散桥：扩散过程、随机微分方程等
16. 流模型：可逆神经网络、概率流等"""