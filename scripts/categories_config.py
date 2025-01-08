"""
计算机视觉论文分类配置文件
"""

# 类别阈值配置
CATEGORY_THRESHOLDS = {
    "3D场景": 1.5,      # 降低阈值，因为3D相关论文关键词较少但很明确
    "生成模型": 1.8,    # 保持较高阈值，因为生成相关词较多
    "多模态学习": 1.8,  # 保持较高阈值，跨模态任务普遍
    "检测分割": 1.5,    # 降低阈值，检测分割任务关键词明确
    "图像理解": 1.8,    # 提高阈值，避免过度匹配
    "视频理解": 1.5,    # 降低阈值，视频相关论文关键词较少
    "图像处理": 1.8,    # 保持较高阈值，处理任务普遍
    "人体分析": 1.5,    # 降低阈值，人体相关任务明确
    "人脸技术": 1.5,    # 降低阈值，人脸相关任务明确
    "数字人": 1.8,      # 保持较高阈值，避免误匹配
    "模型优化": 1.5,    # 降低阈值，优化任务关键词明确
}

# 类别关键词配置
CATEGORY_KEYWORDS = {
    # 3D场景（3D重建、渲染、场景理解）
    "3D场景": {
        "keywords": [
            ("nerf", 1.5),                     # 最具特征性的技术
            ("3d reconstruction", 1.4),         # 核心任务
            ("point cloud", 1.4),              # 核心数据类型
            ("depth estimation", 1.3),         # 相关任务
            ("novel view", 1.3),               # 相关任务
            ("3d scene", 1.4),                 # 核心概念
            ("mesh", 1.2),                     # 相关概念
            ("rendering", 1.2),                # 相关技术
            ("slam", 1.3),                     # 相关技术
            ("3d object", 1.3),                # 相关概念
            ("3d shape", 1.3),                 # 相关概念
            ("geometry", 1.2),                 # 相关概念
            ("volumetric", 1.2),              # 相关概念
            ("stereo", 1.2),                  # 相关技术
            ("multi-view", 1.2),              # 相关技术
            ("surface reconstruction", 1.3),   # 相关任务
            ("camera pose", 1.2),             # 相关概念
            ("pointcloud", 1.4),              # 另一种写法
            ("lidar", 1.3),                   # 相关技术
        ],
        "negative_keywords": [
            "2d",                             # 2D相关
            "image classification",           # 图像分类任务
            "detection",                      # 检测任务
        ]
    },
    
    # 生成模型（文生图、图像生成、视频生成）
    "生成模型": {
        "keywords": [
            ("diffusion", 1.5),               # 核心技术
            ("gan", 1.4),                     # 核心技术
            ("generative", 1.4),              # 核心概念
            ("text to image", 1.5),           # 核心任务
            ("image synthesis", 1.4),         # 核心任务
            ("text to video", 1.5),           # 核心任务
            ("video synthesis", 1.4),         # 核心任务
            ("image generation", 1.4),        # 核心任务
            ("video generation", 1.4),        # 核心任务
            ("style transfer", 1.3),          # 相关任务
            ("editing", 1.2),                 # 相关任务
            ("inpainting", 1.3),             # 相关任务
            ("controllable", 1.2),           # 相关特征
            ("latent", 1.2),                 # 相关概念
            ("stable diffusion", 1.5),       # 特定技术
            ("text2image", 1.5),             # 另一种写法
            ("text2video", 1.5),             # 另一种写法
            ("img2img", 1.4),                # 特定任务
            ("synthetic", 1.2),              # 相关概念
            ("adversarial", 1.3),            # 相关技术
            ("stylegan", 1.4),               # 特定技术
            ("vae", 1.3),                    # 相关技术
        ],
        "negative_keywords": [
            "classification",                 # 分类任务
            "detection",                      # 检测任务
            "recognition",                    # 识别任务
        ]
    },
    
    # 多模态学习（视觉-语言理解、跨模态学习）
    "多模态学习": {
        "keywords": [
            ("visual language", 1.5),         # 核心概念
            ("vision language", 1.5),         # 核心概念
            ("multimodal", 1.4),             # 关键特征词
            ("cross modal", 1.4),            # 关键特征词
            ("text vision", 1.4),            # 核心概念
            ("visual question", 1.4),        # 典型任务
            ("image text", 1.4),             # 核心概念
            ("vision text", 1.4),            # 核心概念
            ("visual grounding", 1.3),       # 相关任务
            ("referring", 1.3),              # 相关任务
            ("caption", 1.3),                # 相关任务
            ("alignment", 1.2),              # 相关概念
            ("instruction", 1.2),            # 相关概念
            ("prompt", 1.2),                 # 相关概念
            ("vqa", 1.4),                    # 特定任务缩写
            ("visual reasoning", 1.3),       # 相关任务
            ("cross attention", 1.3),        # 相关技术
            ("vision-language", 1.5),        # 另一种写法
            ("text-vision", 1.4),            # 另一种写法
            ("image-text", 1.4),             # 另一种写法
        ],
        "negative_keywords": [
            "single modal",                  # 单模态
            "unimodal",                      # 单模态
        ]
    },

    # 检测分割（目标检测、实例分割、语义分割）
    "检测分割": {
        "keywords": [
            ("object detection", 1.4),        # 核心任务
            ("instance segmentation", 1.4),   # 核心任务
            ("semantic segmentation", 1.4),   # 核心任务
            ("detection", 1.3),              # 核心概念
            ("segmentation", 1.3),           # 核心概念
            ("detector", 1.3),               # 相关概念
            ("bbox", 1.3),                   # 相关概念
            ("boundary", 1.2),               # 相关概念
            ("mask", 1.2),                   # 相关概念
            ("anchor", 1.2),                 # 相关概念
            ("proposal", 1.2),               # 相关概念
            ("localization", 1.2),           # 相关概念
            ("region", 1.2),                 # 相关概念
            ("segment", 1.2),                # 相关概念
            ("yolo", 1.4),                   # 特定技术
            ("rcnn", 1.4),                   # 特定技术
            ("faster rcnn", 1.4),            # 特定技术
            ("mask rcnn", 1.4),              # 特定技术
            ("object localization", 1.3),    # 相关任务
            ("panoptic", 1.4),              # 全景分割
        ],
        "negative_keywords": [
            "generation",                    # 生成任务
            "synthesis",                     # 合成任务
            "style",                         # 风格相关
        ]
    },

    # 图像理解（分类、场景理解、细粒度识别）
    "图像理解": {
        "keywords": [
            ("image classification", 1.4),    # 核心任务
            ("scene understanding", 1.4),     # 核心任务
            ("fine grained", 1.4),           # 核心任务
            ("recognition", 1.3),            # 核心概念
            ("classifier", 1.3),             # 相关概念
            ("category", 1.2),               # 相关概念
            ("attribute", 1.2),              # 相关概念
            ("scene parsing", 1.3),          # 相关任务
            ("scene graph", 1.3),            # 相关概念
            ("visual reasoning", 1.3),       # 相关任务
            ("knowledge", 1.2),              # 相关概念
            ("concept", 1.2),                # 相关概念
            ("semantic", 1.2),               # 相关概念
            ("hierarchy", 1.2),              # 相关概念
            ("image retrieval", 1.3),        # 相关任务
            ("image annotation", 1.3),       # 相关任务
            ("image tagging", 1.3),          # 相关任务
            ("image captioning", 1.3),       # 相关任务
        ],
        "negative_keywords": [
            "detection",                      # 检测任务
            "segmentation",                   # 分割任务
            "tracking",                       # 跟踪任务
        ]
    },

    # 视频理解（动作识别、目标追踪、时序分析）
    "视频理解": {
        "keywords": [
            ("action recognition", 1.4),      # 核心任务
            ("video tracking", 1.4),          # 核心任务
            ("temporal", 1.3),                # 核心概念
            ("motion", 1.3),                  # 核心概念
            ("tracking", 1.3),                # 核心概念
            ("trajectory", 1.2),              # 相关概念
            ("dynamic", 1.2),                 # 相关概念
            ("sequence", 1.2),                # 相关概念
            ("action detection", 1.3),        # 相关任务
            ("video analysis", 1.3),          # 相关任务
            ("activity", 1.2),                # 相关概念
            ("event", 1.2),                   # 相关概念
            ("frame", 1.2),                   # 相关概念
            ("optical flow", 1.3),            # 相关技术
            ("motion estimation", 1.3),       # 相关任务
            ("object detection", 1.3),        # 相关任务
            ("scene understanding", 1.3),     # 相关任务
        ],
        "negative_keywords": [
            "image",                          # 图像相关
            "static",                         # 静态相关
            "2d",                             # 2D相关
        ]
    },

    # 图像处理（图像增强、修复、编辑）
    "图像处理": {
        "keywords": [
            ("image enhancement", 1.4),       # 核心任务
            ("image restoration", 1.4),       # 核心任务
            ("super resolution", 1.4),        # 核心任务
            ("denoising", 1.3),              # 相关任务
            ("deblurring", 1.3),             # 相关任务
            ("enhancement", 1.3),             # 相关概念
            ("restoration", 1.3),             # 相关概念
            ("quality", 1.2),                 # 相关概念
            ("artifact", 1.2),                # 相关概念
            ("degradation", 1.2),             # 相关概念
            ("compression", 1.2),             # 相关任务
            ("style transfer", 1.3),          # 相关任务
            ("harmonization", 1.2),           # 相关任务
            ("retouching", 1.2),              # 相关任务
            ("inpainting", 1.3),             # 相关任务
            ("image editing", 1.3),           # 相关任务
            ("image manipulation", 1.3),      # 相关任务
        ],
        "negative_keywords": [
            "detection",                      # 检测任务
            "segmentation",                   # 分割任务
            "recognition",                    # 识别任务
        ]
    },

    # 人体分析（姿态估计、动作分析、重识别）
    "人体分析": {
        "keywords": [
            ("pose estimation", 1.4),         # 核心任务
            ("human pose", 1.4),              # 核心任务
            ("action analysis", 1.4),         # 核心任务
            ("person re", 1.4),               # 核心任务
            ("human motion", 1.3),            # 相关任务
            ("skeleton", 1.3),                # 相关概念
            ("body", 1.2),                    # 相关概念
            ("gesture", 1.2),                 # 相关概念
            ("pedestrian", 1.2),              # 相关概念
            ("gait", 1.2),                    # 相关概念
            ("human parsing", 1.3),           # 相关任务
            ("human mesh", 1.3),              # 相关任务
            ("human shape", 1.3),             # 相关任务
            ("human reconstruction", 1.3),     # 相关任务
            ("facial analysis", 1.3),         # 相关任务
            ("hand pose", 1.3),               # 相关任务
        ],
        "negative_keywords": [
            "object",                         # 物体相关
            "scene",                          # 场景相关
            "image",                          # 图像相关
        ]
    },

    # 人脸技术（人脸识别、生成、动画）
    "人脸技术": {
        "keywords": [
            ("face recognition", 1.4),        # 核心任务
            ("facial recognition", 1.4),      # 核心任务
            ("face generation", 1.4),         # 核心任务
            ("face animation", 1.4),          # 核心任务
            ("face editing", 1.3),            # 相关任务
            ("face synthesis", 1.3),          # 相关任务
            ("facial expression", 1.3),       # 相关概念
            ("face reconstruction", 1.3),     # 相关任务
            ("face detection", 1.3),          # 相关任务
            ("face tracking", 1.3),           # 相关任务
            ("face alignment", 1.3),          # 相关任务
            ("face verification", 1.3),       # 相关任务
            ("face attribute", 1.2),          # 相关概念
            ("face landmark", 1.2),           # 相关概念
            ("facial landmark", 1.3),         # 相关概念
            ("facial feature", 1.3),          # 相关概念
        ],
        "negative_keywords": [
            "body",                           # 身体相关
            "gesture",                        # 手势相关
            "action",                         # 动作相关
        ]
    },

    # 数字人（数字人、虚拟人、数字孪生）
    "数字人": {
        "keywords": [
            ("digital human", 1.4),           # 核心概念
            ("virtual human", 1.4),           # 核心概念
            ("digital avatar", 1.4),          # 核心概念
            ("talking head", 1.4),            # 核心任务
            ("digital twin", 1.3),            # 相关概念
            ("virtual character", 1.3),       # 相关概念
            ("avatar", 1.2),                  # 相关概念
            ("character animation", 1.3),     # 相关任务
            ("human synthesis", 1.3),         # 相关任务
            ("human generation", 1.3),        # 相关任务
            ("human animation", 1.3),         # 相关任务
            ("facial animation", 1.3),        # 相关任务
            ("motion synthesis", 1.2),        # 相关任务
            ("performance capture", 1.2),     # 相关任务
            ("motion capture", 1.3),          # 相关任务
            ("character modeling", 1.3),      # 相关任务
        ],
        "negative_keywords": [
            "real",                           # 真实相关
            "physical",                       # 物理相关
            "robot",                          # 机器人相关
        ]
    },

    # 模型优化（模型压缩、加速、轻量化）
    "模型优化": {
        "keywords": [
            ("model compression", 1.4),        # 核心任务
            ("model acceleration", 1.4),       # 核心任务
            ("network pruning", 1.4),          # 核心任务
            ("quantization", 1.3),             # 相关技术
            ("distillation", 1.3),             # 相关技术
            ("lightweight", 1.3),              # 相关概念
            ("efficient", 1.2),                # 相关概念
            ("acceleration", 1.2),             # 相关概念
            ("compression", 1.2),              # 相关概念
            ("optimization", 1.2),             # 相关概念
            ("pruning", 1.2),                  # 相关概念
            ("sparse", 1.2),                   # 相关概念
            ("deployment", 1.2),               # 相关概念
            ("inference", 1.2),                # 相关概念
            ("knowledge distillation", 1.3),   # 相关技术
            ("model simplification", 1.3),     # 相关任务
            ("model reduction", 1.3),          # 相关任务
        ],
        "negative_keywords": [
            "training",                       # 训练相关
            "learning",                       # 学习相关
            "inference",                      # 推理相关
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
2. 生成模型：扩散模型、GAN、文生图、视频生成等
3. 多模态学习：视觉-语言理解、跨模态对齐、多模态融合等
4. 检测分割：目标检测、实例分割、语义分割等
5. 图像理解：图像分类、场景理解、细粒度识别等
6. 视频理解：动作识别、目标追踪、时序分析等
7. 图像处理：图像增强、超分辨率、图像修复等
8. 人体分析：姿态估计、动作分析、人体重识别等
9. 人脸技术：人脸识别、生成、动画等
10. 数字人：数字人生成、数字孪生、虚拟人等
11. 模型优化：模型压缩、加速、轻量化等"""