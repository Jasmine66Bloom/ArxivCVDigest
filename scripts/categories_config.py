"""
计算机视觉论文分类配置文件
"""

# 类别阈值配置
CATEGORY_THRESHOLDS = {
    "3D场景": 1.0,      # 传统3D重建和SLAM相关论文
    "神经渲染": 1.0,    # NeRF及神经隐式表示相关论文
    "3DGS": 1.0,       # 3D Gaussian Splatting相关论文
    "生成模型": 1.2,    # 文图生成、视频生成等
    "多模态": 1.2,     # 跨模态学习和对齐
    "检测分割": 1.0,    # 目标检测和分割任务
    "图像理解": 1.0,    # 图像分类和场景理解
    "视频理解": 1.0,    # 视频分析和动作识别
    "图像处理": 1.0,    # 图像增强和修复
    "人体分析": 1.0,    # 人体姿态和动作分析
    "人脸技术": 1.0,    # 人脸识别和生成
    "数字人": 1.0,      # 数字人生成和动画
    "模型优化": 1.0,    # 模型压缩和加速
    "具身智能": 1.0,    # 机器人和环境交互
    "扩散桥": 1.0,      # 扩散过程和随机微分方程
    "流模型": 1.0,      # 可逆神经网络和概率流
}

# 类别关键词配置
CATEGORY_KEYWORDS = {
    # 3D场景（传统3D重建、SLAM、多视图几何）
    "3D场景": {
        "keywords": [
            # SLAM核心概念
            ("visual slam", 1.5),              # 视觉SLAM
            ("simultaneous localization", 1.5), # SLAM全称
            ("visual odometry", 1.4),          # 视觉里程计
            ("loop closure", 1.4),             # 回环检测
            ("pose graph", 1.4),               # 位姿图优化
            
            # 多视图几何
            ("structure from motion", 1.5),     # 运动恢复结构
            ("multi view geometry", 1.4),       # 多视图几何
            ("bundle adjustment", 1.4),         # 光束法平差
            ("epipolar geometry", 1.4),         # 对极几何
            ("triangulation", 1.4),             # 三角化
            
            # 3D重建
            ("3d reconstruction", 1.5),         # 3D重建
            ("mvs reconstruction", 1.4),        # 多视图立体重建
            ("dense reconstruction", 1.4),      # 稠密重建
            ("photogrammetry", 1.4),           # 摄影测量
            ("point cloud registration", 1.4),  # 点云配准
        ],
        "negative_keywords": [
            # 神经渲染相关
            "nerf",
            "neural radiance",
            "neural field",
            "implicit neural",
            "neural implicit",
            "neural rendering",
            
            # 3DGS相关
            "gaussian splatting",
            "3d gaussian",
            "gaussian splat",
            "gaussian point",
            
            # 人体/人脸相关
            "human pose",
            "face",
            "body",
            "avatar",
            
            # 生成相关
            "diffusion",
            "gan",
            "generation",
            "synthesis"
        ]
    },

    # 神经渲染（NeRF及其变体、神经隐式表示）
    "神经渲染": {
        "keywords": [
            # NeRF核心概念
            ("nerf", 1.5),                      # 神经辐射场
            ("neural radiance field", 1.5),     # 神经辐射场全称
            ("neural rendering", 1.5),          # 神经渲染
            ("radiance field", 1.4),            # 辐射场
            ("novel view synthesis", 1.4),      # 新视角合成
            
            # NeRF变体
            ("instant ngp", 1.4),               # Instant-NGP
            ("dynamic nerf", 1.4),              # 动态NeRF
            ("deformable nerf", 1.4),           # 可变形NeRF
            ("neus", 1.4),                      # NeuS
            ("neural volume", 1.4),             # 神经体积
            
            # 神经隐式表示
            ("neural implicit", 1.4),           # 神经隐式
            ("implicit neural", 1.4),           # 隐式神经
            ("occupancy network", 1.4),         # 占用网络
            ("signed distance function", 1.4),   # 有符号距离函数
            ("neural sdf", 1.4),                # 神经SDF
        ],
        "negative_keywords": [
            # 3DGS相关
            "gaussian splatting",
            "3d gaussian",
            "gaussian splat",
            "gaussian point",
            "covariance",
            
            # 传统3D重建
            "slam",
            "structure from motion",
            "bundle adjustment",
            "point cloud registration",
            "photogrammetry",
            
            # 生成相关
            "diffusion",
            "gan",
            "generation",
            "synthesis",
            
            # 避免与其他神经方法混淆
            "neural network",
            "neural architecture",
            "neural style"
        ]
    },

    # 3DGS（3D Gaussian Splatting及其变体）
    "3DGS": {
        "keywords": [
            # 核心概念
            ("gaussian splatting", 1.5),        # 高斯散射
            ("3d gaussian", 1.5),               # 3D高斯
            ("gaussian splat", 1.5),            # 高斯散射变体
            ("3d gaussians", 1.5),              # 3D高斯复数形式
            
            # 技术细节
            ("gaussian rasterization", 1.4),    # 高斯光栅化
            ("covariance matrix", 1.4),         # 协方差矩阵
            ("anisotropic gaussian", 1.4),      # 各向异性高斯
            ("spherical gaussian", 1.4),        # 球面高斯
            ("differentiable splatting", 1.4),  # 可微散射
            
            # 变体和扩展
            ("4d gaussian", 1.4),               # 4D高斯
            ("dynamic gaussian", 1.4),          # 动态高斯
            ("deformable gaussian", 1.4),       # 可变形高斯
            ("animated gaussian", 1.4),         # 动画高斯
            ("gaussian optimization", 1.4),      # 高斯优化
        ],
        "negative_keywords": [
            # 神经渲染相关
            "nerf",
            "neural radiance",
            "neural field",
            "implicit neural",
            "neural implicit",
            "neural rendering",
            
            # 传统3D重建
            "slam",
            "structure from motion",
            "bundle adjustment",
            "point cloud registration",
            "photogrammetry",
            
            # 生成相关
            "diffusion",
            "gan",
            "generation",
            "synthesis",
            
            # 避免与其他高斯方法混淆
            "gaussian mixture",
            "gaussian process",
            "gaussian noise"
        ]
    },

    # 生成模型（扩散模型、GAN、文生图）
    "生成模型": {
        "keywords": [
            # 扩散模型
            ("diffusion model", 1.5),           # 扩散模型
            ("latent diffusion", 1.5),          # 潜空间扩散
            ("stable diffusion", 1.5),          # Stable Diffusion
            ("score based", 1.4),               # 基于分数
            ("denoising diffusion", 1.4),       # 去噪扩散
            
            # GAN
            ("generative adversarial", 1.5),    # GAN
            ("style gan", 1.4),                 # StyleGAN
            ("adversarial network", 1.4),       # 对抗网络
            ("gan", 1.4),                       # GAN缩写
            
            # 文生图/视频
            ("text to image", 1.5),             # 文本到图像
            ("text to video", 1.5),             # 文本到视频
            ("imagen", 1.4),                    # Imagen
            ("dall e", 1.4),                    # DALL-E
            ("midjourney", 1.4),                # Midjourney
        ],
        "negative_keywords": [
            # 3D相关
            "nerf",
            "gaussian splatting",
            "3d reconstruction",
            "point cloud",
            "mesh",
            
            # 流模型相关
            "normalizing flow",
            "continuous flow",
            "flow based",
            "invertible",
            
            # 扩散桥相关
            "diffusion bridge",
            "sde",
            "stochastic differential",
            "martingale",
            
            # 避免与传统生成方法混淆
            "procedural generation",
            "rule based",
            "template based"
        ]
    },

    # 多模态（跨模态学习和对齐）
    "多模态": {
        "keywords": [
            # 跨模态学习
            ("cross modal learning", 1.5),      # 跨模态学习
            ("multi modal learning", 1.5),      # 多模态学习
            ("modal alignment", 1.4),           # 模态对齐
            
            # 视觉-语言理解
            ("visual language understanding", 1.4), # 视觉-语言理解
            ("visual question answering", 1.4), # 视觉问答
            
            # 视觉-语言生成
            ("visual language generation", 1.4), # 视觉-语言生成
            ("image captioning", 1.4),          # 图像字幕
            
            # 跨模态检索
            ("cross modal retrieval", 1.4),     # 跨模态检索
            ("multi modal retrieval", 1.4),     # 多模态检索
        ],
        "negative_keywords": [
            # 生成相关
            "generative",
            "synthesis",
            "generation",
            
            # 3D相关
            "nerf",
            "gaussian splatting",
            "3d reconstruction",
            "point cloud",
        ]
    },

    # 检测分割（目标检测、实例分割、语义分割）
    "检测分割": {
        "keywords": [
            # 目标检测
            ("object detection", 1.5),          # 目标检测
            ("detection", 1.4),                 # 检测
            
            # 实例分割
            ("instance segmentation", 1.5),     # 实例分割
            ("instance", 1.4),                  # 实例
            
            # 语义分割
            ("semantic segmentation", 1.5),     # 语义分割
            ("semantic", 1.4),                  # 语义
            
            # 分割算法
            ("segmentation algorithm", 1.4),    # 分割算法
            ("segmentation method", 1.4),       # 分割方法
        ],
        "negative_keywords": [
            # 生成相关
            "generative",
            "synthesis",
            "generation",
            
            # 3D相关
            "nerf",
            "gaussian splatting",
            "3d reconstruction",
            "point cloud",
        ]
    },

    # 图像理解（图像分类、场景理解、细粒度识别）
    "图像理解": {
        "keywords": [
            # 图像分类
            ("image classification", 1.5),      # 图像分类
            ("classification", 1.4),            # 分类
            
            # 场景理解
            ("scene understanding", 1.5),       # 场景理解
            ("scene", 1.4),                     # 场景
            
            # 细粒度识别
            ("fine grained recognition", 1.5),  # 细粒度识别
            ("fine grained", 1.4),              # 细粒度
            
            # 图像识别
            ("image recognition", 1.4),         # 图像识别
            ("recognition", 1.4),               # 识别
        ],
        "negative_keywords": [
            # 生成相关
            "generative",
            "synthesis",
            "generation",
            
            # 3D相关
            "nerf",
            "gaussian splatting",
            "3d reconstruction",
            "point cloud",
        ]
    },

    # 视频理解（动作识别、目标追踪、时序分析）
    "视频理解": {
        "keywords": [
            # 动作识别
            ("action recognition", 1.5),        # 动作识别
            ("action", 1.4),                    # 动作
            
            # 目标追踪
            ("object tracking", 1.5),           # 目标追踪
            ("tracking", 1.4),                  # 追踪
            
            # 时序分析
            ("temporal analysis", 1.5),         # 时序分析
            ("temporal", 1.4),                  # 时序
            
            # 视频分析
            ("video analysis", 1.4),            # 视频分析
            ("video", 1.4),                     # 视频
        ],
        "negative_keywords": [
            # 生成相关
            "generative",
            "synthesis",
            "generation",
            
            # 3D相关
            "nerf",
            "gaussian splatting",
            "3d reconstruction",
            "point cloud",
        ]
    },

    # 图像处理（图像增强、修复、编辑）
    "图像处理": {
        "keywords": [
            # 图像增强
            ("image enhancement", 1.5),         # 图像增强
            ("enhancement", 1.4),               # 增强
            
            # 图像修复
            ("image restoration", 1.5),         # 图像修复
            ("restoration", 1.4),               # 修复
            
            # 图像编辑
            ("image editing", 1.5),             # 图像编辑
            ("editing", 1.4),                   # 编辑
            
            # 图像处理算法
            ("image processing algorithm", 1.4), # 图像处理算法
            ("image processing method", 1.4),   # 图像处理方法
        ],
        "negative_keywords": [
            # 生成相关
            "generative",
            "synthesis",
            "generation",
            
            # 3D相关
            "nerf",
            "gaussian splatting",
            "3d reconstruction",
            "point cloud",
        ]
    },

    # 人体分析（姿态估计、动作分析、重识别）
    "人体分析": {
        "keywords": [
            # 姿态估计
            ("pose estimation", 1.5),           # 姿态估计
            ("pose", 1.4),                      # 姿态
            
            # 动作分析
            ("action analysis", 1.5),           # 动作分析
            ("action", 1.4),                    # 动作
            
            # 重识别
            ("re identification", 1.5),         # 重识别
            ("re id", 1.4),                     # 重识别
            
            # 人体识别
            ("human recognition", 1.4),         # 人体识别
            ("recognition", 1.4),               # 识别
        ],
        "negative_keywords": [
            # 生成相关
            "generative",
            "synthesis",
            "generation",
            
            # 3D相关
            "nerf",
            "gaussian splatting",
            "3d reconstruction",
            "point cloud",
        ]
    },

    # 人脸技术（人脸识别、生成、动画）
    "人脸技术": {
        "keywords": [
            # 人脸识别
            ("face recognition", 1.5),          # 人脸识别
            ("face", 1.4),                      # 人脸
            
            # 人脸生成
            ("face generation", 1.5),           # 人脸生成
            ("generation", 1.4),                # 生成
            
            # 人脸动画
            ("face animation", 1.5),            # 人脸动画
            ("animation", 1.4),                 # 动画
            
            # 人脸编辑
            ("face editing", 1.4),              # 人脸编辑
            ("editing", 1.4),                   # 编辑
        ],
        "negative_keywords": [
            # 生成相关
            "generative",
            "synthesis",
            "generation",
            
            # 3D相关
            "nerf",
            "gaussian splatting",
            "3d reconstruction",
            "point cloud",
        ]
    },

    # 数字人（数字人、虚拟人、数字孪生）
    "数字人": {
        "keywords": [
            # 数字人
            ("digital human", 1.5),             # 数字人
            ("digital", 1.4),                   # 数字
            
            # 虚拟人
            ("virtual human", 1.5),             # 虚拟人
            ("virtual", 1.4),                   # 虚拟
            
            # 数字孪生
            ("digital twin", 1.5),              # 数字孪生
            ("twin", 1.4),                      # 孪生
            
            # 虚拟现实
            ("virtual reality", 1.4),           # 虚拟现实
            ("vr", 1.4),                        # 虚拟现实
        ],
        "negative_keywords": [
            # 生成相关
            "generative",
            "synthesis",
            "generation",
            
            # 3D相关
            "nerf",
            "gaussian splatting",
            "3d reconstruction",
            "point cloud",
        ]
    },

    # 模型优化（模型压缩、加速、轻量化）
    "模型优化": {
        "keywords": [
            # 模型压缩
            ("model compression", 1.5),         # 模型压缩
            ("compression", 1.4),               # 压缩
            
            # 模型加速
            ("model acceleration", 1.5),        # 模型加速
            ("acceleration", 1.4),              # 加速
            
            # 模型轻量化
            ("model lightweight", 1.5),         # 模型轻量化
            ("lightweight", 1.4),                # 轻量化
            
            # 模型优化
            ("model optimization", 1.4),        # 模型优化
            ("optimization", 1.4),              # 优化
        ],
        "negative_keywords": [
            # 生成相关
            "generative",
            "synthesis",
            "generation",
            
            # 3D相关
            "nerf",
            "gaussian splatting",
            "3d reconstruction",
            "point cloud",
        ]
    },

    # 具身智能（机器人、交互、环境）
    "具身智能": {
        "keywords": [
            # 机器人
            ("robot", 1.5),                     # 机器人
            ("robotics", 1.4),                  # 机器人学
            
            # 交互
            ("interaction", 1.5),               # 交互
            ("interactive", 1.4),               # 交互
            
            # 环境
            ("environment", 1.5),               # 环境
            ("env", 1.4),                       # 环境
            
            # 机器人学习
            ("robot learning", 1.4),            # 机器人学习
            ("learning", 1.4),                  # 学习
        ],
        "negative_keywords": [
            # 生成相关
            "generative",
            "synthesis",
            "generation",
            
            # 3D相关
            "nerf",
            "gaussian splatting",
            "3d reconstruction",
            "point cloud",
        ]
    },

    # 扩散桥（扩散过程、随机微分方程）
    "扩散桥": {
        "keywords": [
            # 扩散过程
            ("diffusion process", 1.5),         # 扩散过程
            ("diffusion", 1.4),                 # 扩散
            
            # 随机微分方程
            ("stochastic differential equation", 1.5), # 随机微分方程
            ("sde", 1.4),                       # 随机微分方程
            
            # 扩散桥
            ("diffusion bridge", 1.5),          # 扩散桥
            ("bridge", 1.4),                    # 桥
            
            # 随机过程
            ("stochastic process", 1.4),        # 随机过程
            ("process", 1.4),                   # 过程
        ],
        "negative_keywords": [
            # 生成相关
            "generative",
            "synthesis",
            "generation",
            
            # 3D相关
            "nerf",
            "gaussian splatting",
            "3d reconstruction",
            "point cloud",
        ]
    },

    # 流模型（可逆神经网络、概率流）
    "流模型": {
        "keywords": [
            # 可逆神经网络
            ("invertible neural network", 1.5), # 可逆神经网络
            ("invertible", 1.4),                # 可逆
            
            # 概率流
            ("probability flow", 1.5),          # 概率流
            ("flow", 1.4),                      # 流
            
            # 正则化流
            ("normalizing flow", 1.5),          # 正则化流
            ("normalizing", 1.4),               # 正则化
            
            # 连续流
            ("continuous flow", 1.4),           # 连续流
            ("continuous", 1.4),                # 连续
        ],
        "negative_keywords": [
            # 生成相关
            "generative",
            "synthesis",
            "generation",
            
            # 3D相关
            "nerf",
            "gaussian splatting",
            "3d reconstruction",
            "point cloud",
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
13. 模型优化：模型压缩、加速、轻量化等
14. 具身智能：机器人、交互、环境等
15. 扩散桥：扩散过程、随机微分方程等
16. 流模型：可逆神经网络、概率流等"""