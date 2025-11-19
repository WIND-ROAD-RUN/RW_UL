# ImgPro 模块库说明文档 

## 介绍
封装了若干个可复用的业务逻辑模块，提供给视觉检测项目、工业剔除项目使用。

## 目录结构
```
ImgPro/
├── include/                  # 模块头文件
├── src/                      # 模块源代码
├── test/                     # 模块测试代码
├── testQt/                   # 模块测试代码（Qt 版本）
├── CMakeLists.txt            # CMake 构建配置文件
└── README.md                 # 项目说明文件
```

## 组件概览图
```mermaid
graph TD
  classDef noteStyle fill:white,stroke:yellow,stroke-width:1px,font-size:12px,color:black;

  subgraph "ImgPro"
    ImageProcess["ImageProcess"]
    DefectDrawFunc["DefectDrawFunc"]
    DefectResultInfoFunc["DefectResultInfoFunc"]
    EliminationInfoFunc["EliminationInfoFunc"]
    ImagePainter["ImagePainter"]
    ImageProcessUtilty["ImageProcessUtilty"]
    IndexFunc["IndexFunc"]

    ImageProcess --> ImageProcessUtilty
    ImageProcess --> DefectDrawFunc
    ImageProcess --> DefectResultInfoFunc
    ImageProcess --> EliminationInfoFunc
    ImageProcess --> ImagePainter
    ImageProcess --> IndexFunc
    IndexFunc --> ImageProcessUtilty
    ImagePainter --> ImageProcessUtilty
    EliminationInfoFunc --> ImageProcessUtilty
    DefectResultInfoFunc --> ImageProcessUtilty
    DefectDrawFunc --> ImageProcessUtilty
  end

  NoteImageProcess["说明: ImageProcess封装好了现成的识别、缺陷index字典化、获取剔除信息、获取剔除结果以及绘画的功能接口"] --> ImageProcess
  NoteImageProcessUtilty["说明: ImageProcessUtilty 提供了所有的别名"] --> ImageProcessUtilty
  NoteDefectDrawFunc["说明: DefectDrawFunc 提供了缺陷绘画相关的功能接口以及算法框架"] --> DefectDrawFunc
  NoteDefectResultInfoFunc["说明: DefectResultInfoFunc 提供了缺陷结果相关的功能接口以及算法框架"] --> DefectResultInfoFunc
  NoteEliminationInfoFunc["说明: EliminationInfoFunc 提供了剔除信息相关的功能接口以及算法框架"] --> EliminationInfoFunc
  NoteImagePainter["说明: ImagePainter 提供了图像绘画相关的功能接口以及算法框架"] --> ImagePainter
  NoteIndexFunc["说明: IndexFunc 提供了缺陷index字典化相关的功能接口以及算法框架"] --> IndexFunc

  class NoteImageProcess,NoteImageProcessUtilty,NoteDefectDrawFunc,NoteDefectResultInfoFunc,NoteEliminationInfoFunc,NoteImagePainter,NoteIndexFunc noteStyle
  
```

## 组件说明
### IndexFunc 缺陷index字典化组件