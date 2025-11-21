# RW_UL(road wind utilit library) road wind 复用化工具库

## 介绍
RW_UL 是一个专为视觉检测项目、工业剔除项目设计的复用化工具库，旨在提供一套高效、可靠的基础设施和组件，帮助开发者快速构建和维护复杂的应用程序。该库包含了多种实用的工具函数、组件模块以及底层设施，支持项目的可扩展性和可维护性。

## 目录结构
```
RW_UL/
├── fundation/               # 基础设施模块，提供基础功能函数与工具方法
├── Module/                  # 组件模块，封装了若干个可复用的业务逻辑组件
└── README.md                # 项目说明文件
```

## 组件库模块依赖图
```mermaid
graph TD
  classDef noteStyle fill:white,stroke:yellow,stroke-width:1px,font-size:12px,color:black;

  subgraph "RW_UL"
    Module["Module"] --> fundation["fundation"]
  end

  subgraph "ThirdLibrary"
    MVSSDK["MVSSDK"]
    DSSDK["DSSDK"]
    HalconSDK["Halcon"]
    Onnxruntime-gpu-1.12.0["Onnxruntime-gpu-1.12.0"]
    subgraph "TensorRT"
        TensorRT10.12-cuda12["TensorRT10.12-cuda12"]
        TensorRT10.12-cuda11["TensorRT10.12-cuda11"]
        TensorRT8.6-cuda11["TensorRT8.6-cuda11"]
    end
    VisionMasterSDK4.4.0["VisionMasterSDK4.4.0"]
    ZMotionSDK["ZMotionSDK"]
    opencv4["opencv4"]
    pugixml["pugixml"]
    jsoncpp["jsoncpp"]
    sqlite3["sqlite3"]
    gtest["gtest"]
    spdlog["spdlog"]
    cryptopp["cryptopp"]
    openssl["openssl"]
    libmodbus["libmodbus"]
    libzip["libzip"]
    std["std"]
    WindowsAPI["WindowsAPI"]
    LinuxAPI["LinuxAPI"]
    serial["serial"]
    Qt6["Qt6"]
  end

  RW_UL--> ThirdLibrary


  Note1["说明：Module 封装了若干个组件，提供复用化的业务逻辑。"] --> Module

  Note2["说明：fundation 负责提供基础功能函数与工具方法以及一些底层设施。"] --> fundation
  
  class Note1,Note2 noteStyle
```