# fundation 

## 介绍
fundation 模块负责提供基础功能函数与工具方法以及一些底层设施。该模块为 RW_UL 组件库的其他模块提供支持和服务，确保各个组件能够高效、稳定地运行。fundation 模块包含多个子模块，每个子模块专注于特定的功能领域，如加密算法、数据处理、文件操作等。通过这些子模块，fundation 模块为整个组件库提供了坚实的基础，促进了代码的复用和维护。

## 目录结构
```
fundation/
├── cla
├── dsl
├── hoe
├── ime
├── oso
├── rqw
├── scc
├── utilty
└── README.md            
```

## 组件库模块依赖图
```mermaid
graph TD
  classDef noteStyle fill:white,stroke:yellow,stroke-width:1px,font-size:12px,color:black;

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

  subgraph "RW_UL"
    fundation["Module"]
    subgraph "fundation"
        cla["cla"]
        cla --> openssl
        cla --> cryptopp


        dsl["dsl"]
        dsl --> std

        ime["ime"]
        ime --> Onnxruntime-gpu-1.12.0
        ime --> TensorRT10.12-cuda12
        ime --> TensorRT10.12-cuda11
        ime --> TensorRT8.6-cuda11

        oso["oso"]
        oso --> pugixml
        oso --> jsoncpp
        oso --> sqlite3

        scc["scc"] 
        scc --> ZMotionSDK

        utilty["utilty"]

        subgraph "hoe"
            hoec["hoec"]
            hoec --> DSSDK
            hoec --> MVSSDK

            hoeRefactor["hoeRefactor"]
            hoeRefactor --> DSSDK
            hoeRefactor --> MVSSDK


            hoei["hoei"]
            hoei --> WindowsAPI
            hoei --> LinuxAPI

            hoem["hoem"]
            hoem --> libmodbus

            hoes["hoes"]
            hoes --> serial


            hoe::utilty["hoe::utilty"]
        end

        subgraph "rqw"
            main["main"]
            main --> dsl
            main --> cla
            main --> oso

            utilty["utilty"]

            subgraph "expandComponet"
                rqwhalcon["rqwhalcon"]
                rqwhalcon --> HalconSDK

                rqwhoec["rqwhoec"]
                rqwhoec --> hoeRefactor

                rqwime["rqwime"]
                rqwime --> ime

                rqwm["rqwm"]
                rqwm --> hoec
                rqwm --> scc
                rqwm --> ime

                rqwmodbus["rqwmodbus"]
                rqwmodbus --> hoem

                rqwscc["rqwscc"]
                rqwscc --> scc

            end 
        end 

        rqw --> Qt6
    end
    Module --> fundation
  end


  Notedsl["说明: dsl 封装了一些复杂的数据结构"] --> dsl

  Notehoec["说明：hoec 已经弃用请使用hoecRefactor"] --> hoec
  
  class Notedsl,Notehoec noteStyle
```