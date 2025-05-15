@echo off
REM If you need to activate a conda environment, uncomment and modify the next line
REM call conda activate yolov11

REM Run trtexec to convert ONNX to TensorRT engine
.\trtexec.exe --onnx=.\runs\detect\train\weights\best.onnx --saveEngine=D:\zfkjData\ButtonScanner\ModelStorage\Temp\customOO.engine

REM Define the directory where the engine file is saved
set ENGINE_DIR=D:\zfkjData\ButtonScanner\ModelStorage\Temp

REM Define the source file and target files
set SOURCE_FILE=%ENGINE_DIR%\customOO.engine
set TARGET_FILE1=%ENGINE_DIR%\customOO1.engine
set TARGET_FILE2=%ENGINE_DIR%\customOO2.engine
set TARGET_FILE3=%ENGINE_DIR%\customOO3.engine
set TARGET_FILE4=%ENGINE_DIR%\customOO4.engine

REM Check if the source file exists
if exist "%SOURCE_FILE%" (
    REM Copy the source file to the target files
    copy "%SOURCE_FILE%" "%TARGET_FILE1%"
    copy "%SOURCE_FILE%" "%TARGET_FILE2%"
    copy "%SOURCE_FILE%" "%TARGET_FILE3%"
    copy "%SOURCE_FILE%" "%TARGET_FILE4%"

    REM Delete the original source file
    del "%SOURCE_FILE%"
    echo All operations completed successfully.
) else (
    echo Source file "%SOURCE_FILE%" does not exist. Exiting.
)