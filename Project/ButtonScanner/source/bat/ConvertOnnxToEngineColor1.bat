@echo off
REM If you need to activate a conda environment, uncomment and modify the next line
REM call conda activate yolov11

REM Run trtexec to convert ONNX to TensorRT engine
.\trtexec.exe --onnx=.\runs\detect\train\weights\best.onnx --saveEngine=D:\zfkjData\ButtonScanner\ModelStorage\Temp\customOO1.engine