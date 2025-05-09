#pragma once
#include<onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

struct Detection
{
    float conf;
    int class_id;
    Rect bbox;
};

class YOLOv11
{

public:
    bool trt = 0;
    Ort::Env env;
    Ort::Session session= Ort::Session(nullptr);
    std::vector<Ort::Value>output_tensors;
    std::vector<const char*> input_node_names;
    std::vector<const char*> output_node_names;
    Ort::Value input_tensor = Ort::Value(nullptr);
    std::vector<Ort::Value> ort_inputs;
    cv::Mat infer_image;
    YOLOv11(string model_path,bool trt);
    ~YOLOv11();

    void preprocess(Mat& image);
    void infer();
    void postprocess(vector<Detection>& output);
    void draw(Mat& image, const vector<Detection>& output);

private:
    void init(std::string engine_path);

    float* gpu_buffers[2];               //!< The vector of device buffers needed for engine execution
    float* cpu_output_buffer;

    // Model parameters
    int input_w;
    int input_h;
    int num_detections;
    int detection_attribute_size;
    int num_classes = 80;
    float conf_threshold = 0.3f;
    float nms_threshold = 0.4f;

    vector<Scalar> colors;
};