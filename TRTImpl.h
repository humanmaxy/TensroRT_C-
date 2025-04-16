#pragma once
#include "NvInfer.h"
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <iostream>
#include <unordered_map>
#include<opencv2/opencv.hpp>
#include <NvInferRuntime.h>

#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}
class Logger : public nvinfer1::ILogger {
public:
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override;
};


class TRTImpl {
public:
    TRTImpl(const std::string& engine_path);
    ~TRTImpl();
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T>;
    cv::Mat doInference(const cv::Mat& inputMat);
    //cv::Mat doInference(const cv::Mat& input);
private:
    //std::vector<void*> buffers;
    cudaStream_t stream;

    nvinfer1::IRuntime* runtime;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;
    std::shared_ptr<nvinfer1::IRuntime> mRuntime;   //!< The TensorRT runtime used to deserialize the engine
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
    
    //std::shared_ptr<nvinfer1::IRuntime> mRuntime;   //!< The TensorRT runtime used to deserialize the engine
    //std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network
    int inputIndex;
    int outputIndex;
    int inputSize = 1;
    int outputSize = 1;
    const int channels = 3;
    const int inputH = 512;
    const int inputW = 960;
    const int out_channels = 5;  // Àà±ðÊý=5
    const int height = 512;   // 512
    const int width = 960;    // 960
    //struct InferDeleter
    //{
    //    template <typename T>
    //    void operator()(T* obj) const
    //    {
    //        delete obj;
    //    }
    //};
    Logger logger;
};