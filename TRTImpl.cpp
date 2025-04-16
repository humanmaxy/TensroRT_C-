#include "TRTImpl.h"
#include <fstream>
#include <stdexcept>
#include<iostream>
#include<filesystem>
#include"buffer.h"
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>    // 包含算术操作头文件
#include "trt_builder.hpp" 
template <typename T>
using SampleUniquePtr = std::unique_ptr<T>;
struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        delete obj;
    }
};
using namespace std;
//using samplesCommon::SampleUniquePtr;
void gpu_processing(const cv::Mat& input, float* hostBuffer, int h, int w) {
    // 上传数据到GPU
    cv::cuda::GpuMat d_input, d_high, d_low, d_global;
    d_input.upload(input);

    // 创建CUDA流实现异步操作
    cv::cuda::Stream stream;

    // 分解高/低字节（CUDA核函数）
    cv::cuda::GpuMat d_high_byte, d_low_byte;
    d_input.convertTo(d_high_byte, CV_8UC1, 1.0 / 255, 0, stream); // 高字节
    d_input.convertTo(d_low_byte, CV_8UC1, 1.0, 0, stream);     // 低字节
    cv::cuda::bitwise_and(d_low_byte, cv::Scalar(0xFF), d_low_byte, cv::noArray(), stream);

    // 归一化处理
    cv::cuda::GpuMat d_high_norm, d_low_norm, d_global_norm;
    d_high_byte.convertTo(d_high_norm, CV_32F, 2.0 / 255, -1.0, stream);
    d_low_byte.convertTo(d_low_norm, CV_32F, 2.0 / 255, -1.0, stream);
    d_input.convertTo(d_global_norm, CV_32F, 2.0 / 65535, -1.0, stream);

    // 下载结果到CPU缓冲区
    d_high_norm.download(cv::Mat(h, w, CV_32F, hostBuffer), stream);
    d_low_norm.download(cv::Mat(h, w, CV_32F, hostBuffer + h * w), stream);
    d_global_norm.download(cv::Mat(h, w, CV_32F, hostBuffer + 2 * h * w), stream);

    stream.waitForCompletion();
}

void Logger::log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept {
    if (severity <= nvinfer1::ILogger::Severity::kWARNING) {
        std::cout << "TRT " << msg << std::endl;
    }
}
int getInputIndex(nvinfer1::ICudaEngine* engine, const char* inputName) {
    for (int i = 0; i < engine->getNbIOTensors(); ++i) {
        const char* name = engine->getIOTensorName(i);
        if (strcmp(name, inputName) == 0) {
            return i;
        }
    }
    throw std::runtime_error("Input tensor not found: " + std::string(inputName));
}
int getoutputIndex(nvinfer1::ICudaEngine* engine, const char* outputName) {
    for (int i = 0; i < engine->getNbIOTensors(); ++i) {
        const char* name = engine->getIOTensorName(i);
        if (strcmp(name, outputName) == 0) {
            return i;
        }
    }
    throw std::runtime_error("Output tensor not found: " + std::string(outputName));
}
samplesCommon::BufferManager* buffers;

TRTImpl::TRTImpl(const std::string& engine_path) :
    runtime(nullptr),
    engine(nullptr),
    context(nullptr),
    inputIndex(-1),
    outputIndex(-1)
{

    //if (engine_path.empty()) {
    //    throw std::runtime_error("引擎路径不能为空");
    //}
    //std::cout << "当前工作目录: " << std::filesystem::current_path() << std::endl;
    cudaStreamCreate(&stream);


    std::ifstream file(engine_path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("无法打开引擎文件 test.trt");
    }
    if (file.peek() == std::ifstream::traits_type::eof()) {
        throw std::runtime_error("文件为空");
    }
    if (!file.is_open()) {
        throw std::runtime_error("文件打开失败");
    }
    this->runtime = nvinfer1::createInferRuntime(logger);






    file.seekg(0, std::ios::end);
    std::streamoff fileSize = file.tellg();
    if (fileSize <= 0) {
        throw std::runtime_error("引擎文件为空或读取失败");
    }
    //size_t size =  file.tellg();
    file.seekg(0, std::ios::beg);
    size_t size = static_cast<size_t>(fileSize);

    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);
    if (file.gcount() != size) {
        throw std::runtime_error("文件读取不完整");
    }
    file.close();

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()),
        InferDeleter());

    buffers = new samplesCommon::BufferManager(mEngine);

    cudaStreamCreate(&stream);


    context = mEngine->createExecutionContext();
    if (!context)
    {
        std::cout << "empty!!!!!!!!!!!!!!!!!" << std::endl;
    }
    for (int32_t i = 0, e = mEngine->getNbIOTensors(); i < e; i++)
    {
        auto const name = mEngine->getIOTensorName(i);
        // context->setTensorAddress(name, buffers.getDeviceBuffer(name));
        context->setTensorAddress(name, buffers->getDeviceBuffer(name));
    }

}


TRTImpl::~TRTImpl() {




    if (stream) {
        cudaStreamDestroy(stream);
    }


    if (context) {
        delete context;
        context = nullptr;
    }

    if (engine) {
        delete engine;
        engine = nullptr;
    }

    if (runtime) {
        delete runtime;
        runtime = nullptr;
    }
    if (buffers) {
        delete buffers;
        buffers = nullptr;
    }
}


cv::Mat TRTImpl::doInference(const cv::Mat& inputMat) {

    const uint16_t* imageData = reinterpret_cast<uint16_t*>(inputMat.data);

    float* hostDataBuffer = static_cast<float*>(buffers->getHostBuffer("input"));

    gpu_processing(inputMat, hostDataBuffer, 512, 960);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    std::cout << "{start} TRT caculation" << std::endl;
    buffers->copyInputToDevice();
    bool status = context->executeV2(buffers->getDeviceBindings().data());
    buffers->copyOutputToHost();
    std::cout << "\t TRT caculation TIME = " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count() << "[ms]" << std::endl;

    float* res = static_cast<float*>(buffers->getHostBuffer("output"));



    // 创建标签矩阵和彩色图像
 /*   cv::Mat labelMat(height, width, CV_8UC1);*/
    cv::Mat resultImg(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat labelMat(height, width, CV_32FC(5), cv::Scalar::all(0.0f));
    cv::Mat label(height, width, CV_8UC1);
    // 并行化像素处理
#pragma omp parallel for collapse(2)
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            // CHW布局下计算基础偏移量
            const size_t pixelOffset = h * width + w;  // 当前像素在单个通道内的位置
            float* label_ptr = labelMat.ptr<float>(h, w);
            // 遍历所有通道寻找最大值
            int maxIdx = 0;
            float maxVal = res[pixelOffset];  // 初始化为通道0的数据
            for (int c = 1; c < out_channels; ++c) {
                // 每个通道的数据块大小为height*width
                const size_t channelOffset = c * height * width;
                const float current = res[channelOffset + pixelOffset];
                const size_t offset = c * height * width + h * width + w;
                const float current1 = res[offset];

                // 直接写入浮点通道数据
                label_ptr[c] = current1;
                if (current > maxVal) {
                    maxVal = current;
                    maxIdx = c;
                }
            }
            const size_t baseLabelOffset = (h * width + w) * 5;
            labelMat.data[baseLabelOffset] = res[pixelOffset];
           
        }
    }

    return labelMat;
}
