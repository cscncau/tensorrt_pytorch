#ifndef _FEATUREEXTRACT_
#define _FEATUREEXTRACT_

#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include <time.h>
#include "NvInfer.h"

class FeatureExtract
{
public:
    FeatureExtract(std::string model_file);
    ~FeatureExtract();
public:
    bool doInference(float* d_src_data, float* d_feature_data, int image_num, const cudaStream_t& stream = NULL);
private:
    bool initiallize(std::string model_file);
    bool onnxToTRTModel(std::string model_file, int batch_size, nvinfer1::IHostMemory*& trtModelStream);
    bool doInference(nvinfer1::IExecutionContext& context, float* input, float* output, int batchSize, const cudaStream_t& stream);
private:
    nvinfer1::IExecutionContext* m_context;
    nvinfer1::ICudaEngine* m_engine;
    nvinfer1::IRuntime* m_runtime;
};

#endif
