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
#include <vector>

class FeatureExtract
{
public:
    FeatureExtract(std::string model_file);
    ~FeatureExtract();
public:
    bool doInference(float* d_src_data, float*  output, int image_num,  const cudaStream_t& stream = NULL);
private:
    bool initiallize(std::string model_file);
    bool doInference(nvinfer1::IExecutionContext& context, float* input, float* output, int batchSize, const cudaStream_t& stream);
private:
    nvinfer1::IExecutionContext* m_context;
    nvinfer1::ICudaEngine* m_engine;
    nvinfer1::IRuntime* m_runtime;
};

#endif
