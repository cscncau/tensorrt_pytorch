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
#include <streambuf>

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "argsParser.h"
#include "logger.h"
#include "common.h"

using namespace nvinfer1;
samplesCommon::Args gArgs;

bool onnxToEngine(
    std::string model_file, 
    int batch_size, 
    std::string engine_file)
{
    IBuilder* builder = createInferBuilder(gLogger.getTRTLogger());
    if (builder == nullptr) {
        printf("create infer builder is failed\n");
        return false;
    }

    nvinfer1::INetworkDefinition* network = builder->createNetwork();
    auto parser = nvonnxparser::createParser(*network, gLogger.getTRTLogger());

    if (!parser->parseFromFile(model_file.c_str(), static_cast<int>(gLogger.getReportableSeverity()))) {
        gLogError << "Failure while parsing ONNX file" << std::endl;
        parser->destroy();
        return false;
    }
    
    builder->setMaxBatchSize(batch_size);
    builder->setMaxWorkspaceSize(1 << 30);
    builder->setFp16Mode(true);
    builder->setInt8Mode(false);    
    samplesCommon::enableDLA(builder, gArgs.useDLACore);
    
    ICudaEngine* engine = builder->buildCudaEngine(*network);
    if(engine == nullptr) {
        printf("build cuda engine fail\n");
        return false;
    }
    parser->destroy();

    
    IHostMemory* trtModelStream = engine->serialize();
    if(trtModelStream == nullptr) {
        printf("engine serialize is failed\n");
        engine->destroy();
	    network->destroy();
        builder->destroy();
        return false;
    }

    std::ofstream serialize_output_stream;
    std::string serialize_str;
    serialize_str.resize(trtModelStream->size());
    memcpy((void*)serialize_str.data(),trtModelStream->data(),trtModelStream->size());
    serialize_output_stream.open(engine_file);
    serialize_output_stream << serialize_str;
    serialize_output_stream.close();

    engine->destroy();
    network->destroy();
    builder->destroy();
    trtModelStream->destroy();

    return true;
}

int main(int argc, char** argv) {
   if(argc < 4) {
       printf("please input model path, max batch size and engine path\n");
       printf("./model_to_engine ./shufflenet_op9.onnx 256 ./shufflenet_engine.txt\n");
       return -1;
   }
   
    if(!onnxToEngine(argv[1], std::stoi(argv[2]), argv[3])){
        printf("onnx to engine is failed\n");
        return -1;
    }

   return 0;
}
