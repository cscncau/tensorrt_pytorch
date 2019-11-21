#include "FeatureExtract.h"

#include "NvOnnxParser.h"
#include "argsParser.h"
#include "logger.h"
#include "common.h"
using namespace nvinfer1;

samplesCommon::Args gArgs;

FeatureExtract::FeatureExtract(std::string model_file)
{
    initiallize(model_file);
}

FeatureExtract::~FeatureExtract() {
    if(m_context != nullptr) {
        m_context->destroy();
    }    

    if(m_engine != nullptr) {
        m_engine->destroy();
    } 

    if(m_runtime != nullptr) {
        m_runtime->destroy();
    } 
}

bool FeatureExtract::initiallize(std::string model_file)
{
    std::ifstream file_cache(model_file);
    if(!file_cache){
       printf("open %s is failed", model_file.c_str());
       return false;
    }
    std::string engine_data((std::istreambuf_iterator<char>(file_cache)), std::istreambuf_iterator<char>());

    m_runtime = createInferRuntime(gLogger);
    if(m_runtime == nullptr) {
        printf("create infer runtime is failed\n");
        return false;
    }
    if (gArgs.useDLACore >= 0)
        m_runtime->setDLACore(gArgs.useDLACore);
    m_engine = m_runtime->deserializeCudaEngine(engine_data.data(), engine_data.size(), nullptr);
    if(m_engine == nullptr) {
        printf("deserialize cuda engine is failed\n");
        m_runtime->destroy();
        return false;
    }
    
    m_context = m_engine->createExecutionContext();
    
    return true;
}

bool FeatureExtract::onnxToTRTModel(
    std::string model_file, 
    int batch_size, 
    IHostMemory*& trtModelStream)
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
    builder->setFp16Mode(false);
    builder->setInt8Mode(false);    
    samplesCommon::enableDLA(builder, gArgs.useDLACore);
    
    ICudaEngine* engine = builder->buildCudaEngine(*network);
    if(engine == nullptr) {
        printf("build cuda engine fail\n");
        return false;
    }
    parser->destroy();

    
    trtModelStream = engine->serialize();
    if(trtModelStream == nullptr) {
        printf("engine serialize is failed\n");
        engine->destroy();
	network->destroy();
        builder->destroy();
        return false;
    }
    engine->destroy();
    network->destroy();
    builder->destroy();

    return true;
}

bool FeatureExtract::doInference(
    float* d_src_data,
    float* d_feature_data, 
    int image_num, 
    const cudaStream_t& stream)
{
     return doInference(*m_context, d_src_data, d_feature_data, image_num, stream);
}

bool FeatureExtract::doInference(
    IExecutionContext& context, 
    float* input, 
    float* output, 
    int batchSize, 
    const cudaStream_t& stream) 
{
    const ICudaEngine& engine = context.getEngine();
    if (engine.getNbBindings() != 2) {
        printf("Nb Bindings of engine is not 2");
        return false;
    }
    
    void* buffers[2];
    for (int i=0; i < engine.getNbBindings(); ++i) {
        if (engine.bindingIsInput(i))
            buffers[i] = input;
        else
            buffers[i] = output;
    }

    context.enqueue(batchSize, buffers, stream, nullptr);

    return true;
}
