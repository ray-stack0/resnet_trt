#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include <fstream>
#include <map>
#include <chrono>
#include <iostream>
#include <cmath>
#include "include/logging.h"
#include <opencv4/opencv2/opencv.hpp>

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: "<< ret << std::endl;\
            abort();\
        }\
    } while(0)

static const int INPUT_H = 224;
static const int INPUT_W = 224;
static const int OUTPUT_SIZE = 1000;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";


using namespace nvinfer1;

static Logger gLogger;

/**
 * @brief 加载权重
 */
std::map<std::string, Weights> loadWeights(const std::string file)
{
    std::cout<<"Loading weights: "<<file<<std::endl;
    std::map<std::string,Weights> weightMap;

    //* 1.打开权重文件
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    //* 2.读取文件
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    std::cout<<"resnet50 parameters num: "<<count<<std::endl;
    while (count--)
    {
        Weights wt{DataType::kFLOAT,nullptr,0};
        uint32_t size;
        std::string name;

        input >> name >> std::dec >> size; // std::dec 按10进制读取
        wt.type = DataType::kFLOAT;
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val)*size));
        for (uint32_t x = 0; x < size; ++x)
        {
            input >> std::hex >> val[x]; // 按16进制读取
        }
        wt.count = size;
        wt.values = val;
        weightMap[name] = wt;
    }

    return weightMap;
    
}

/**
 * @brief param 使用scale layer创建一个2D的batchnorm层.
 */
IScaleLayer* addBatchNorm2d(INetworkDefinition* network,std::map<std::string, Weights>& weightMap,
                            ITensor& input,std::string layername,float eps)
{
    float* gamma = (float*)weightMap[layername+".weight"].values;
    float* beta = (float*)weightMap[layername+".bias"].values;
    float* mean = (float*)weightMap[layername+".running_mean"].values;
    float* var = (float*)weightMap[layername+".running_var"].values;
    
    int len = weightMap[layername+".weight"].count;
    float* scval = reinterpret_cast<float*>(malloc(sizeof(float)*len));
    float* shval = reinterpret_cast<float*>(malloc(sizeof(float)*len));
    float* pval = reinterpret_cast<float*>(malloc(sizeof(float)*len));

    for (int i=0; i < len; i++)
    {
        // scale参数
        scval[i] = gamma[i]/sqrt(var[i]+eps);
        // shift参数
        shval[i] = beta[i] - scval[i] * mean[i];
        // power参数
        pval[i] = 1.0;
    }
    Weights scale{DataType::kFLOAT,scval,len};
    Weights shift{DataType::kFLOAT,shval,len};
    Weights power{DataType::kFLOAT,pval,len};
    weightMap[layername+".scale"] = scale;
    weightMap[layername+".shift"] = shift;
    weightMap[layername+".power"] = power;

    // scale layer
    // (x*scale + shift)^power
    IScaleLayer* scalelayer = network->addScale(input,ScaleMode::kCHANNEL,shift,scale,power);
    assert(scalelayer);
    return scalelayer;
}


/**
 * @brief 创建bottleneck结构
 * @param inch 第一个巻积层输入
 * @param outch 第一个巻积层输出
 */
IActivationLayer* addBottleneck(INetworkDefinition* network,std::map<std::string,Weights>& weightmap,
                                ITensor& input,int inch,int outch,int stride,std::string layername)
{
    std::cout<<"layername: "<<layername<<std::endl;
    std::cout<<"inch: "<<inch<<", outch: "<<outch<<std::endl;
    Weights emptywts{DataType::kFLOAT,nullptr,0};
    IConvolutionLayer* conv1 = network->addConvolutionNd(input,outch,Dims2(1,1),
                             weightmap[layername+".conv1.weight"],emptywts);
    assert(conv1);
    IScaleLayer* bn1 = addBatchNorm2d(network,weightmap,*conv1->getOutput(0),layername+".bn1",1e-5);
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0),ActivationType::kRELU);

    IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0),outch,Dims2(3,3),weightmap[layername+".conv2.weight"],emptywts);
    conv2->setStrideNd(Dims2(stride,stride));
    conv2->setPaddingNd(Dims2(1,1));
    Dims dims2 = conv2->getOutput(0)->getDimensions();
    std::cout<<"conv2:"<<std::endl;
    for (auto dim : dims2.d)
    {
        std::cout<<dim<<",";
    }
    std::cout<<std::endl;;
    IScaleLayer* bn2 = addBatchNorm2d(network,weightmap,*conv2->getOutput(0),layername+".bn2",1e-5);
    IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0),ActivationType::kRELU);
    std::cout<<"conv3:"<<std::endl;
    IConvolutionLayer* conv3 = network->addConvolutionNd(*relu2->getOutput(0),outch*4,Dims2(1,1),weightmap[layername+".conv3.weight"],emptywts);
    Dims dims3 = conv3->getOutput(0)->getDimensions();
    for (auto dim : dims3.d)
    {
        std::cout<<dim<<",";
    }
    IScaleLayer* bn3 = addBatchNorm2d(network,weightmap,*conv3->getOutput(0),layername+".bn3",1e-5);


    IElementWiseLayer* add_layer;
    if (stride != 1 || inch != 4*outch)
    {
        IConvolutionLayer* conv4 = network->addConvolutionNd(input,outch*4,Dims2(1,1),weightmap[layername+".downsample.0.weight"],emptywts);
        conv4->setStrideNd(Dims2(stride,stride));
        IScaleLayer* bn4 = addBatchNorm2d(network,weightmap,*conv4->getOutput(0),layername+".downsample.1",1e-5);
        add_layer = network->addElementWise(*bn4->getOutput(0),*bn3->getOutput(0),ElementWiseOperation::kSUM);
        
    }
    else
    {
        add_layer = network->addElementWise(input,*bn3->getOutput(0),ElementWiseOperation::kSUM);
    }
    IActivationLayer* relu3 = network->addActivation(*add_layer->getOutput(0),ActivationType::kRELU);
    assert(relu3);
    std::cout<<std::endl;
    return relu3;
}


/**
 * @brief 创建engine
 */
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt)
{
    //* 1.创建空网络,默认参数
    INetworkDefinition* network = builder->createNetworkV2(0U);

    //* 2.设置输入
    ITensor* data = network->addInput(INPUT_BLOB_NAME,dt,Dims3{3,INPUT_H,INPUT_W});
    assert(data);

    //* 3.读取权重文件
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    std::map<std::string, Weights> weightMap = loadWeights("../python_scripts/resnet50.wts"); // imagenet 1000训练

    //* 4.创建网络
    IConvolutionLayer* conv1 = network->addConvolutionNd(*data,64,Dims2(7,7),weightMap["conv1.weight"],emptywts);
    assert(conv1);
    conv1->setStrideNd(Dims2(2,2));
    conv1->setPaddingNd(Dims2(3,3));
    
    IScaleLayer* bn1 = addBatchNorm2d(network,weightMap,*conv1->getOutput(0),"bn1",1e-5);

    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0),ActivationType::kRELU);

    IPoolingLayer* maxpool1 = network->addPoolingNd(*relu1->getOutput(0),PoolingType::kMAX,Dims2(3,3));
    maxpool1->setStrideNd(Dims2(2,2));
    maxpool1->setPaddingNd(Dims2(1,1));

    IActivationLayer* bottleneck_out = addBottleneck(network,weightMap,*maxpool1->getOutput(0),64,64,1,"layer1.0");
    bottleneck_out = addBottleneck(network,weightMap,*bottleneck_out->getOutput(0),256,64,1,"layer1.1");
    bottleneck_out = addBottleneck(network,weightMap,*bottleneck_out->getOutput(0),256,64,1,"layer1.2");

    bottleneck_out = addBottleneck(network,weightMap,*bottleneck_out->getOutput(0),256,128,2,"layer2.0");
    bottleneck_out = addBottleneck(network,weightMap,*bottleneck_out->getOutput(0),512,128,1,"layer2.1");
    bottleneck_out = addBottleneck(network,weightMap,*bottleneck_out->getOutput(0),512,128,1,"layer2.2");
    bottleneck_out = addBottleneck(network,weightMap,*bottleneck_out->getOutput(0),512,128,1,"layer2.3");

    bottleneck_out = addBottleneck(network,weightMap,*bottleneck_out->getOutput(0),512,256,2,"layer3.0");
    bottleneck_out = addBottleneck(network,weightMap,*bottleneck_out->getOutput(0),1024,256,1,"layer3.1");
    bottleneck_out = addBottleneck(network,weightMap,*bottleneck_out->getOutput(0),1024,256,1,"layer3.2");
    bottleneck_out = addBottleneck(network,weightMap,*bottleneck_out->getOutput(0),1024,256,1,"layer3.3");
    bottleneck_out = addBottleneck(network,weightMap,*bottleneck_out->getOutput(0),1024,256,1,"layer3.4");
    bottleneck_out = addBottleneck(network,weightMap,*bottleneck_out->getOutput(0),1024,256,1,"layer3.5");
    
    bottleneck_out = addBottleneck(network,weightMap,*bottleneck_out->getOutput(0),1024,512,2,"layer4.0");
    bottleneck_out = addBottleneck(network,weightMap,*bottleneck_out->getOutput(0),2048,512,1,"layer4.1");
    bottleneck_out = addBottleneck(network,weightMap,*bottleneck_out->getOutput(0),2048,512,1,"layer4.2");

    IPoolingLayer* pool2 = network->addPoolingNd(*bottleneck_out->getOutput(0),PoolingType::kAVERAGE,Dims2(7,7));
    assert(pool2);
    pool2->setStrideNd(Dims2(1,1));

    IFullyConnectedLayer* fc = network->addFullyConnected(*pool2->getOutput(0),1000,weightMap["fc.weight"],weightMap["fc.bias"]);
    assert(fc);

    //* 5.设置输出
    fc->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*fc->getOutput(0));

    //* 6.创建engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(1 << 20);
    ICudaEngine* engine = builder->buildEngineWithConfig(*network,*config);

    //* 7.释放权重文件占用的内存
    for (auto& mem : weightMap)
    {
        free((void*)(mem.second.values));
    }
    return engine;
}



/**
 * @brief 创建序列化的模型.
 * @param maxBatchSize 
 * @param modelStream 指向模型流指针的指针
 */
void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream)
{
    //* 创建engine
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    ICudaEngine* engine = createEngine(maxBatchSize,builder,config,DataType::kFLOAT);
    assert(engine != nullptr);

    (*modelStream) = engine->serialize();

    //* 释放缓存
    engine->destroy();
    builder->destroy();
    config->destroy();
}
/**
 * @brief 执行推理
 */
void doInference(IExecutionContext& context, float* input, float* output, int batchsize)
{
    const ICudaEngine& engine = context.getEngine();

    assert(engine.getNbBindings() == 2); // 1个输入1个输出
    void* buffers[2];

    // 根据搭建网络时设置的输入输出名称获得其对应的索引
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // 在GPU上给输入输出分配内存
    // 将buffers分别指向GPU上的地址
    CHECK(cudaMalloc(&buffers[inputIndex],batchsize*3*INPUT_H*INPUT_W*sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex],batchsize*OUTPUT_SIZE*sizeof(float)));

    // 创建stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // 将数据从CPU复制到GPU上指定位置
    CHECK(cudaMemcpyAsync(buffers[inputIndex],input,batchsize*3*INPUT_H*INPUT_W*sizeof(float),cudaMemcpyHostToDevice,stream));
    // 执行推理
    context.enqueue(batchsize,buffers,stream,nullptr);
    // 将输出从GPU复制到CPU
    CHECK(cudaMemcpyAsync(output,buffers[outputIndex],OUTPUT_SIZE*batchsize*sizeof(float),cudaMemcpyDeviceToHost,stream));
    // 阻塞线程,直到执行完成.
    cudaStreamSynchronize(stream);
 
    // 释放资源
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[outputIndex]));
    CHECK(cudaFree(buffers[inputIndex]));
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::cerr<<"argument not right!"<<std::endl;
        std::cerr<<"./resnet50 -s // serialize model to plan file"<<std::endl;
        std::cerr <<"./lenet -d   // deserialize plan file and run inference"<<std::endl;
        return -1;
    }

    // 用于读取engine
    char* trtModelStream{nullptr};
    size_t size{0};

    if (std::string(argv[1]) == "-s")
    {
        //* 模型序列化并保存

        IHostMemory* modelStream{nullptr};
        APIToModel(1,&modelStream);
        assert(modelStream != nullptr);

        std::ofstream p("resnet50.engine",std::ios::binary);
        if (!p)
        {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()),modelStream->size());
        modelStream->destroy();
        return 1;
    }
    else if (std::string(argv[1]) == "-d")
    {
        //* 反序列化模型并推理

        std::ifstream file("resnet50.engine",std::ios::binary);
        if (file.good())
        {
            file.seekg(0,file.end); // 将文件指针移动到末尾
            size = file.tellg();
            file.seekg(0,file.beg); // 文件指针移动到开头
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream,size);
            file.close();
        }
    }
    else
        return -1;

    //* 推理
    
    // 创建输入
    std::string image_path = "../snowmobile.jpg";
    cv::Mat image = cv::imread(image_path,cv::IMREAD_COLOR);
    if (image.empty())
    {
        std::cerr<<"failed to load image!"<<std::endl;
        return -1;
    }
    if (image.channels() != 3)
    {
        std::cerr<<"This is likely a color image (RGB or BGR)."<<std::endl;
        return -1;
    }
    std::cout<<"image size: "<<image.rows<<", "<<image.cols<<std::endl;
    cv::Mat r_image;
    cv::resize(image,r_image,cv::Size(INPUT_H,INPUT_W));
    std::cout<<"resize image size: "<<r_image.rows<<", "<<r_image.cols<<std::endl;
    cv::imshow("Original Image", r_image);
    cv::waitKey(0);
    r_image.convertTo(image, CV_32FC3, 1.0 / 255.0);
    static float data[3*INPUT_H*INPUT_W];
    // TRT默认使用NCHW格式
    // OpenCV使用HWC
    // for (int y = 0; y < INPUT_H; ++y) {
    //     for (int x = 0; x < INPUT_W; ++x) {
    //         for (int c = 0; c < 3; ++c) {
    //             data[c * INPUT_H * INPUT_W + y * INPUT_H + x] = image.at<cv::Vec3f>(y, x)[c];
    //         }
    //     }
    // }
    for (int c = 0; c < 3; ++c)
    {
        for (int y = 0; y < INPUT_H; ++y)
        {
            for (int x = 0; x < INPUT_W; ++x)
            {
                data[c*INPUT_H*INPUT_W+y*INPUT_W+x] = image.at<cv::Vec3f>(y, x)[c];
            }
        }
    }
    

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    // 反序列化engine
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream,size,nullptr);
    assert(engine != nullptr);
    // 创建context
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    
    static float probs[OUTPUT_SIZE];
    auto start = std::chrono::system_clock::now();
    int inference_times = 100;
    for (int i = 0; i < inference_times; i++)
    {
        doInference(*context,data,probs,1);
    }
    auto end = std::chrono::system_clock::now();
    std::cout<<"inference "<<inference_times<<" times cost time: "
             <<std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()
             <<" ms."<<std::endl;
    std::cout<<"\nOutput:\n";

    int index;
    float prob = 0;
    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        if (probs[i] > prob)
        {
            index = i;
            prob = probs[i];
        }
    }
    std::cout<<"index: "<<index<<", max prob: "<<prob<<std::endl;
    

    return 0;
}