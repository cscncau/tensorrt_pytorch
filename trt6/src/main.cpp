
#include "opencv2/opencv.hpp"
#include <FeatureExtract.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdlib.h>
#include <sys/time.h>
#include <thread>
#include <time.h>
#include <unistd.h>
#include <time.h>
#include <memory.h>
#include <dirent.h>
#include <vector>
using namespace std;
vector<string> getFilesList(string dir);
static pthread_barrier_t barrier;

using namespace cv;
static const int img_num = 1;
static const int INPUT_H = 256;
static const int INPUT_W = 256;
static const int INPUT_C = 3;
static const int INPUT_SIZE = INPUT_H * INPUT_W * INPUT_C;
static const int OUTPUT_SIZE = 2;

double elasped_time = 0;

struct parameters
{
  std::string serialize_output_path;
  float *input_data;
  float *output_feature;
  int img_num;
  std::string filename;
};

void *infer(void *parameter)
{
  float *output_feature = new float[OUTPUT_SIZE * img_num];
  clock_t start, finish; 
  double duration; 
  struct parameters *my_data;
  my_data = (struct parameters *)parameter;
  FeatureExtract extract(my_data->serialize_output_path);
  start = clock();
  extract.doInference(my_data->input_data, my_data->output_feature, my_data->img_num);
  finish  = clock();
  duration = (double)(finish - start) / CLOCKS_PER_SEC;
  elasped_time += duration;
  cudaMemcpyAsync(output_feature, my_data->output_feature, my_data->img_num * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

  //float result = output_feature[OUTPUT_SIZE];
  /*
  if (output_feature[0] > output_feature[1])
      std::cout<<my_data->filename<<" "<<0<<" "<<output_feature[0]<<" "<<output_feature[1]<<std::endl;
  else
      std::cout<<my_data->filename<<" "<<1<<" "<<output_feature[0]<<" "<<output_feature[1]<<std::endl;
  */
  return 0;
}

int main(int argc, char **argv)
{
  pthread_barrier_init(&barrier, NULL, 4);
  clock_t start, finish;
  std::string path;
  path = "googlenet.txt";
  
  std::string dir="../resized";
  vector<string>allFileName = getFilesList(dir);
  Mat img_src;
  //double elasped_time = 0;
  for (int k = 0; k< allFileName.size(); ++k)
  {
    string filename = allFileName.at(k);
    img_src = imread(filename,CV_LOAD_IMAGE_COLOR);
    float *input_data = new float[INPUT_SIZE * img_num];
    float *output_feature = new float[OUTPUT_SIZE * img_num];
    float *d_input_data = NULL;
    float *d_output_feature = NULL;
    //double elasped_time = 0;
    //std::cout<<filename<<std::endl;
    int c = 0;
    //std::cout<<img_src.cols<<" image cols "<<std::endl;
    for (int i = 0; i < (INPUT_H * INPUT_W); ++i)
    {
      input_data[INPUT_H * INPUT_W * 0 + i] = float(img_src.data[c + 2]/255.0*2-1 );
      input_data[INPUT_H * INPUT_W * 1 + i] = float(img_src.data[c + 1]/255.0*2-1 );
      input_data[INPUT_H * INPUT_W * 2 + i] = float(img_src.data[c]/255.0*2-1);
      c += 3;
    }
    //std::cout<<"image is transformed"<<std::endl;
    cudaMalloc((void **)&d_input_data, img_num * INPUT_SIZE * sizeof(float));
    cudaMalloc((void **)&d_output_feature, img_num * OUTPUT_SIZE * sizeof(float));
    cudaMemcpy(d_input_data, input_data, img_num * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    struct parameters para;
    //start = clock();
    double duration;
    para.serialize_output_path = path;
    para.input_data = d_input_data;
    para.output_feature = d_output_feature;
    para.img_num = img_num;
    para.filename = allFileName.at(k);
    //start = clock();
    infer((void *)&para);
    //finish = clock();
    //duration = (double)(finish - start) / CLOCKS_PER_SEC; 
    //elasped_time += duration;
    delete[] input_data;
    delete[] output_feature;
    cudaFree(d_input_data);
    cudaFree(d_output_feature);
  }
  std::cout<<"time: "<<elasped_time/600.0<<std::endl;
  return 0;
}


vector<string> getFilesList(string dirpath){
    DIR *dir = opendir(dirpath.c_str());
    if (dir == NULL)
    {
        cout << "opendir error" << endl;
    }
    vector<string> allPath;
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL)
    {
        if (entry->d_type == DT_DIR){//It's dir
            if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
                continue;
            string dirNew = dirpath + "/" + entry->d_name;
            vector<string> tempPath = getFilesList(dirNew);
            allPath.insert(allPath.end(), tempPath.begin(), tempPath.end());
        }else {
            string name = entry->d_name;
            string imgdir = dirpath +"/"+ name;
            allPath.push_back(imgdir);
        }
    }
    closedir(dir);
    return allPath;
}

