//
//  Forward.cpp
//  CNN
//
//  Created by Dongjin Kim on 5/11/21.
//  Copyright © 2021 Dongjin Kim. All rights reserved.
//

#include "Forward.hpp"
#include "./utils/Functions.hpp"
#include "common.hpp"

void convolution(vector<vector<vector<double>>>&result, vector<vector<vector<double>>> image, vector<vector<vector<vector<double>>>> filter, vector<vector<double>> bias, int stride){
    //img => e.g. 128x128x3, filter => e.g. 5x5x40, output dim upon conv = [(W−K)/S]+1 if utilizing zero padding where W=128, K=5, S=stride
    
    //filter dims (num_filters, img_channel, kernel1, kernel2)
    //img dims (channel, height, width)
    //Here, height=width and kernel1=kernel2 are assumed
    int num_filters = filter.size();
    int filt_img_channel = filter[0].size();
    int kernel = filter[0][0].size();
    int channel = image.size();
    int height = image[0].size();
    int width = image[0][0].size();
    int out_dim = (int)((height-kernel)/stride)+1;
    if (channel != filt_img_channel)
    {
      cout << "ERROR: Dimensions of Filter and Image must match!" << endl;
    }
    result = vector<vector<vector<double>>>(num_filters, vector<vector<double>>(out_dim, vector<double>(out_dim,0)));
    
    
    for(int curr_f=0;curr_f<num_filters;curr_f++){ //For each filter,
        //first conv on y dims
        int curr_y = 0;
        int out_y = 0;
        while(curr_y+kernel<=height){
            //conv on x dims
            int curr_x=0;
            int out_x = 0;
            while(curr_x+kernel<=width){
                vector<vector<vector<double>>> focusedSection;
                for(int k=0; k<filt_img_channel;k++){//for each image channel
                    vector<vector<double>> temp_y;
                    for(int j=curr_y;j<curr_y+kernel;j++){
                        vector<double> temp_x;
                        for(int i=curr_x;i<curr_x+kernel;i++){
                            temp_x.push_back(image[k][j][i]);
                        }
                        temp_y.push_back(temp_x);
                    }
                    focusedSection.push_back({temp_y});
                }
                //This focusedSection's each grid is multiplied by each corresponding kernel grids, and all summed up
                vector<vector<vector<double>>> res2;
                multMatrices3D(res2, filter[curr_f], focusedSection);
                sum3D(res2, res2, 0);
                //additing the bias terms
                res2[0][0][0]+=bias[curr_f][0];
                result[curr_f][out_y][out_x] = res2[0][0][0];
                curr_x+=stride;
                out_x++;
            }
            curr_y+=stride;
            out_y++;
        }
    }
}

void ReLU(vector<vector<vector<double>>>&result){
    for (int i = 0; i < result.size(); i++)
    {
      for (int j = 0; j < result[0].size(); j++)
      {
        for (int k = 0; k < result[0][0].size(); k++)
        {
          if (result[i][j][k] < 0)
          {
            result[i][j][k] = 0;
          }
        }
      }
    }
}

void ReLU2D(vector<vector<double>>&result){
    for (int i = 0; i < result.size(); i++)
    {
      for (int j = 0; j < result[0].size(); j++)
      {
        if (result[i][j] < 0)
        {
          result[i][j] = 0;
        }
      }
    }
}
    
void maxpool(vector<vector<vector<double>>>&result, vector<vector<vector<double>>> image, int size, int stride){
    int channel = image.size();
    int height = image[0].size();
    int width = image[0][0].size();
    int h = (int)((height-size)/stride)+1;
    int w = (int)((width-size)/stride)+1;
    result = vector<vector<vector<double>>>(channel, vector<vector<double>>(h, vector<double>(w,0)));
    for(int c=0;c<channel;c++){
        int curr_y = 0;
        int out_y = 0;
        while(curr_y+size<=height){
            int curr_x = 0;
            int out_x = 0;
            while(curr_x+size<=width){
                vector<vector<double>>temp_y;
                for(int j=curr_y;j<curr_y+size;j++){
                    vector<double> temp_x;
                    for(int i=curr_x;i<curr_x+size;i++){
                        temp_x.push_back(image[c][j][i]);
                    }
                    temp_y.push_back({temp_x});
                }
                //For each focused section,
                double max = temp_y[0][0];
                for(int j=0;j<temp_y.size();j++){
                    for(int i=0;i<temp_y[0].size();i++){
                        if(temp_y[j][i]>max){
                            max = temp_y[j][i];
                        }
                    }
                }
                result[c][out_y][out_x] = max;
                curr_x+=stride;
                out_x++;
            }
            curr_y+=stride;
            out_y++;
        }
    }
}

void softmax(vector<vector<double>>&result,vector<vector<double>>X){
    int row = X.size();
    int col = X[0].size();
    result = vector<vector<double>>(row, vector<double>(col, 0));
    for (int i = 0; i < row; i++)
    {
      for (int j = 0; j < col; j++)
      {
        result[i][j] = exp(X[i][j]);
      }
    }
    vector<vector<double>> sumAll;
    sum2D(sumAll, result, 0);
    for (int i = 0; i < row; i++)
    {
      for (int j = 0; j < col; j++)
      {
        result[i][j] /= sumAll[0][0];
      }
    }
}

void categoricalCrossEntropy(double &result, vector<vector<double>>probs, vector<vector<double>> label){
    int row = probs.size();
    int col = probs[0].size();
    if (row != label.size() or col!=label[0].size())
    {
      cout << "ERROR: Dimensions of logits and labels must match for CategoricalCrossEntropy!" << endl;
    }
    vector<vector<double>> temp(row, vector<double>(col, 0));
    for(int i=0;i<row;i++){
        for(int j=0;j<col;j++){
            temp[i][j] = log(probs[i][j]); //predictions
        }
    }
    //total sum of each target*log(prediction)
    multMatrices2D(temp, temp, label);
    sum2D(temp, temp, 0);
    result = -1*temp[0][0];
}
