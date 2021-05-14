//
//  Backward.cpp
//  CNN
//
//  Created by Dongjin Kim on 5/12/21.
//  Copyright © 2021 Dongjin Kim. All rights reserved.
//

#include "Backward.hpp"
#include "./utils/Functions.hpp"
#include "common.hpp"

void convolutionBackward(vector<vector<vector<double>>>&result, vector<vector<vector<vector<double>>>>&df, vector<vector<double>>&db, vector<vector<vector<double>>> dconv_prev, vector<vector<vector<double>>> conv_in,vector<vector<vector<vector<double>>>> filter, int stride ){
    
    int num_filters = filter.size();
    int filt_img_channel = filter[0].size();
    int kernel = filter[0][0].size();
    int origHeight = conv_in[0].size();
    int origWidth = conv_in[0][0].size();
    result = vector<vector<vector<double>>>(conv_in.size(),vector<vector<double>>(origHeight,vector<double>(origWidth,0)));
    vector<vector<vector<vector<double>>>> dfilt(num_filters,vector<vector<vector<double>>>(filt_img_channel,vector<vector<double>>(kernel,vector<double>(filter[0][0][0].size(),0))));
    vector<vector<double>> dbias;
    
    for(int curr_f=0;curr_f<num_filters;curr_f++){
        int curr_y=0;
        int out_y=0;
        while(curr_y+kernel<=origHeight){
            int curr_x=0;
            int out_x=0;
            while (curr_x+kernel<=origWidth) {
                vector<vector<vector<double>>> focusedSection;
                for(int k=0;k<filt_img_channel;k++){
                    vector<vector<double>>temp_y;
                    for(int j=curr_y;j<curr_y+kernel;j++){
                        vector<double>temp_x;
                        for(int i=curr_x;i<curr_x+kernel;i++){
                            temp_x.push_back(conv_in[k][j][i]);
                        }
                        temp_y.push_back(temp_x);
                    }
                    focusedSection.push_back({temp_y});
                }
                vector<vector<vector<double>>> res;
                //For each kernel-sized window (translated as one output grid), 
                mult3D(res, focusedSection, dconv_prev[curr_f][out_y][out_x]);
                add3D(dfilt[curr_f], dfilt[curr_f], res);
                //With mult3D and add3D, dL/dF = essentially, conv(conv_in, dconv_prev)
                
                
                //Now, let's continue to get dL/dX, which will become dconv_prev for the previous convolution layer.
                mult3D(res, filter[curr_f], dconv_prev[curr_f][out_y][out_x]);
                for(int k=0;k<filt_img_channel;k++){
                    for(int j=curr_y,j2=0;j<curr_y+kernel;j++,j2++){
                        for(int i=curr_x,i2=0;i<curr_x+kernel;i++,i2++){
                            result[k][j][i] += res[k][j2][i2];
                        }
                    }
                }
                //this res input is acting like a sliding window, and applied to the entire input to the conv layer.
                curr_x+=stride;
                out_x++;
            }
            curr_y+=stride;
            out_y++;
        }
        vector<vector<double>> temp_b;
        sum2D(temp_b, dconv_prev[curr_f], 0);
        //dbias equivalent to sum of each filter's dconv_prev
        dbias.push_back({temp_b[0][0]});
    }
    df = dfilt;
    db = dbias;
}

void maxpoolBackward(vector<vector<vector<double>>>&result, vector<vector<vector<double>>>&dpool, vector<vector<vector<double>>> orig, int size, int stride){
    int channel = orig.size();
    int height = orig[0].size();
    int width = orig[0][0].size();
    result = vector<vector<vector<double>>> (channel, vector<vector<double>>(height, vector<double>(width,0)));
    
    for(int curr_c;curr_c<channel;curr_c++){
        int curr_y=0;
        int out_y=0;
        while(curr_y+size<=height){
            int curr_x=0;
            int out_x=0;
            while(curr_x+size<=width){
                vector<vector<double>>temp_y;
                for(int j=curr_y;j<curr_y+size;j++){
                    vector<double>temp_x;
                    for(int i=curr_x;i<curr_x+size;i++){
                        temp_x.push_back(orig[curr_c][j][i]);
                    }
                    temp_y.push_back({temp_x});
                }
                vector<int> maxIndex{0,0};
                double max = temp_y[0][0];
                for(int j=0;j<temp_y.size();j++){
                    for(int i=0;i<temp_y[0].size();j++){
                        if(temp_y[j][j]>max){
                            max = temp_y[j][i];
                            maxIndex[0] = j;
                            maxIndex[1] = i;
                        }
                    }
                }
                //only the max indices are updated with gradients
                //rest are zeros.
                result[curr_c][curr_y+maxIndex[0]][curr_x+maxIndex[1]] = dpool[curr_c][out_y][out_x];
                curr_x+=stride;
                out_x++;
            }
            curr_y+=stride;
            out_y++;
        }
    }
}
