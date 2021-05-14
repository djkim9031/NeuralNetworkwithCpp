//
//  Forward.hpp
//  CNN
//
//  Created by Dongjin Kim on 5/11/21.
//  Copyright © 2021 Dongjin Kim. All rights reserved.
//

#ifndef Forward_hpp
#define Forward_hpp

#include <vector>
using namespace std;

void convolution(vector<vector<vector<double>>>&result, vector<vector<vector<double>>> image, vector<vector<vector<vector<double>>>> filter, vector<vector<double>> bias, int stride=1);
void ReLU(vector<vector<vector<double>>>&result);
void ReLU2D(vector<vector<double>>&result);
void maxpool(vector<vector<vector<double>>>&result, vector<vector<vector<double>>> image, int size=2, int stride=2);
void softmax(vector<vector<double>>&result,vector<vector<double>>X);
void categoricalCrossEntropy(double &result, vector<vector<double>>probs, vector<vector<double>> label);

#endif /* Forward_hpp */
