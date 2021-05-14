//
//  Backward.hpp
//  CNN
//
//  Created by Dongjin Kim on 5/12/21.
//  Copyright © 2021 Dongjin Kim. All rights reserved.
//

#ifndef Backward_hpp
#define Backward_hpp

#include <vector>
using namespace std;

void convolutionBackward(vector<vector<vector<double>>>&result, vector<vector<vector<vector<double>>>>&df, vector<vector<double>>&db, vector<vector<vector<double>>> dconv_prev, vector<vector<vector<double>>> conv_in,vector<vector<vector<vector<double>>>> filter, int stride=1 );

void maxpoolBackward(vector<vector<vector<double>>>&result, vector<vector<vector<double>>>&dpool, vector<vector<vector<double>>> orig, int size=2, int stride=2);

#endif /* Backward_hpp */
