//
//  main.cpp
//  NeuralNetwork
//
//  Created by Dongjin Kim on 5/8/21.
//  Copyright © 2021 Dongjin Kim. All rights reserved.
//

#include <iostream>
#include <vector>
#include <cstdio>
#include <fstream>
#include <ostream>
#include <streambuf>
#include <ctime>

#include "Matrix.hpp"
#include "Math.hpp"
#include "NeuralNetwork.hpp"

using namespace std;

int main(int argc, const char * argv[]) {
    vector<double> input;
    input.push_back(0.3);
    input.push_back(0.8);
    input.push_back(0.1);
    input.push_back(0.5);
    input.push_back(0.2);
    input.push_back(0.9);
    input.push_back(0.0);
    input.push_back(0.5);
    input.push_back(0.2);
    input.push_back(1.0);
    
    
    vector<double> target;
    target.push_back(1);
    target.push_back(2);
    target.push_back(3);
    
    double learningRate  = 0.005;
    double momentum      = 1;
    double bias          = 0.1;
    int epochs = 500;
    
    vector<int> topology;
    topology.push_back(10);
    topology.push_back(10);
    topology.push_back(3);
    
    
    NeuralNetwork *n = new NeuralNetwork(topology,2, 2, 1, bias, learningRate, momentum);
    
    cout<<"Start Training: "<<endl;
    n->train(input, target, bias, learningRate, momentum, epochs);
    cout<<"Finished Training."<<endl;
    
    return 0;
}
