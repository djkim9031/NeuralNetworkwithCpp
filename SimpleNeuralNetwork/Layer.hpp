//
//  Layer.hpp
//  NeuralNetwork
//
//  Created by Dongjin Kim on 5/8/21.
//  Copyright © 2021 Dongjin Kim. All rights reserved.
//

#ifndef Layer_hpp
#define Layer_hpp

#include <iostream>
#include <vector>
#include "Matrix.hpp"
#include "Neuron.hpp"

using namespace std;

class Layer{
    int size;
    vector<Neuron *> neurons;
    
public:
    Layer(int size);
    Layer(int size, int activatationType);
    
    void setVal(int i, double v); //set a specific neuron's (ith) value (v)
    void setNeuron(vector<Neuron *> neurons){this->neurons = neurons;};
    
    vector<double> getActivatedVals();
    vector<Neuron *> getNuerons(){return this->neurons;};
    
    Matrix *matrixifyVals();
    Matrix *matrixifyActivatedVals();
    Matrix *matrixifyDerivedVals();
    
    int getSize(){return this->neurons.size();};
};

#endif /* Layer_hpp */
