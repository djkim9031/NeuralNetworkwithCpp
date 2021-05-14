//
//  Neuron.hpp
//  NeuralNetwork
//
//  Created by Dongjin Kim on 5/8/21.
//  Copyright © 2021 Dongjin Kim. All rights reserved.
//

#ifndef Neuron_hpp
#define Neuron_hpp

#define TANH 1
#define RELU 2
#define SIGM 3

#include <iostream>
#include <cmath>

using namespace std;

class Neuron{
    double val;
    double activatedVal;
    double derivedVal;
    int activationType = 3;
public:
    Neuron(double val);
    Neuron(double val, int activationType);
    
    void setVal(double v);
    void activate();
    void derive();
    
    double getVal(){return this->val;};
    double getActivatedVal(){return this->activatedVal;};
    double getDerivedVal(){return this->derivedVal;};
};

#endif /* Neuron_hpp */
