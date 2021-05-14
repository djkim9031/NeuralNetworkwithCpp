//
//  Neuron.cpp
//  NeuralNetwork
//
//  Created by Dongjin Kim on 5/8/21.
//  Copyright © 2021 Dongjin Kim. All rights reserved.
//

#include "Neuron.hpp"


//Constructors
Neuron::Neuron(double val){
    this->setVal(val);
}
Neuron::Neuron(double val, int activationType){
    this->setVal(val);
    this->activationType = activationType;
}

void Neuron::setVal(double v){
    this->val = v;
    activate();
    derive();
}

void Neuron::activate(){
    if(this->activationType == TANH){
        this->activatedVal = tanh(this->val);
    } else if(this->activationType == RELU){
        if(this->val>0){
            this->activatedVal = this->val;
        } else{
            this->activatedVal = 0;
        }
    } else if(this->activationType == SIGM){
        this->activatedVal = (1/(1+exp(-this->val)));
    } else {
        //No activation
    }
}

void Neuron::derive(){ //differentiated value of the activatedVal with respect to z (=wx+b)
    if(this->activationType == TANH){
        this->derivedVal = (1.0 - (this->activatedVal * this->activatedVal));
    } else if(this->activationType == RELU){
        if(this->val>0){
            this->derivedVal = 1;
        } else{
            this->derivedVal = 0;
        }
    } else if(this->activationType == SIGM){ //A' = A(1-A)
        this->derivedVal = (this->activatedVal * (1 - this->activatedVal));
    } else {
        this->derivedVal=1;
    }
}
