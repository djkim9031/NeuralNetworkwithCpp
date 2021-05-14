//
//  Layer.cpp
//  NeuralNetwork
//
//  Created by Dongjin Kim on 5/8/21.
//  Copyright © 2021 Dongjin Kim. All rights reserved.
//

#include "Layer.hpp"

//Constructors
Layer::Layer(int size){
    this->size = size;
    for(int i=0;i<size;i++){
        Neuron *n = new Neuron(0.0010000);
        this->neurons.push_back(n);
    }
}
Layer::Layer(int size, int activatationType){
    this->size = size;
    for(int i=0;i<size;i++){
        Neuron *n = new Neuron(0.0010000,activatationType);
        this->neurons.push_back(n);
    }
}

//Setter
void Layer::setVal(int i, double v){
    this->neurons.at(i)->setVal(v);
}

//Getter
vector<double> Layer::getActivatedVals(){
    vector<double> result;
    for(int i=0;i<this->neurons.size();i++){
        double v = this->neurons.at(i)->getActivatedVal();
        result.push_back(v);
    }
    return result;
}

//Matrices
Matrix *Layer::matrixifyVals(){
    Matrix *m = new Matrix(1, this->neurons.size(), false);
    
    for(int i=0;i<this->neurons.size();i++){
        m->setValue(0, i, this->neurons.at(i)->getVal());
    }
    return m;
}
Matrix *Layer::matrixifyActivatedVals(){
    Matrix *m = new Matrix(1, this->neurons.size(), false);
    
    for(int i=0;i<this->neurons.size();i++){
        m->setValue(0, i, this->neurons.at(i)->getActivatedVal());
    }
    return m;
}
Matrix *Layer::matrixifyDerivedVals(){
    Matrix *m = new Matrix(1, this->neurons.size(), false);
    
    for(int i=0;i<this->neurons.size();i++){
        m->setValue(0, i, this->neurons.at(i)->getDerivedVal());
    }
    return m;
}

