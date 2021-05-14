//
//  NeuralNetwork.cpp
//  NeuralNetwork
//
//  Created by Dongjin Kim on 5/8/21.
//  Copyright © 2021 Dongjin Kim. All rights reserved.
//

#include <cassert>
#include "NeuralNetwork.hpp"
#include "Math.hpp"

NeuralNetwork::NeuralNetwork(vector<int> topology, double bias, double learningRate, double momentum){
    this->topology = topology;
    this->topologySize = topology.size();
    this->bias = bias;
    this->learningRate = learningRate;
    this->momentum = momentum;
    
    for(int i=0;i<topologySize;i++){
        if(i>0 && i<(topologySize-1)){ //hidden layers
            this->layers.push_back(new Layer(topology.at(i),this->hiddenActivationType));
        } else if(i==(topologySize-1)){ //output layer
            this->layers.push_back(new Layer(topology.at(i),this->outputActivationType));
        } else { //input layer
            this->layers.push_back(new Layer(topology.at(i)));
        }
    }
    
    for(int i=0;i<(topologySize-1);i++){
        Matrix *weightMatrix = new Matrix(topology.at(i),topology.at(i+1),true);
        this->weightMatrices.push_back(weightMatrix);
    }
    
    for(int i=0; i<topology.at(topologySize-1);i++){
        errors.push_back(0.00);
        derivedErrors.push_back(0.00);
    }
    
    this->error = 0.00;
}

NeuralNetwork::NeuralNetwork(vector<int> topology,int hiddenActivationType, int outputActivationType, int costFunctionType, double bias, double learningRate, double momentum){
    this->topology      = topology;
    this->topologySize  = topology.size();
    this->learningRate  = learningRate;
    this->momentum      = momentum;
    this->bias          = bias;
    this->hiddenActivationType      = hiddenActivationType;
    this->outputActivationType      = outputActivationType;
    this->costFunctionType          = costFunctionType;
    
    for (int i = 0; i < topologySize; i++) {
      if (i > 0 && i < (topologySize - 1)) {
        Layer *l  = new Layer(topology.at(i), this->hiddenActivationType);
        this->layers.push_back(l);
      } else if (i == (topologySize - 1)) {
        Layer *l  = new Layer(topology.at(i), this->outputActivationType);
        this->layers.push_back(l);
      } else {
        Layer *l  = new Layer(topology.at(i));
        this->layers.push_back(l);
      }
    }

      for (int i = 0; i < (topologySize - 1); i++) {
          Matrix *weightMatrix = new Matrix(topology.at(i), topology.at(i + 1), true);
          this->weightMatrices.push_back(weightMatrix);
      }

      for (int i = 0; i < topology.at(topologySize - 1); i++) {
          errors.push_back(0.00);
          derivedErrors.push_back(0.00);
      }

      this->error = 0.00;
}

void NeuralNetwork::setCurrentInput(vector<double> input){
    this->input = input;
    for(int i=0;i<input.size();i++){
        this->layers.at(0)->setVal(i, input.at(i));
    }
}

void NeuralNetwork::feedForward(){
    Matrix *a; //Matrix of previous layer's neurons (activated or the very input)
    Matrix *b; //Matrix of current layer's weights
    Matrix *c; //Matrix of current layer's (output) neurons (multiplied by a and b) prior to activation
    
    // i=|x ----- w->|, i+1=|z(i+1)=wx+b->A(wx+b) -------- w2 ->|, ...
    //Neuron matrix (layer0 = input, layer1=hidden1, ...)
    //Weight matrix (layer1's weights, layer2's weights,...)
    //input x = z1 = a1
    //z2 = a1w1 + b1, z3 = a2w2+b2, ...
    for(int i=0;i<(this->topologySize-1);i++){
        
        if(i!=0){//hidden layers, getting the "activated" previous layer's neurons
            a = this->getActivatedNeuronMatrix(i);
        }else{ //input layer
            a = this->getNeuronMatrix(i);
        }
        b = this->getWeightMatrix(i);
        c = new Matrix(a->getNumRows(),b->getNumCols(),false);
        
        utils::Math::multiplyMatrix(a, b, c);
        
        for(int c_index=0;c_index<c->getNumCols();c_index++){
            //setting the neuron value, activate (A), and differentiate A with respect to the z(wx+b)
            this->setNeuronValue(i+1, c_index, c->getValue(0, c_index)+this->bias);
            
        }
        delete a;
        delete b;
        delete c;
    }
}

void NeuralNetwork::backPropagation(){
    vector<Matrix *> newWeights;
    Matrix *deltaWeights;
    Matrix *gradients;
    Matrix *derivedValues;
    Matrix *gradientsTransposed;
    Matrix *zActivatedVals;
    Matrix *tempNewWeights;
    Matrix *pGradients;
    Matrix *transposedPWeights;
    Matrix *hiddenDerived;
    Matrix *transposedHidden;
    
    //Outputs of the final layer
    int indexOutputLayer = this->topology.size() - 1;
    gradients = new Matrix(1,this->topology.at(indexOutputLayer),false);
    derivedValues = this->layers.at(indexOutputLayer)->matrixifyDerivedVals();
    
    for(int i=0;i<this->topology.at(indexOutputLayer);i++){
        double e = this->derivedErrors.at(i); //dL/dA
        double y = derivedValues->getValue(0, i); //dA/dZ
        double g = e*y; //dL/dZ
        gradients->setValue(0, i, g);
    }
    
    //Gradient.T * A_prev, and by chain rule, dL/dZ * A_prev = dL/dZ * dZ/dw_prev = dL/dw_prev, shape=(topology.at(indexOutputLayer), topology.at(indexOutputLayer-1))
    gradientsTransposed = gradients->transpose();
    zActivatedVals = this->layers.at(indexOutputLayer-1)->matrixifyActivatedVals();
    deltaWeights = new Matrix(gradientsTransposed->getNumRows(),zActivatedVals->getNumCols(),false);
    utils::Math::multiplyMatrix(gradientsTransposed, zActivatedVals, deltaWeights);
    
    
    //Compute for new weights (between last hidden layer and output layer)
    tempNewWeights = new Matrix(this->topology.at(indexOutputLayer-1),this->topology.at(indexOutputLayer),false);
    
    for(int r=0;r<this->topology.at(indexOutputLayer-1);r++){
        for(int c=0;c<this->topology.at(indexOutputLayer);c++){
            double originalVal = this->weightMatrices.at(indexOutputLayer-1)->getValue(r, c);
            double deltaVal = deltaWeights->getValue(c, r);
            
            originalVal = this->momentum*originalVal;
            deltaVal = this->learningRate*deltaVal;
            tempNewWeights->setValue(r, c, (originalVal-deltaVal));
        }
    }
    newWeights.push_back(new Matrix(*tempNewWeights));
    
    delete gradientsTransposed;
    delete zActivatedVals;
    delete tempNewWeights;
    delete deltaWeights;
    delete derivedValues;
    
    /*Last hidden layer to the input layer updates*/
    for(int i=(indexOutputLayer-1);i>0;i--){
        pGradients = new Matrix(*gradients); //dL/dZ, shape = (1,topology.at(i+1))
        delete gradients;
        
        transposedPWeights = this->weightMatrices.at(i)->transpose(); //transposed matrix of dZ/dA_prev, transposed shape = (topo.at(i+1),topo.at(i))
        gradients = new Matrix(pGradients->getNumRows(),transposedPWeights->getNumCols(),false);
        // dL/dZ * dZ/dA_prev => dL/dA_prev, shape = (1,topo.at(i))
        utils::Math::multiplyMatrix(pGradients, transposedPWeights, gradients);
        //dA_prev/dZ_prev
        hiddenDerived = this->layers.at(i)->matrixifyDerivedVals();
        
        //for loop -> dL/dA_prev * dA_prev/dZ_prev = dL/dZ_prev, shape = (1,topo.at(i))
        for(int colCounter=0;colCounter<hiddenDerived->getNumRows();colCounter++){
            double g = gradients->getValue(0, colCounter)*hiddenDerived->getValue(0, colCounter);
            gradients->setValue(0, colCounter, g);
        }
        if(i==1){
            zActivatedVals  = this->layers.at(0)->matrixifyVals();
        }else{
            zActivatedVals  = this->layers.at(i-1)->matrixifyActivatedVals();
        }
        transposedHidden = zActivatedVals->transpose(); //shape = (topo.at(i-1),1)
        deltaWeights = new Matrix(transposedHidden->getNumRows(),gradients->getNumCols(),false);
        //shape = (topo.at(i-1),topo.at(i))
        utils::Math::multiplyMatrix(transposedHidden, gradients, deltaWeights);
        tempNewWeights = new Matrix(this->topology.at(i-1),this->topology.at(i),false);
        for(int r=0;r<this->topology.at(i-1);r++){
            for(int c=0;c<this->topology.at(i);c++){
                double originalVal = this->weightMatrices.at(i-1)->getValue(r,c);
                double deltaVal = deltaWeights->getValue(r, c);
                
                originalVal = this->momentum*originalVal;
                deltaVal = this->learningRate*deltaVal;
                tempNewWeights->setValue(r, c, (originalVal-deltaVal));
            }
        }
        newWeights.push_back(new Matrix(*tempNewWeights));
        
        delete pGradients;
        delete transposedPWeights;
        delete hiddenDerived;
        delete zActivatedVals;
        delete transposedHidden;
        delete deltaWeights;
        delete tempNewWeights;
    }
    delete gradients;
    
    //delete the original weights
    for(int i = 0; i < this->weightMatrices.size(); i++) {
      delete this->weightMatrices[i];
    }
    this->weightMatrices.clear();
    
    //replace the weightMatrices with the updated weights
    reverse(newWeights.begin(),newWeights.end());
    for(int i=0;i<newWeights.size();i++){
        this->weightMatrices.push_back(new Matrix(*newWeights[i]));
        delete newWeights[i];
    }
}

void NeuralNetwork::setErrors(){
    switch(costFunctionType){
        case(COST_MSE): this->setErrorMSE(); break;
        default: this->setErrorMSE(); break;
    }
}

void NeuralNetwork::setErrorMSE(){
    int outputLayerIndex = this->layers.size()-1;
    vector<Neuron *> outputNeurons = this->layers.at(outputLayerIndex)->getNuerons();
    this->error = 0.00;
    for(int i=0;i<target.size();i++){
        double t = target.at(i);
        double y = outputNeurons.at(i)->getActivatedVal();
        outputs.at(i) = y;
        errors.at(i) = 0.5*pow(abs(t-y),2);
        derivedErrors.at(i) = (y-t); //dL/dA
        this->error += errors.at(i);
    }
}

void NeuralNetwork::train(vector<double> input, vector<double> target, double bias, double learningRate, double momentum, int epochs){
    this->learningRate = learningRate;
    this->momentum = momentum;
    this->bias = bias;
    this->setCurrentInput(input);
    this->setCurrentTarget(target);
    this->epochs = epochs;
    for(int i=0;i<target.size();i++){
        outputs.push_back(0.00);
    }
    
    for(int i=0;i<epochs;i++){
        cout<<"Epoch "<<i+1<<endl;
        this->feedForward();
        this->setErrors();
        this->backPropagation();
        cout<<"Outputs: ";
        for(int j=0;j<target.size();j++){
            cout<<this->outputs.at(j)<<" ";
        }
        cout << "Error: " << this->error << endl;
        cout<<"______________________"<<endl;
    }
    
}
