//
//  Matrix.cpp
//  NeuralNetwork
//
//  Created by Dongjin Kim on 5/8/21.
//  Copyright © 2021 Dongjin Kim. All rights reserved.
//

#include "Matrix.hpp"

//Constructor
Matrix::Matrix(int numRows, int numCols, bool isRandom){
    this->numRows = numRows;
    this->numCols = numCols;
    for(int i=0;i<numRows;i++){
        vector<double> colValues;
        for(int j=0;j<numCols;j++){
            double v = isRandom==true?this->generateRandomNumber():0.00;
            colValues.push_back(v);
        }
        this->values.push_back(colValues);
    }
}

//Transpose
Matrix *Matrix::transpose(){
    Matrix *m = new Matrix(this->numCols,this->numRows,false);
    
    for(int i=0;i<this->numRows;i++){
        for(int j=0;j<this->numCols;j++){
            m->setValue(j, i, this->getValue(i, j));
        }
    }
    return m;
}

//Copy
Matrix *Matrix::copy(){
    Matrix *m = new Matrix(this->numRows,this->numCols,false);
    
    for(int i=0;i<this->numRows;i++){
        for(int j=0;j<this->numCols;j++){
            m->setValue(i, j, this->getValue(i, j));
        }
    }
    return m;
}

//For debugging
void Matrix::printToConsole(){
    for(int i=0;i<this->numRows;i++){
        for(int j=0;j<this->numCols;j++){
            cout<<this->values.at(i).at(j)<<"\t\t";
        }
        cout<<endl;
    }
}

//Random number generation
double Matrix::generateRandomNumber(){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-.0001, .0001);
    return dis(gen);
}
