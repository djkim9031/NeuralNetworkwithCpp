//
//  Matrix.hpp
//  NeuralNetwork
//
//  Created by Dongjin Kim on 5/8/21.
//  Copyright © 2021 Dongjin Kim. All rights reserved.
//

#ifndef Matrix_hpp
#define Matrix_hpp

#include <stdio.h>
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

using namespace std;
class Matrix {

    int numRows;
    int numCols;
    vector<vector<double>> values;
    double generateRandomNumber();
    
public:
    Matrix(int numRows, int numCols, bool isRandom);
    Matrix *transpose();
    Matrix *copy();
    
    //Debugging
    void printToConsole();
    
    //Setter
    void setValue(int r, int c, double v){
        this->values.at(r).at(c) = v;
    };
    
    //Getters
    int getNumRows(){return this->numRows;};
    int getNumCols(){return this->numCols;};
    double getValue(int r, int c){return this->values.at(r).at(c);};
    
    
};

#endif /* Matrix_hpp */
