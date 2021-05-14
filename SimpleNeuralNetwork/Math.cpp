//
//  Math.cpp
//  NeuralNetwork
//
//  Created by Dongjin Kim on 5/9/21.
//  Copyright © 2021 Dongjin Kim. All rights reserved.
//

#include "Math.hpp"
void utils::Math::multiplyMatrix(Matrix *a, Matrix *b, Matrix *c){
    for(int i=0;i<a->getNumRows();i++){
        for(int j=0;j<b->getNumCols();j++){
            double p = 0;
            for(int k=0;k<b->getNumRows();k++){
                p += a->getValue(i, k)*b->getValue(k, j); //a*b matrix multiplication
            }
            double newVal = c->getValue(i, j) + p;
            c->setValue(i, j, newVal);
        }
    }
}
