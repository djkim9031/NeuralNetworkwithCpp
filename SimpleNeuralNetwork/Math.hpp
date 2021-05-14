//
//  Math.hpp
//  NeuralNetwork
//
//  Created by Dongjin Kim on 5/9/21.
//  Copyright © 2021 Dongjin Kim. All rights reserved.
//

#ifndef Math_hpp
#define Math_hpp

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cassert>
#include <cmath>
#include "Matrix.hpp"

using namespace std;

namespace utils {
class Math{
public:
    static void multiplyMatrix(Matrix *a, Matrix *b, Matrix *c);
};
}

#endif /* Math_hpp */
