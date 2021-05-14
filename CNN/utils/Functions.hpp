//
//  Functions.hpp
//  CNN
//
//  Created by Dongjin Kim on 5/10/21.
//  Copyright © 2021 Dongjin Kim. All rights reserved.
//

#ifndef Functions_hpp
#define Functions_hpp

#include <vector>
using namespace std;

//2D matrices
void dot(vector<vector<double>>&result, vector<vector<double>> A, vector<vector<double>> B);
void transpose(vector<vector<double>>&result, vector<vector<double>> A);
void meanAll(double &result, vector<vector<double>> A);
void stdAll(double &result, vector<vector<double>> A);
void square2D(vector<vector<double>> &result, vector<vector<double>> A);
void add2D(vector<vector<double>> &result, vector<vector<double>> A, vector<vector<double>> B);
void sub2D(vector<vector<double>> &result, vector<vector<double>> A, vector<vector<double>> B);
void mult2D(vector<vector<double>> &result, vector<vector<double>> A, double n);
void multMatrices2D(vector<vector<double>> &result, vector<vector<double>> A, vector<vector<double>> B);
void sum2D(vector<vector<double>> &result, vector<vector<double>> A, int axis);
void divi2D(vector<vector<double>>&result, vector<vector<double>>A, double n);
void addN2D(vector<vector<double>>&result,vector<vector<double>>A, double n);
void divi2DMat(vector<vector<double>>&result, vector<vector<double>>A, vector<vector<double>>B);
//void sub2DMat(vector<vector<double>>&result, vector<vector<double>>A, vector<vector<double>>B);
void sqrt2D(vector<vector<double>>&result, vector<vector<double>>A);

//3D matrices
void add3D(vector<vector<vector<double>>>&result, vector<vector<vector<double>>> A, vector<vector<vector<double>>> B);
void mult3D(vector<vector<vector<double>>> &result, vector<vector<vector<double>>> A, double n);
void multMatrices3D(vector<vector<vector<double>>> &result, vector<vector<vector<double>>> A, vector<vector<vector<double>>> B);
void sum3D(vector<vector<vector<double>>> &result, vector<vector<vector<double>>> A, int axis);

//4D operations
void add4D(vector<vector<vector<vector<double>>>> &result, vector<vector<vector<vector<double>>>>A,vector<vector<vector<vector<double>>>>B);
void mult4D(vector<vector<vector<vector<double>>>>&result, vector<vector<vector<vector<double>>>>A, double n);
void divi4D(vector<vector<vector<vector<double>>>>&result, vector<vector<vector<vector<double>>>>A, double n);
void square4D(vector<vector<vector<vector<double>>>>&result, vector<vector<vector<vector<double>>>> A);
void sqrt4D(vector<vector<vector<vector<double>>>>&result,vector<vector<vector<vector<double>>>>A);
void addN4D(vector<vector<vector<vector<double>>>>&result, vector<vector<vector<vector<double>>>>A, double n);
void divi4DMat(vector<vector<vector<vector<double>>>>&result, vector<vector<vector<vector<double>>>>A, vector<vector<vector<vector<double>>>>B);
void sub4DMat(vector<vector<vector<vector<double>>>>&result, vector<vector<vector<vector<double>>>>A, vector<vector<vector<vector<double>>>>B);


#endif /* Functions_hpp */
