//
//  Functions.cpp
//  CNN
//
//  Created by Dongjin Kim on 5/10/21.
//  Copyright © 2021 Dongjin Kim. All rights reserved.
//
#include "../common.hpp"
#include "Functions.hpp"
#include <numeric>

void dot(vector<vector<double>> &result, vector<vector<double>> A, vector<vector<double>> B){
    int rowA = A.size();
    int colA = A[0].size();
    int rowB = B.size();
    int colB = B[0].size();
    if(colA!=rowB){
        cout << "Dot Function error: Dimension Mismatch" << endl;
    }
    //shape (rowA, colB), fill all elements with 0
    result = vector<vector<double>>(rowA, vector<double>(colB, 0));
    for(int i=0; i<rowA; i++){
        for(int j=0; j<colB; j++){
            double sum = 0;
            for(int k=0;k<colA;k++){
                sum+=A[i][k]*B[k][j];
            }
            result[i][j] = sum;
        }
    }
}

void transpose(vector<vector<double>>&result, vector<vector<double>> A){
    int row = A.size();
    int col = A[0].size();
    vector<vector<double>> res(col, vector<double>(row,0));
    for(int i=0;i<row;i++){
        for(int j=0;j<col;j++){
            res[j][i] = A[i][j];
        }
    }
    result=res;
}

void meanAll(double &result, vector<vector<double>> A){
    int row = A.size();
    int col = A[0].size();
    double sum=0;
    for(int i=0;i<row;i++){
        for(int j=0;j<col;j++){
            sum+=A[i][j];
        }
    }
    result = sum/(row*col);
}

void stdAll(double &result, vector<vector<double>> A){
    int row = A.size();
    int col = A[0].size();
    double firstMean;
    meanAll(firstMean, A);
    vector<vector<double>> res(row, vector<double>(col, 0));
    for(int i=0;i<row;i++){
        for(int j=0;j<col;j++){
            res[i][j] = A[i][j] - firstMean;
        }
    }
    square2D(res,res);
    double secondMean;
    meanAll(secondMean, res);
    result = sqrt(secondMean);
}

void square2D(vector<vector<double>> &result, vector<vector<double>> A){
    int row = A.size();
    int col = A[0].size();
    vector<vector<double>> res(row,vector<double>(col,0));
    for(int i=0;i<row;i++){
        for(int j=0;j<col;j++){
            res[i][j] = A[i][j]*A[i][j];
        }
    }
    result = res;
}

void add2D(vector<vector<double>> &result, vector<vector<double>> A, vector<vector<double>> B){
    int rowA = A.size();
    int colA = A[0].size();
    int rowB = B.size();
    int colB = B[0].size();
    int lenA = rowA + colA;
    int lenB = rowB + colB;
    
    //broadcasting if necessary
    int tempColB = colB;
    if(colB==1){
        colB = colA;
    }
    int tempColA = colA;
    if (colA == 1)
    {
      colA = colB;
    }
    if(colA!=colB){
        cout << "Add2D Function error: Dimension Mismatch" << endl;
    }
    colA = tempColA;
    colB = tempColB;
    //Broadcasting
    vector<vector<double>> mA = A;
    vector<vector<double>> mB = B;
    int resultRow;
    int resultCol;
    if(rowB<rowA){
        for(int i=0;i<rowA-rowB;i++){
            //for the remainder add the copy of the first rows
            mB.push_back(mB[0]);
        }
    }else if(rowB>rowA){
        for (int i = 0; i < rowB - rowA; i++)
        {
          mA.push_back(mA[0]);
        }
    }
    if (colB < colA)
    {
      for (int i = 0; i < colA - colB; i++)
      {
        for (int j = 0; j < rowB; j++)
        {
          mB[j].push_back(mB[j][0]);
        }
      }
    }
    else if (colA < colB)
    {
      for (int i = 0; i < colB - colA; i++)
      {
        for (int j = 0; j < rowA; j++)
        {
          mA[j].push_back(mA[j][0]);
        }
      }
    }
    resultRow = max(rowA, rowB);
    resultCol = max(colA, colB);
    vector<vector<double>> res(resultRow, vector<double>(resultCol, 0));
    for (int i = 0; i < resultRow; i++)
    {
      for (int j = 0; j < resultCol; j++)
      {
        res[i][j] = mA[i][j] + mB[i][j];
      }
    }
    result = res;
    
}

void sub2D(vector<vector<double>> &result, vector<vector<double>> A, vector<vector<double>> B){
    int rowA = A.size();
    int colA = A[0].size();
    int rowB = B.size();
    int colB = B[0].size();
    if (rowA * colA != rowB * colB)
    {
      cout << "Sub Function error: Dimensions are not the same" << endl;
    }
    vector<vector<double>> res(rowA, vector<double>(colA, 0));
    for (int i = 0; i < rowA; i++)
    {
      for (int j = 0; j < colA; j++)
      {
        res[i][j] = A[i][j] - B[i][j];
      }
    }
    result = res;
}

void mult2D(vector<vector<double>> &result, vector<vector<double>> A, double n){
    int row = A.size();
    int col = A[0].size();
    vector<vector<double>> res(row, vector<double>(col, 0));
    for (int i = 0; i < row; i++)
    {
      for (int j = 0; j < col; j++)
      {
        res[i][j] = A[i][j] * n;
      }
    }
    result = res;
}

void multMatrices2D(vector<vector<double>> &result, vector<vector<double>> A, vector<vector<double>> B){
    int rowA = A.size();
    int colA = A[0].size();
    int rowB = B.size();
    int colB = B[0].size();
    if (rowA * colA != rowB * colB)
    {
      cout << "MultMatrices Function error: Dimensions are not the same" << endl;
    }
    vector<vector<double>> res(rowA, vector<double>(colA, 0));
    for (int i = 0; i < rowA; i++)
    {
      for (int j = 0; j < colA; j++)
      {
        res[i][j] = A[i][j] * B[i][j];
      }
    }
    result = res;
}

void sum2D(vector<vector<double>> &result, vector<vector<double>> A, int axis){
    // Axis (2 for row, 1 for col, 0 for all)
    vector<vector<double>> res;
    int row = A.size();
    int col = A[0].size();
    if(axis==1){
        //sum cols -> shape(row,1)
        res = vector<vector<double>>(row, vector<double>(1,0));
        for(int i=0;i<row;i++){
            double sum=0;
            for(int j=0;j<col;j++){
                sum+=A[i][j];
            }
            res[i][0] = sum;
        }
    }else if(axis==2){
        //sum rows -> shape(1,col)
        res = vector<vector<double>>(1, vector<double>(col, 0));
        for (int i = 0; i < col; i++)
        {
          double sum = 0;
          for (int j = 0; j < row; j++)
          {
            sum += A[j][i];
          }
          res[0][i] = sum;
        }
    } else{
        res = vector<vector<double>>(1, vector<double>(1, 0));
        double sum = 0;
        for(int i=0;i<row;i++){
            for(int j=0;j<col;j++){
                sum+=A[i][j];
            }
        }
        res[0][0]=sum;
    }
    result = res;
}

void divi2D(vector<vector<double>>&result, vector<vector<double>>A, double n){
    int row = A.size();
    int col = A[0].size();
    vector<vector<double>> res(row, vector<double>(col, 0));
    for (int i = 0; i < row; i++)
    {
      for (int j = 0; j < col; j++)
      {
        res[i][j] = A[i][j] / n;
      }
    }
    result = res;
}

void addN2D(vector<vector<double>>&result,vector<vector<double>>A, double n){
    int row = A.size();
    int col = A[0].size();
    vector<vector<double>> res(row, vector<double>(col, 0));
    for (int i = 0; i < row; i++)
    {
      for (int j = 0; j < col; j++)
      {
        res[i][j] = A[i][j] + n;
      }
    }
    result = res;
}

void divi2DMat(vector<vector<double>>&result, vector<vector<double>>A, vector<vector<double>>B){
    int row = A.size();
    int col = A[0].size();
    if (row * col != B.size() * B[0].size())
    {
      cout << "Divi2DMat Function error: Dimensions are not the same" << endl;
    }
    vector<vector<double>> res(row, vector<double>(col, 0));
    for (int i = 0; i < row; i++)
    {
      for (int j = 0; j < col; j++)
      {
        res[i][j] = A[i][j] / B[i][j];
      }
    }
    result = res;
}

void sqrt2D(vector<vector<double>>&result, vector<vector<double>>A){
    int row = A.size();
    int col = A[0].size();
    vector<vector<double>> res(row, vector<double>(col, 0));
    for (int i = 0; i < row; i++)
    {
      for (int j = 0; j < col; j++)
      {
        res[i][j] = sqrt(A[i][j]);
      }
    }
    result = res;
}


void add3D(vector<vector<vector<double>>>&result, vector<vector<vector<double>>> A, vector<vector<vector<double>>> B){
    int row = A.size();
    int col = A[0].size();
    if (row * col * A[0][0].size() != B.size() * B[0].size() * B[0][0].size())
    {
      cout << "Add3D Function error: Dimensions are not the same" << endl;
    }
    vector<vector<vector<double>>> res(row, vector<vector<double>>(col, vector<double>(A[0][0].size(),0)));
    for(int i=0;i<row;i++){
        for(int j=0;j<col;j++){
            for(int k=0;k<A[0][0].size();k++){
                res[i][j][k] = A[i][j][k] + B[i][j][k];
            }
        }
    }
    result = res;
}

void mult3D(vector<vector<vector<double>>> &result, vector<vector<vector<double>>> A, double n){
    int row = A.size();
    int col = A[0].size();
    vector<vector<vector<double>>> res(row, vector<vector<double>>(col, vector<double>(A[0][0].size(), 0)));
    for (int i = 0; i < row; i++)
    {
      for (int j = 0; j < col; j++)
      {
        for (int k = 0; k < A[0][0].size(); k++)
        {
          res[i][j][k] = A[i][j][k] * n;
        }
      }
    }
    result = res;
}

void multMatrices3D(vector<vector<vector<double>>> &result, vector<vector<vector<double>>> A, vector<vector<vector<double>>> B){
    int rowA = A.size();
    int colA = A[0].size();
    int rowB = B.size();
    int colB = B[0].size();
    if (rowA * colA * A[0][0].size() != rowB * colB * B[0][0].size())
    {
      cout << "MultMatrices3D Function error: Dimensions are not the same" << endl;
    }
    result = vector<vector<vector<double>>>(rowA, vector<vector<double>>(colA, vector<double>(A[0][0].size(), 0)));

    for (int i = 0; i < rowA; i++)
    {
      for (int j = 0; j < colA; j++)
      {
        for (int k = 0; k < A[0][0].size(); k++)
          result[i][j][k] = A[i][j][k] * B[i][j][k];
      }
    }
}

void sum3D(vector<vector<vector<double>>> &result, vector<vector<vector<double>>> A, int axis){
    // Axis (3 for row, 2 for col, 1 for channel, 0 for all)
    vector<vector<vector<double>>> res;
    int row = A.size();
    int col = A[0].size();
    if(axis==1){
        //sum channels -> shape(row,cols,1)
        res = vector<vector<vector<double>>>(row, vector<vector<double>>(col,vector<double>(1,0)));
        for(int i=0;i<row;i++){
            for(int j=0;j<col;j++){
                double sum=0;
                for(int k=0;k<A[0][0].size();k++){
                    sum+=A[i][j][k];
                }
                res[i][j][0] = sum;
            }
        }
    }else if(axis==2){
        //sum cols -> shape(row,1,channel)
        res = vector<vector<vector<double>>>(row, vector<vector<double>>(1, vector<double>(A[0][0].size(),0)));
        for (int i = 0; i < row; i++)
        {
            for (int k = 0; k <A[0][0].size(); k++)
          {
              double sum = 0;
              for(int j=0;j<col;j++){
                  sum += A[i][j][k];
              }
              res[i][0][k] = sum;
          }
        }
    } else if(axis==3){
        //sum rows -> shape(1,col,channel)
        res = vector<vector<vector<double>>>(1, vector<vector<double>>(col, vector<double>(A[0][0].size(),0)));
        for(int j=0;j<col;j++){
            for(int k=0;j<A[0][0].size();k++){
                double sum=0;
                for(int i=0;i<row;i++){
                    sum+=A[i][j][k];
                }
                res[0][j][k] = sum;
            }
        }
    } else{
        res = vector<vector<vector<double>>>(1, vector<vector<double>>(1, vector<double>(1,0)));
        double sum = 0;
        for(int i=0;i<row;i++){
            for(int j=0;j<col;j++){
                for(int k=0;k<A[0][0].size();k++){
                    sum+=A[i][j][k];
                }
            }
        }
        res[0][0][0]=sum;
    }
    result = res;
}

void add4D(vector<vector<vector<vector<double>>>> &result, vector<vector<vector<vector<double>>>>A,vector<vector<vector<vector<double>>>>B){
    int row = A.size();
    int col = A[0].size();
    if (row * col * A[0][0].size() * A[0][0][0].size() != B.size() * B[0].size() * B[0][0].size() * B[0][0][0].size())
    {
      cout << "Add4D Function error: Dimensions are not the same" << endl;
    }
    vector<vector<vector<vector<double>>>> res(row, vector<vector<vector<double>>>(col, vector<vector<double>>(A[0][0].size(), vector<double>(A[0][0][0].size(), 0))));
    for (int i = 0; i < row; i++)
    {
      for (int j = 0; j < col; j++)
      {
        for (int k = 0; k < A[0][0].size(); k++)
        {
          for (int l = 0; l < A[0][0][0].size(); l++)
          {
            res[i][j][k][l] = A[i][j][k][l] + B[i][j][k][l];
          }
        }
      }
    }
    result = res;
}

void mult4D(vector<vector<vector<vector<double>>>>&result, vector<vector<vector<vector<double>>>>A, double n){
    int row = A.size();
    int col = A[0].size();
    vector<vector<vector<vector<double>>>> res(row, vector<vector<vector<double>>>(col, vector<vector<double>>(A[0][0].size(), vector<double>(A[0][0][0].size(), 0))));
    for (int i = 0; i < row; i++)
    {
      for (int j = 0; j < col; j++)
      {
        for (int k = 0; k < A[0][0].size(); k++)
        {
          for (int l = 0; l < A[0][0][0].size(); l++)
          {
            res[i][j][k][l] = A[i][j][k][l] * n;
          }
        }
      }
    }
    result = res;
}

void divi4D(vector<vector<vector<vector<double>>>>&result, vector<vector<vector<vector<double>>>>A, double n){
    int row = A.size();
    int col = A[0].size();
    vector<vector<vector<vector<double>>>> res(row, vector<vector<vector<double>>>(col, vector<vector<double>>(A[0][0].size(), vector<double>(A[0][0][0].size(), 0))));
    for (int i = 0; i < row; i++)
    {
      for (int j = 0; j < col; j++)
      {
        for (int k = 0; k < A[0][0].size(); k++)
        {
          for (int l = 0; l < A[0][0][0].size(); l++)
          {
            res[i][j][k][l] = A[i][j][k][l] / n;
          }
        }
      }
    }
    result = res;
}

void square4D(vector<vector<vector<vector<double>>>>&result, vector<vector<vector<vector<double>>>> A){
    int row = A.size();
    int col = A[0].size();
    vector<vector<vector<vector<double>>>> res(row, vector<vector<vector<double>>>(col, vector<vector<double>>(A[0][0].size(), vector<double>(A[0][0][0].size(), 0))));
    for (int i = 0; i < row; i++)
    {
      for (int j = 0; j < col; j++)
      {
        for (int k = 0; k < A[0][0].size(); k++)
        {
          for (int l = 0; l < A[0][0][0].size(); l++)
          {
            res[i][j][k][l] = A[i][j][k][l] * A[i][j][k][l];
          }
        }
      }
    }
    result = res;
}

void sqrt4D(vector<vector<vector<vector<double>>>>&result,vector<vector<vector<vector<double>>>>A){
    int row = A.size();
    int col = A[0].size();
    vector<vector<vector<vector<double>>>> res(row, vector<vector<vector<double>>>(col, vector<vector<double>>(A[0][0].size(), vector<double>(A[0][0][0].size(), 0))));
    for (int i = 0; i < row; i++)
    {
      for (int j = 0; j < col; j++)
      {
        for (int k = 0; k < A[0][0].size(); k++)
        {
          for (int l = 0; l < A[0][0][0].size(); l++)
          {
            res[i][j][k][l] = sqrt(A[i][j][k][l]);
          }
        }
      }
    }
    result = res;
}

void addN4D(vector<vector<vector<vector<double>>>>&result, vector<vector<vector<vector<double>>>>A, double n){
    int row = A.size();
    int col = A[0].size();
    vector<vector<vector<vector<double>>>> res(row, vector<vector<vector<double>>>(col, vector<vector<double>>(A[0][0].size(), vector<double>(A[0][0][0].size(), 0))));
    for (int i = 0; i < row; i++)
    {
      for (int j = 0; j < col; j++)
      {
        for (int k = 0; k < A[0][0].size(); k++)
        {
          for (int l = 0; l < A[0][0][0].size(); l++)
          {
            res[i][j][k][l] = A[i][j][k][l] + n;
          }
        }
      }
    }
    result = res;
}
void divi4DMat(vector<vector<vector<vector<double>>>>&result, vector<vector<vector<vector<double>>>>A, vector<vector<vector<vector<double>>>>B){
    int row = A.size();
    int col = A[0].size();
    if (row * col * A[0][0].size() * A[0][0][0].size() != B.size() * B[0].size() * B[0][0].size() * B[0][0][0].size())
    {
      cout << "Divi4DMat Function error: Dimensions are not the same" << endl;
    }
    vector<vector<vector<vector<double>>>> res(row, vector<vector<vector<double>>>(col, vector<vector<double>>(A[0][0].size(), vector<double>(A[0][0][0].size(), 0))));
    for (int i = 0; i < row; i++)
    {
      for (int j = 0; j < col; j++)
      {
        for (int k = 0; k < A[0][0].size(); k++)
        {
          for (int l = 0; l < A[0][0][0].size(); l++)
          {
            res[i][j][k][l] = A[i][j][k][l] / B[i][j][k][l];
          }
        }
      }
    }
    result = res;
}
void sub4DMat(vector<vector<vector<vector<double>>>>&result, vector<vector<vector<vector<double>>>>A, vector<vector<vector<vector<double>>>>B){
    int row = A.size();
    int col = A[0].size();
    if (row * col * A[0][0].size() * A[0][0][0].size() != B.size() * B[0].size() * B[0][0].size() * B[0][0][0].size())
    {
      cout << "Sub4DMat Function error: Dimensions are not the same" << endl;
    }
    vector<vector<vector<vector<double>>>> res(row, vector<vector<vector<double>>>(col, vector<vector<double>>(A[0][0].size(), vector<double>(A[0][0][0].size(), 0))));
    for (int i = 0; i < row; i++)
    {
      for (int j = 0; j < col; j++)
      {
        for (int k = 0; k < A[0][0].size(); k++)
        {
          for (int l = 0; l < A[0][0][0].size(); l++)
          {
            res[i][j][k][l] = A[i][j][k][l] - B[i][j][k][l];
          }
        }
      }
    }
    result = res;
}
