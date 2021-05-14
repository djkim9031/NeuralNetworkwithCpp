//
//  CNN.cpp
//  CNN
//
//  Created by Dongjin Kim on 5/11/21.
//  Copyright © 2021 Dongjin Kim. All rights reserved.
//

#include "CNN.hpp"
#include "./utils/Functions.hpp"
#include "./utils/CSVReader.hpp"
#include "Forward.hpp"
#include "Backward.hpp"
#include "common.hpp"

//Constructor
CNN::CNN(){
    params.push_back({numFilter1,1,5,5}); //f1
    params.push_back({numFilter2,numFilter1,5,5}); //f2
    params.push_back({128,numFilter2*100}); //w3
    params.push_back({10,128}); //w4
    
    //for each layer above, corresponding bias is added
    params.push_back({numFilter1, 1});                // b1
    params.push_back({numFilter2, 1});                // b2
    params.push_back({128, 1});                       // b3
    params.push_back({10, 1});                        // b4
    
    //Assigning each values with the dimension from params
    f1 = vector<vector<vector<vector<double>>>>(params[0][0], vector<vector<vector<double>>>(params[0][1], vector<vector<double>>(params[0][2], vector<double>(params[0][3], 0))));
    f2 = vector<vector<vector<vector<double>>>>(params[1][0], vector<vector<vector<double>>>(params[1][1], vector<vector<double>>(params[1][2], vector<double>(params[1][3], 0))));
    w3 = vector<vector<double>>(params[2][0], vector<double>(params[2][1], 0));
    w4 = vector<vector<double>>(params[3][0], vector<double>(params[3][1], 0));
    b1 = vector<vector<double>>(params[4][0], vector<double>(params[4][1], 0));
    b2 = vector<vector<double>>(params[5][0], vector<double>(params[5][1], 0));
    b3 = vector<vector<double>>(params[6][0], vector<double>(params[6][1], 0));
    b4 = vector<vector<double>>(params[7][0], vector<double>(params[7][1], 0));
    initializeParams();
}

void CNN::initializeParams(){
    srand(time(NULL));
    double dev1 = 1 / sqrt(f1.size() * f1[0].size() * f1[0][0].size() * f1[0][0][0].size());
    double dev2 = 1 / sqrt(f2.size() * f2[0].size() * f2[0][0].size() * f2[0][0][0].size());
    
    //f1 and f2 filter random initialization
    for(int i=0;i<f1.size();i++){
        for(int j=0;j<f1[0].size();j++){
            for(int k=0;k<f1[0][0].size();k++){
                for(int l=0;l<f1[0][0][0].size();l++){
                    double f = (double)rand() / RAND_MAX;
                    double r = f * 4 - 2;
                    f1[i][j][k][l] = dev1 * exp(-1 * ((r * r) / 2));
                    if (((double)rand() / (RAND_MAX)) > 0.5)
                    {
                      f1[i][j][k][l] *= -1;
                    }
                    else
                    {
                      f1[i][j][k][l] *= 1;
                    }
                }
            }
        }
    }
    for (int i = 0; i < f2.size(); i++)
    {
      for (int j = 0; j < f2[0].size(); j++)
      {
        for (int k = 0; k < f2[0][0].size(); k++)
        {
          for (int l = 0; l < f2[0][0][0].size(); l++)
          {
            double f = (double)rand() / RAND_MAX;
            double r = f * 6 - 3;
            f2[i][j][k][l] = dev2 * exp(-1 * ((r * r) / 2));
            if (((double)rand() / (RAND_MAX)) > 0.5)
            {
              f2[i][j][k][l] *= -1;
            }
            else
            {
              f2[i][j][k][l] *= 1;
            }
          }
        }
      }
    }
    
    //w3 and w4 initializations
    for (int i = 0; i < w3.size(); i++)
    {
      for (int j = 0; j < w3[0].size(); j++)
      {
        w3[i][j] = randGaussian() * 0.01;
      }
    }
    for (int i = 0; i < w4.size(); i++)
    {
      for (int j = 0; j < w4[0].size(); j++)
      {
        w4[i][j] = randGaussian() * 0.01;
      }
    }
}

double CNN::randGaussian(){
    double r = ((double)rand() / (RAND_MAX));
    return min(1.0, (((double)rand() / (RAND_MAX)) > 0.5 ? sqrt(-2 * log(r + .05)) : -1 * sqrt(-2 * log(r + .05))));
}

void CNN::conv(double &_loss, vector<vector<vector<vector<double> > > > &_df1, vector<vector<vector<vector<double> > > > &_df2, vector<vector<double> > &_dw3, vector<vector<double> > &_dw4, vector<vector<double> > &_db1, vector<vector<double> > &_db2, vector<vector<double> > &_db3, vector<vector<double> > &_db4, vector<vector<vector<double> > > image, vector<vector<double> > label){
    
    //----------------------------------------------------//
    /*Forward path*/
    //conv1 layer, input shape = (1,28,28)
    vector<vector<vector<double>>> conv1;
    convolution(conv1, image, f1, b1); //f1 = (8,1,5,5)
    ReLU(conv1);
    
    //conv2 layer, input shape = (8,24,24)
    vector<vector<vector<double>>> conv2;
    convolution(conv2, conv1, f2, b2); //f2 = (8,8,5,5)
    ReLU(conv2);
    
    //Maxpool layer, input shape = (8,20,20)
    vector<vector<vector<double>>>pooled;
    maxpool(pooled, conv2);
    
    //Flatten, input shape = (8,10,10)
    vector<vector<double>>flat;
    for(auto &row:pooled){
        for(auto &col:row){
            for(auto &elem:col){
                flat.push_back({elem});
            }
        }
    }
    
    //FC Dense layer
    vector<vector<double>>z;
    dot(z,w3,flat); // w3 = (128, 8*100), flat = (800,1) after dot
    add2D(z, z, b3); //z = (flat)*w3+b3
    ReLU2D(z);
    
    //FC Dense output layer, input shape = (128,1)
    vector<vector<double>>out;
    dot(out, w4, z); //w4 = (10,128), z=(128,1)
    add2D(out, out, b4); // out = zw4 + b4
    vector<vector<double>>probs; //(10, 1)
    softmax(probs, out);
    
    
    
    //----------------------------------------------------//
    /*Backpropagation*/
    //loss calculation
    categoricalCrossEntropy(_loss, probs, label);
    
    // dL/d(out) ((out)=output of the previous layer prior to the softmax activation)
    // = probs - label
    vector<vector<double>> dout;
    sub2D(dout, probs, label); //(10,1)
    
    //dL/d(out) * z (=input) = dL/d(out) * d(out)/d(w4) since, out = zw4 + b4
    //Hence, the following is for dL/d(w4) = _dw4
    vector<vector<double>>zT;
    transpose(zT, z); //(1,128)
    dot(_dw4, dout, zT); //(10,128)
    //dL/d(out) = dL/db4
    _db4 = dout;
    
    //dL/dw3. First, to apply chain rule, we need dL/dz = dL/d(out)*d(out)/d(z) = dout * w4
    //dL/dz = w4.T * dout (128,10)* (10,1) = (128,1)
    vector<vector<double>>w4T;
    transpose(w4T, w4);
    vector<vector<double>>dz;
    dot(dz,w4T,dout);
    //Now, dL/dw3 = dL/dz*dz/(dw3) = dL/dz*(flat)
    vector<vector<double>> flatT;
    transpose(flatT, flat);
    dot(_dw3,dz,flatT);//(128,1)*(1,800) = (128,800)
    //dL/dz = dL/db3
    _db3 = dz;
    
    //For further backward propagation, dL/d(flat) is needed;
    //dFlat = dL/d(flat) = dL/dz * dz/d(flat) = dz * w3
    //dFlat = w3.T * dz (800,128)*(128,1) =  (800,1)
    vector<vector<double>> w3T;
    transpose(w3T, w3);
    vector<vector<double>> dFlat;
    dot(dFlat, w3T, dz); //(800,1)
    
    //Now, let's get back to the maxpool output which has a shape (8,10,10)
    vector<vector<vector<double>>>dpool;
    int index=0;
    for(int i=0;i<numFilter2;i++){
        vector<vector<double>>temp_i;
        for(int j=0;j<10;j++){
            vector<double>temp_j;
            for(int k=0;k<10;k++){
                temp_j.push_back(dFlat[index][0]);
                index++;
            }
            temp_i.push_back({temp_j});
        }
        dpool.push_back({temp_i});
    }
    //dFlat is rearraged to a 3D vector for further propagation
    vector<vector<vector<double>>>dconv2;
    maxpoolBackward(dconv2, dpool, conv2);
    
    //conv2 layer back propagation
    vector<vector<vector<double>>>dconv1;
    convolutionBackward(dconv1, _df2, _db2, dconv2, conv1, f2);
    
    //conv1 layer back propagation
    vector<vector<vector<double>>>_; //don't care
    convolutionBackward(_, _df1, _db1, dconv1, image, f1);
    
}

void CNN::adamGD(int imageAmount, int startRowNum, vector<double> &cost){
    double _cost = 0;
    
    //Initialize gradients and momentum, RMS params
    vector<vector<vector<vector<double>>>> df1(params[0][0], vector<vector<vector<double>>>(params[0][1], vector<vector<double>>(params[0][2], vector<double>(params[0][3], 0))));
    vector<vector<vector<vector<double>>>> df2(params[1][0], vector<vector<vector<double>>>(params[1][1], vector<vector<double>>(params[1][2], vector<double>(params[1][3], 0))));
    vector<vector<double>> dw3(params[2][0], vector<double>(params[2][1], 0));
    vector<vector<double>> dw4(params[3][0], vector<double>(params[3][1], 0));
    vector<vector<double>> db1(params[4][0], vector<double>(params[4][1], 0));
    vector<vector<double>> db2(params[5][0], vector<double>(params[5][1], 0));
    vector<vector<double>> db3(params[6][0], vector<double>(params[6][1], 0));
    vector<vector<double>> db4(params[7][0], vector<double>(params[7][1], 0));
    
    vector<vector<vector<vector<double>>>> v1(params[0][0], vector<vector<vector<double>>>(params[0][1], vector<vector<double>>(params[0][2], vector<double>(params[0][3], 0))));
    vector<vector<vector<vector<double>>>> v2(params[1][0], vector<vector<vector<double>>>(params[1][1], vector<vector<double>>(params[1][2], vector<double>(params[1][3], 0))));
    vector<vector<double>> v3(params[2][0], vector<double>(params[2][1], 0));
    vector<vector<double>> v4(params[3][0], vector<double>(params[3][1], 0));
    vector<vector<double>> bv1(params[4][0], vector<double>(params[4][1], 0));
    vector<vector<double>> bv2(params[5][0], vector<double>(params[5][1], 0));
    vector<vector<double>> bv3(params[6][0], vector<double>(params[6][1], 0));
    vector<vector<double>> bv4(params[7][0], vector<double>(params[7][1], 0));

    vector<vector<vector<vector<double>>>> s1(params[0][0], vector<vector<vector<double>>>(params[0][1], vector<vector<double>>(params[0][2], vector<double>(params[0][3], 0))));
    vector<vector<vector<vector<double>>>> s2(params[1][0], vector<vector<vector<double>>>(params[1][1], vector<vector<double>>(params[1][2], vector<double>(params[1][3], 0))));
    vector<vector<double>> s3(params[2][0], vector<double>(params[2][1], 0));
    vector<vector<double>> s4(params[3][0], vector<double>(params[3][1], 0));
    vector<vector<double>> bs1(params[4][0], vector<double>(params[4][1], 0));
    vector<vector<double>> bs2(params[5][0], vector<double>(params[5][1], 0));
    vector<vector<double>> bs3(params[6][0], vector<double>(params[6][1], 0));
    vector<vector<double>> bs4(params[7][0], vector<double>(params[7][1], 0));
    
    
    
    //__________________________________________________________________________
    for(int i=startRowNum;i<startRowNum + imageAmount;i++){ //for the entire image batch,
        vector<vector<vector<double>>> image; //(1,28,28)
        int _label;
        getMNISTdata(image, _label, i, "./csv/mnist_train.csv");
        vector<vector<double>> label(10, vector<double>(1,0)); //(10, 1) one-hot-encoded
        label[_label][0] = 1;
        double loss;
        vector<vector<vector<vector<double>>>> _df1;
        vector<vector<vector<vector<double>>>> _df2;
        vector<vector<double>> _dw3;
        vector<vector<double>> _dw4;
        vector<vector<double>> _db1;
        vector<vector<double>> _db2;
        vector<vector<double>> _db3;
        vector<vector<double>> _db4;
        conv(loss, _df1, _df2, _dw3, _dw4, _db1, _db2, _db3, _db4, image, label);
        _cost+=loss;
        
        //Entire image batch's learned gradients are each summed up
        add4D(df1, df1, _df1);
        add4D(df2, df2, _df2);
        add2D(dw3, dw3, _dw3);
        add2D(dw4, dw4, _dw4);
        add2D(db1, db1, _db1);
        add2D(db2, db2, _db2);
        add2D(db3, db3, _db3);
        add2D(db4, db4, _db4);
    }
    
    //__________________________________________________________________________
    //Each parameter update (Adam gradient descent)
    // v(t) = beta1*v(t-1) + (1-beta1)*gradient, here the gradient should be averaged out by dividing imageAmount
    // s(t) = beta2*s(t-1) + (1-beta2)*gradient^2, gradient = averaged out gradient
    // w(t) = w(t-1) - lr*v(t)/(sqrt[s(t)+epsilon])
    //__________________________________________________________________________
    vector<vector<vector<vector<double>>>> temp;
    vector<vector<vector<vector<double>>>> temp2;
    vector<vector<double>> temp3;
    vector<vector<double>> temp4;
    double epsilon = 0.00000001;
    
    /*f1*/
    //v(t)
    mult4D(temp, v1, beta1);
    mult4D(temp2, df1, 1-beta1);
    divi4D(temp2, temp2, imageAmount); //averaged out gradient
    add4D(v1, temp, temp2);
    //s(t)
    mult4D(temp, s1, beta2);
    divi4D(temp2, df1, imageAmount); //averaged out gradient
    square4D(temp2, temp2);
    mult4D(temp2, temp2, 1-beta2);
    add4D(s1, temp, temp2);
    //update
    mult4D(temp, v1, lr);
    addN4D(temp2, s1, epsilon);
    sqrt4D(temp2, temp2);
    divi4DMat(temp, temp, temp2);
    sub4DMat(f1, f1, temp);
    
    /*b1*/
    //v(t)
    mult2D(temp3, bv1, beta1);
    mult2D(temp4, db1, 1-beta1);
    divi2D(temp4, temp4, imageAmount);
    add2D(bv1, temp3, temp4);
    //s(t)
    mult2D(temp3, bs1, beta2);
    divi2D(temp4, db1, imageAmount);
    square2D(temp4, temp4);
    mult2D(temp4, temp4, 1-beta2);
    add2D(bs1, temp3, temp4);
    //update
    mult2D(temp3, bv1, lr);
    addN2D(temp4, bs1, epsilon);
    sqrt2D(temp4, temp4);
    divi2DMat(temp3, temp3, temp4);
    sub2D(b1, b1, temp3);
    
    /*f2*/
    //v(t)
    mult4D(temp, v2, beta1);
    mult4D(temp2, df2, 1-beta1);
    divi4D(temp2, temp2, imageAmount);
    add4D(v2, temp, temp2);
    //s(t)
    mult4D(temp, s2, beta2);
    divi4D(temp2, df2, imageAmount);
    square4D(temp2, temp2);
    mult4D(temp2, temp2, 1-beta2);
    add4D(s2, temp, temp2);
    //update
    mult4D(temp, v2, lr);
    addN4D(temp2, s2, epsilon);
    sqrt4D(temp2, temp2);
    divi4DMat(temp, temp, temp2);
    sub4DMat(f2, f2, temp);
    
    /*b2*/
    //v(t)
    mult2D(temp3, bv2, beta1);
    mult2D(temp4, db2, 1-beta1);
    divi2D(temp4, temp4, imageAmount);
    add2D(bv2, temp3, temp4);
    //s(t)
    mult2D(temp3, bs2, beta2);
    divi2D(temp4, db2, imageAmount);
    square2D(temp4, temp4);
    mult2D(temp4, temp4, 1-beta2);
    add2D(bs2, temp3, temp4);
    //update
    mult2D(temp3, bv2, lr);
    addN2D(temp4, bs2, epsilon);
    sqrt2D(temp4, temp4);
    divi2DMat(temp3, temp3, temp4);
    sub2D(b2, b2, temp3);
    
    /*w3*/
    //v(t)
    mult2D(temp3, v3, beta1);
    mult2D(temp4, dw3, 1-beta1);
    divi2D(temp4, temp4, imageAmount);
    add2D(v3, temp3, temp4);
    //s(t)
    mult2D(temp3, s3, beta2);
    divi2D(temp4, dw3, imageAmount);
    square2D(temp4, temp4);
    mult2D(temp4, temp4, 1-beta2);
    add2D(s3, temp3, temp4);
    //update
    mult2D(temp3, v3, lr);
    addN2D(temp4, s3, epsilon);
    sqrt2D(temp4, temp4);
    divi2DMat(temp3, temp3, temp4);
    sub2D(w3, w3, temp3);
    
    /*b3*/
    //v(t)
    mult2D(temp3, bv3, beta1);
    mult2D(temp4, db3, 1-beta1);
    divi2D(temp4, temp4, imageAmount);
    add2D(bv3, temp3, temp4);
    //s(t)
    mult2D(temp3, bs3, beta2);
    divi2D(temp4, db3, imageAmount);
    square2D(temp4, temp4);
    mult2D(temp4, temp4, 1-beta2);
    add2D(bs3, temp3, temp4);
    //update
    mult2D(temp3, bv3, lr);
    addN2D(temp4, bs3, epsilon);
    sqrt2D(temp4, temp4);
    divi2DMat(temp3, temp3, temp4);
    sub2D(b3, b3, temp3);
    
    /*w4*/
    //v(t)
    mult2D(temp3, v4, beta1);
    mult2D(temp4, dw4, 1-beta1);
    divi2D(temp4, temp4, imageAmount);
    add2D(v4, temp3, temp4);
    //s(t)
    mult2D(temp3, s4, beta2);
    divi2D(temp4, dw4, imageAmount);
    square2D(temp4, temp4);
    mult2D(temp4, temp4, 1-beta2);
    add2D(s4, temp3, temp4);
    //update
    mult2D(temp3, v4, lr);
    addN2D(temp4, s4, epsilon);
    sqrt2D(temp4, temp4);
    divi2DMat(temp3, temp3, temp4);
    sub2D(w4, w4, temp3);
    
    /*b4*/
    //v(t)
    mult2D(temp3, bv4, beta1);
    mult2D(temp4, db4, 1-beta1);
    divi2D(temp4, temp4, imageAmount);
    add2D(bv4, temp3, temp4);
    //s(t)
    mult2D(temp3, bs4, beta2);
    divi2D(temp4, db4, imageAmount);
    square2D(temp4, temp4);
    mult2D(temp4, temp4, 1-beta2);
    add2D(bs4, temp3, temp4);
    //update
    mult2D(temp3, bv4, lr);
    addN2D(temp4, bs4, epsilon);
    sqrt2D(temp4, temp4);
    divi2DMat(temp3, temp3, temp4);
    sub2D(b4, b4, temp3);
    
    _cost/=imageAmount;
    cost.push_back(_cost);
}

void CNN::train(int epochs, int dataAmount, int batchSize){
    int batchNum = dataAmount/batchSize;
    vector<double> costs;
    cout<<"Start Training....."<<endl;
    cout<<"Training model with "<<dataAmount<<" images"<<endl;
    cout<<endl;
    for(int i=0;i<epochs;i++){
        //For each epoch,
        cout<<"Epoch "<<i+1<<": [";
        for(int j=0;j<batchNum;j++){//batch iteration
            adamGD(batchSize, j*batchSize, costs);
            if(j>0 and j%(((int) batchNum/10))==0){
                cout<<"=";
            }
        }
        cout<<"] -- Completed with the cost: "<<costs.back()<<endl;
    }
    
    cout<<endl;
    cout<<"Training completed"<<endl;
}

void CNN::predict(vector<vector<double>>&_probs,vector<vector<vector<double>>> image){
    
    vector<vector<vector<double>>> conv1;
    convolution(conv1, image, f1, b1); //f1 = (8,1,5,5)
    ReLU(conv1);
    
    //conv2 layer, input shape = (8,24,24)
    vector<vector<vector<double>>> conv2;
    convolution(conv2, conv1, f2, b2); //f2 = (8,8,5,5)
    ReLU(conv2);
    
    //Maxpool layer, input shape = (8,20,20)
    vector<vector<vector<double>>>pooled;
    maxpool(pooled, conv2);
    
    //Flatten, input shape = (8,10,10)
    vector<vector<double>>flat;
    for(auto &row:pooled){
        for(auto &col:row){
            for(auto &elem:col){
                flat.push_back({elem});
            }
        }
    }
    
    //FC Dense layer
    vector<vector<double>>z;
    dot(z,w3,flat); // w3 = (128, 8*100), flat = (800,1) after dot
    add2D(z, z, b3); //z = (flat)*w3+b3
    ReLU2D(z);
    
    //FC Dense output layer, input shape = (128,1)
    vector<vector<double>>out;
    dot(out, w4, z); //w4 = (10,128), z=(128,1)
    add2D(out, out, b4); // out = zw4 + b4
    vector<vector<double>>probs; //(10, 1)
    softmax(probs, out);
    _probs = probs;
}

void CNN::getMNISTdata(vector<vector<vector<double>>>&d, int &l, int rowNum, string fileName){
    vector<string> read;
    getRow(read, fileName, rowNum);
    l = stoi(read[0]); //label
    vector<vector<double>> mnist1;
    
    //getting each pixel values
    int index=1;
    for(int i=0;i<28;i++){
        vector<double> temp;
        for(int j=0;j<28;j++){
            temp.push_back(stoi(read[index]));
            index++;
        }
        mnist1.push_back(temp);
    }
    //mnist1 = 28x28 pixel data
    //now, normalize the pixel val
    double mean1, std1;
    meanAll(mean1, mnist1);
    stdAll(std1, mnist1);
    vector<vector<double>>result(28,vector<double>(28,0));
    for(int i=0;i<28;i++){
        for(int j=0;j<28;j++){
            result[i][j] = (mnist1[i][j] - mean1)/std1;
        }
    }
    d.push_back(result); //d = (1,28,28)
}
