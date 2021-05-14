//
//  main.cpp
//  CNN
//
//  Created by Dongjin Kim on 5/10/21.
//  Copyright © 2021 Dongjin Kim. All rights reserved.
//

#include "utils/Functions.hpp"
#include "CNN.hpp"
#include "common.hpp"

using namespace std;
int main(int argc, const char * argv[]) {
    CNN model;
    model.train(10, 100, 10);
    
    
    cout<<"Start testing on test dataset..."<<endl;
    for(int i=0;i<20;i++){
        vector<vector<vector<double>>> image;
        int label;
        model.getMNISTdata(image, label, i, "./csv/mnist_test.csv");
        vector<vector<double>> probs;
        cout<<"Correct label is: "<<label<<endl;
        model.predict(probs, image);
        double maxNum=0;
        int prediction=0;
        for(int j=0;j<probs.size();j++){
            if(probs[j][0]>maxNum){
                maxNum = probs[j][0];
                prediction = j;
            }
        }
        cout<<"Predicted label is: "<<prediction<<endl;
        cout<<endl;
        cout<<"--------------------"<<endl;
    }
}
