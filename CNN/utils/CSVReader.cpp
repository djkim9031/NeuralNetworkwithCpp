//
//  CSVReader.cpp
//  CNN
//
//  Created by Dongjin Kim on 5/13/21.
//  Copyright © 2021 Dongjin Kim. All rights reserved.
//

#include "CSVReader.hpp"
#include <fstream>
#include <sstream>
#include <string>
#include "../common.hpp"

void getRow(vector<string> &result, string fileName, int rowNum){
    fstream fin;
    fin.open(fileName, ios::in);
    int count = 0;
    vector<string> row;
    string line, word, temp;
    getline(fin, line);
    int amount = 0;
    while (amount < 60000)
    {
      row.clear();
      getline(fin, line);
      if (amount == rowNum)
      {
        stringstream s(line);
        while (getline(s, word, ','))
        {
          row.push_back(word);
        }
        count = 1;
        break;
      }
      amount++;
    }
    if (count == 0)
      cout << "Record not found"<<endl;
    result = row;
    fin.close();
    
}
