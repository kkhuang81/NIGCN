#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <iostream>
#include <algorithm>
#include <math.h>
#include <cmath>
#include <limits.h>
#include <unistd.h>
#include <fstream>
#include <sstream>
#include <assert.h>
#include <thread>
#include <string>
#include <unistd.h>
#include <sys/time.h>
#include <utility>
#include "Propagation.h"
#include "cnpy.h"

typedef unsigned int uint;

#define SetBit(A, k)     ( A[(k/32)] |= (1 << (k%32)) )
#define ClearBit(A, k)   ( A[(k/32)] &= ~(1 << (k%32)) )
#define TestBit(A, k)    ( A[(k/32)] & (1 << (k%32)) )

using namespace std;

class Node_Set {
public:
  int vert;
  int bit_vert;
  int *HashKey;
  int *HashValue;
  int KeyNumber;
  double *weight;


  Node_Set(int n) {
    vert = n;
    bit_vert = n / 32 + 1;
    HashKey = new int[vert];
    weight = new double[vert];
    HashValue = new int[bit_vert];
    
    for (int i = 0; i < vert; i++) {
      weight[i] = 0.0;
    }
    
    for (int i = 0; i < bit_vert; i++) {
      HashValue[i] = 0;
    }
    
    KeyNumber = 0;
  }


  void Push(int node, double w) {
    if (!TestBit(HashValue, node)) {
      HashKey[KeyNumber] = node;
      KeyNumber++;      
      SetBit(HashValue, node);
    }    
    weight[node] += w;
  }


  pair<int, double> Pop() {
    if (KeyNumber == 0) {
      return make_pair(-1, 0);
    } else {
      int k = HashKey[--KeyNumber];
      ClearBit(HashValue, k);
      double w = weight[k];
      weight[k] = 0.0;
      return make_pair(k, w);
    }
  }

  void Clean() {
    for (int i = 0; i < KeyNumber; i++) {
      ClearBit(HashValue, HashKey[i]);
      weight[i] = 0.0;
    }
    KeyNumber = 0;
  }

};


#endif