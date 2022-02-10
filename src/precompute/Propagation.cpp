#include "base.h"

using namespace std;

vector<vector<double>> ppr(string data, uint nn, uint mm, int level, double lambda, double alpha, double epsilon, double rrr, int size, string TS, bool opt) {
    int NUMTHREAD = 40;    
    Base g(data, nn, mm, NUMTHREAD, level, lambda, alpha, epsilon, rrr, size, TS);
    g.ForwardPush_GNN(opt);
    return g.newFeat;
}

int main(int argc, char* argv[]){
    return 0;
}
