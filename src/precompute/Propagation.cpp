#include "base.h"

using namespace std;

vector<vector<double>> ppr(string data, uint nn, uint mm, double ome, double tau1, double eps, double rho1, int size, string TS) {
    int NUMTHREAD = 40;    
    Base g(data, nn, mm, NUMTHREAD, ome, tau1, eps, rho1, size, TS);
    g.ForwardPush_GNN();
    return g.newFeat;
}



int main(int argc, char* argv[]){
    return 0;
}
