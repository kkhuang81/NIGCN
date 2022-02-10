#ifndef GRAPH_H
#define GRAPH_H

#include "utils.h"

using namespace std;

class Graph
{
public:
    uint n;
    long long  m;    
    vector<uint> el;
    vector<uint> pl;
    int level;
    int NUMTHREAD;
    double lambda;
    double alpha;
    double epsilon;    
    double rrr;      
    double wgtsum;
    vector<int>targetset;
    int setSize, ncols;
    uint nrows;
    vector<vector<double>> feature;
    vector<vector<double>> newFeat;        
    vector<double> weights;
    uint rwsize;

    vector<int>deg;
    string dataset;        

    Graph(string dataStr, uint nn, uint mm, int thrdnum, int Lel, double lamb, double alph, double eps, double rrz, int size, string TS)
    {
        dataset = dataStr;
        level = Lel;
        NUMTHREAD = thrdnum;
        lambda = lamb;
        alpha = alph;
        epsilon = eps;
        rrr = rrz;
        n = nn;
        m = mm;
        el = vector<uint>(m);
        pl = vector<uint>(n + 1);
        
        LoadGraph();
        if (dataStr.find("papers100M") != string::npos)LoadPapersFeatures();
        else LoadFeatures();
        
        setSize = size;        
        string s;
        int i = 0;       

        targetset = vector<int>(setSize, 0);
        newFeat = vector<vector<double>>(setSize, vector<double>(ncols, 0.0));
        stringstream ssin(TS);
        while (ssin.good() && i < setSize)
        {
            ssin >> s;
            targetset[i++] = stoi(s);
        }  
        weights = vector<double>(level+1, 0);
        
        if (lambda < 1.0)        
        {        
            weights[0] = alpha;
            wgtsum = weights[0];
            for (int i = 1;i <= level;i++)
            {
                weights[i] = weights[i - 1] * (1 - alpha);
                wgtsum += weights[i];
            }
        }                
        else
        {            
            weights[0] = exp(-lambda);            
            wgtsum = weights[0];
            for (int i = 1;i <= level;i++)
            {
                weights[i] = weights[i - 1] * lambda / (double)i;
                wgtsum += weights[i];
            }         
        }
        cout << "required neighbor no.: " << 1.0 / epsilon / epsilon << endl;
        cout << "Wgtsum.: " << wgtsum << endl;
        double delta = 0.01;        
	    rwsize = 8.0 * log(wgtsum / epsilon / delta) / epsilon + 1; //failure probability 1%
        cout << "random walk size: " << rwsize << " node num:" << n << endl;
    }

    void LoadGraph()
    {                
        string dataset_el = dataset + "_adj_el.txt";
        const char* p1 = dataset_el.c_str();
        if (FILE* f1 = fopen(p1, "rb"))
        {
            uint rtn = fread(el.data(), sizeof el[0], el.size(), f1);
            if (rtn != m)
                cout << "Error! " << dataset_el << " Incorrect read!" << endl;
            fclose(f1);
        }
        else
        {
            cout << dataset_el << " Not Exists." << endl;
            exit(1);
        }
        string dataset_pl =  dataset + "_adj_pl.txt";
        const char* p2 = dataset_pl.c_str();

        if (FILE* f2 = fopen(p2, "rb"))
        {
            uint rtn = fread(pl.data(), sizeof pl[0], pl.size(), f2);
            if (rtn != n + 1)
                cout << "Error! " << dataset_pl << " Incorrect read!" << endl;
            fclose(f2);
        }
        else
        {
            cout << dataset_pl << " Not Exists." << endl;
            exit(1);
        }        
        deg = vector<int>(n, 0);
        for (uint i = 0;i < n;i++)
            deg[i] = pl[i + 1] - pl[i];        
    }

    void LoadFeatures(){
        cnpy::NpyArray arr_mv1 = cnpy::npy_load( dataset +"_feat.npy");
        auto mv1 = arr_mv1.data<double>();
        nrows = arr_mv1.shape [0];
        ncols = arr_mv1.shape [1];        
        feature = vector<vector<double>>(nrows,vector<double>(ncols));       
        for(uint row = 0; row <nrows; row ++){
            for(int col = 0; col <ncols; col ++){
                double val = mv1[row*ncols+col];
                feature[row][col] = val;
            }
        }        

        for (uint i = 0; i < nrows; i++)
        {
            double d = pow((double)deg[i], rrr);
            if (d == 0.0) continue;             
            transform(feature[i].begin(), feature[i].end(), feature[i].begin(), bind1st(multiplies<double>(), 1.0 / d));
        }                          
    }
    
    void LoadPapersFeatures() {
        nrows = 111059956;
        ncols = 128;
        feature = vector<vector<double>>(nrows, vector<double>(ncols));        

        string dataset_feat = dataset + "_feat.txt";
        const char* p3 = dataset_feat.c_str();

        if (FILE* f3 = fopen(p3, "rb"))
        {
            vector<float>tep = vector<float>(ncols, 0.0);
            for (uint row = 0;row < nrows;row++)
            {
                size_t rtn = fread(tep.data(), sizeof tep[0], tep.size(), f3);
                if ((int)rtn != ncols)
                    cout << "Error! " << dataset_feat << " Incorrect read!" << endl;
                for (int col = 0;col < ncols;col++)
                    feature[row][col] = (double)tep[col];
            }
            fclose(f3);
        }
        else
        {
            cout << dataset_feat << " Not Exists." << endl;
            exit(1);
        }
        double d = 0.0;
        for (uint i = 0; i < nrows; i++)
        {
            if (deg[i] == 0) continue;
            d = pow((double)deg[i], rrr);
            for (int j = 0;j < ncols;j++)feature[i][j] /= d;
        }
        cout << "Papers100M feature Loaded" << endl;
    }
    
    uint x = 123456789, y = 362436069, z = 521288629;

    uint xorshf96(void) 
    {       
        uint t;
        x ^= x << 16;
        x ^= x >> 5;
        x ^= x << 1;

        t = x;
        x = y;
        y = z;
        z = t ^ x ^ y;

        return z;
    }

};

#endif
