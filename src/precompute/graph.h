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
    int NUMTHREAD;
    double omega;
    double epsilon;    
    double rho;   
    double LnSqrtAvgDeg;  
    double tau; 
    double dmin;
    vector<int>targetset;
    int setSize, ncols;
    uint nrows;
    vector<vector<double>> feature;
    vector<vector<double>> newFeat;        
    vector<double> weights;
    double wsum;
    uint rwsize;

    int maxlevel;

    vector<int>deg;
    string dataset;    

    Graph(string dataStr, uint nn, uint mm, int thrdnum, double ome, double tau1, double eps, double rho1, int size, string TS)
    {
        dataset = dataStr;
        NUMTHREAD = thrdnum;
        omega = ome;
        epsilon = eps;
        tau = tau1;
        rho = rho1;
        n = nn;
        m = mm;
        maxlevel = 0;
        el = vector<uint>(m);
        pl = vector<uint>(n + 1);
        LoadGraph();
        if (dataStr.find("papers100M") != string::npos)LoadPapersFeatures();
        else LoadFeatures();      
        setSize = size;
        LnSqrtAvgDeg = log(sqrt(2*double (m) / double (n)));
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
        int level = 512;
        weights = vector<double>(level, 0);
        weights[0] = 1.0;            
        wsum = weights[0];

        for (int i = 1;i < level;i++)
        {
            weights[i] = weights[i - 1] * omega / pow((double)i, rho);
            wsum += weights[i];
        }        

        //cout << "required neighbor no.: " << 1.0 / epsilon / epsilon << endl;
        //cout << "Wgtsum.: " << wgtsum << endl;
        double delta = 0.01;
        //rwsize = 2.0 * log(wgtsum / epsilon / delta) / epsilon / epsilon + 1; //failure probability 1%
	    rwsize = 8.0 * log(1.0 / epsilon / delta) / epsilon + 1; //failure probability 1%
        cout << "random walk size: " << rwsize << " node num:" << n << endl;
    }

    void LoadGraph()
    {        
        //cout << "path: " << dataset << endl;
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
        //cout << "Load finished" << endl;
        
        dmin=n;
        deg = vector<int>(n, 0);
        for (uint i = 0;i < n;i++)
        {
            deg[i] = pl[i + 1] - pl[i];        
            dmin = min(dmin, 1.0*deg[i]);
        }
        dmin=max(1.0, dmin);            
    }

    void LoadFeatures(){
        cnpy::NpyArray arr_mv1 = cnpy::npy_load( dataset +"_feat.npy");
        auto mv1 = arr_mv1.data<double>();
        nrows = arr_mv1.shape [0];
        ncols = arr_mv1.shape [1];
        //cout << "nrow: " << nrows << " cols: " << ncols << endl;
        feature = vector<vector<double>>(nrows,vector<double>(ncols));       
        for(uint row = 0; row <nrows; row ++){
            for(int col = 0; col <ncols; col ++){
                double val = mv1[row*ncols+col];
                feature[row][col] = val;
            }
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
