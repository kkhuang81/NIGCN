#ifndef BASE_H
#define BASE_H

#include "graph.h"
#include <set>

using namespace std;

class Base :public Graph {

public:
    Base(string dataStr, uint nn, uint mm, int thrdnum, double ome, double tau1, double eps, double rho1, int size, string TS) :Graph(dataStr, nn, mm, thrdnum, ome, tau1, eps, rho1, size, TS) {
    }
           
    void ForwardPushOPT(int start, int end)
    {        
        Node_Set* record = new Node_Set(n);
        for (int idx = start; idx < end;idx++)
        {
            auto snode = targetset[idx];
            int level = tau * log(2 * m / sqrt(1.0 * dmin * deg[snode])) / LnSqrtAvgDeg;  // a parameter C            
            maxlevel = maxlevel > level? maxlevel: level;                     
            record->Push(snode, weights[0]);
            //record->Push(snode, weights[0] * (double)rwsize);
            uint node;
            for (uint rw = 0;rw < rwsize;rw++)
            {
                node = snode;
                for (int L = 1;L <= level;L++)
                {
                    node = el[pl[node] + xorshf96() % deg[node]];
                    record->Push(node, weights[L] / (double)rwsize);
                    //record->Push(node, weights[L]);
                }		
                if (record->KeyNumber > 1.0 / epsilon / epsilon)break;                                                                                                       
            }            
            

            /////////////////top-K//////////////////////////
            /*
            set<pair<double, uint>>wei_node;
            //cout << "Neighbor set: " << record->KeyNumber << endl;
            while (record->KeyNumber > 0)
            {
                auto Node_Wei = record->Pop();
                int node = Node_Wei.first;
                //double weight = Node_Wei.second / (double)rwsize;
                double weight = Node_Wei.second;
                wei_node.insert(make_pair(weight, node));
            }
            double cnt=0.0, thrshold=1.0 / 0.05 / 0.05;
            for(auto it=wei_node.rbegin();it!=wei_node.rend();++it)
            {
                double weight=it->first / wsum;
                //cout<< "weight: "<<weight<<endl;
                auto node=it->second;    
                for (int k = 0;k < ncols;k++)newFeat[idx][k] += feature[node][k] * weight;
                cnt+=1.0;
                if(cnt>thrshold)break;
            }
            */
            /////////////////top-K//////////////////////////
            
            while (record->KeyNumber > 0)
            {
                auto Node_Wei = record->Pop();
                int node = Node_Wei.first;                
                double weight = Node_Wei.second / wsum;
                for (int k = 0;k < ncols;k++)newFeat[idx][k] += feature[node][k] * weight;
            }
            
        }
    }

    void ForwardPush_GNN()
    {
        struct timeval t_start, t_end;
        double timeCost;        
        vector<thread> threads;        
        int gap = setSize / NUMTHREAD;
        int start = 0, end = 0, ti = 0;        
        gettimeofday(&t_start, NULL);
        for (ti = 1; ti <= setSize % NUMTHREAD; ti++)
        {
            start = end;
            end += ceil((double)setSize / NUMTHREAD);
            threads.push_back(thread(&Base::ForwardPushOPT, this, start, end)); 
        }
        for (; ti <= NUMTHREAD; ti++)
        {
            start = end;
            end += gap;
            threads.push_back(thread(&Base::ForwardPushOPT, this, start, end));
        }
        for (int t = 0; t < NUMTHREAD; t++)threads[t].join();
        gettimeofday(&t_end, NULL);

        vector<thread>().swap(threads);

        timeCost = t_end.tv_sec - t_start.tv_sec + (t_end.tv_usec - t_start.tv_usec) / 1000000.0;
        cout << dataset << " pre-computation cost: " << timeCost << " s" << endl;
        cout << " maxlevel: " << maxlevel << endl;
    }
};


#endif
