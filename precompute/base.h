#ifndef BASE_H
#define BASE_H

#include "graph.h"


using namespace std;

class Base :public Graph {

public:
    Base(string dataStr, uint nn, uint mm, int thrdnum, int lel, double lambda, double alpha, double epsilon, double rrz, int size, string TS) :Graph(dataStr, nn, mm, thrdnum, lel, lambda, alpha, epsilon, rrz, size, TS) {
    }
    
    void ForwardPush(int start, int end)
    {    
        Node_Set* set1 = new Node_Set(n);
        Node_Set* set2 = new Node_Set(n);
        vector<Node_Set*>visit = { set1, set2 };
        for (int idx = start; idx < end;idx++)
        {
            auto snode = targetset[idx];
            int L = 0;   
            transform(feature[snode].begin(), feature[snode].end(), newFeat[idx].begin(), bind1st(multiplies<double>(), weights[L]));
            visit[L]->Push(snode, 1.0);            
            while (L < level && visit[L % 2]->KeyNumber > 0)
            {
                while (visit[L % 2]->KeyNumber > 0)
                {
                    auto Node_Wei = visit[L % 2]->Pop();
                    int node = Node_Wei.first;                    
                    if (deg[node] == 0)continue;                    
                    double weight = Node_Wei.second / (double)deg[node];                    
                    for (uint im = pl[node];im < pl[node + 1];im++)
                        visit[(L + 1) % 2]->Push(el[im], weight);
                }                                

                for (int i = 0;i < visit[(L + 1) % 2]->KeyNumber;i++)
                {
                    int node = visit[(L + 1) % 2]->HashKey[i];
                    double w = visit[(L + 1) % 2]->weight[node];
                    
                    for (int k = 0;k < ncols;k++)
                        newFeat[idx][k] += feature[node][k] * w * weights[L + 1];
                }
                ++L;
            }            
            visit[0]->Clean();
            visit[1]->Clean();

            if (deg[snode] == 0) continue;
            double d = pow((double)(deg[snode]), rrr);            
            transform(newFeat[idx].begin(), newFeat[idx].end(), newFeat[idx].begin(), bind1st(multiplies<double>(), d));
        }
    }        
            
    void ForwardPushOPT(int start, int end)
    {        
        Node_Set* record = new Node_Set(n);
        for (int idx = start; idx < end;idx++)
        {
            auto snode = targetset[idx];
            record->Push(snode, weights[0]);            
            uint node;
            for (uint rw = 0;rw < rwsize;rw++)
            {
                node = snode;
                for (int L = 1;L <= level;L++)
                {
                    node = el[pl[node] + xorshf96() % deg[node]];
                    record->Push(node, weights[L] / (double)rwsize);                    
                }
            }
            while (record->KeyNumber > 0)
            {
                auto Node_Wei = record->Pop();
                int node = Node_Wei.first;                
                double weight = Node_Wei.second;
                for (int k = 0;k < ncols;k++)newFeat[idx][k] += feature[node][k] * weight;
            }
            if (deg[snode] == 0) continue;
            double d = pow((double)(deg[snode]), rrr);
            for (int j = 0;j < ncols;j++)newFeat[idx][j] *= d;
        }
    }

    void ForwardPush_GNN(bool opt)
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
            if (opt)threads.push_back(thread(&Base::ForwardPushOPT, this, start, end));           
            else threads.push_back(thread(&Base::ForwardPush, this, start, end));
        }
        for (; ti <= NUMTHREAD; ti++)
        {
            start = end;
            end += gap;
            if (opt)threads.push_back(thread(&Base::ForwardPushOPT, this, start, end));            
            else threads.push_back(thread(&Base::ForwardPush, this, start, end));
        }
        for (int t = 0; t < NUMTHREAD; t++)threads[t].join();
        gettimeofday(&t_end, NULL);

        vector<thread>().swap(threads);

        timeCost = t_end.tv_sec - t_start.tv_sec + (t_end.tv_usec - t_start.tv_usec) / 1000000.0;
        cout << dataset << " pre-computation cost: " << timeCost << " s" << endl;
    }
};


#endif
