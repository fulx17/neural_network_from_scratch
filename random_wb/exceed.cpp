#include<bits/stdc++.h>
#define int long long
using namespace std;

mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

vector<int> layer_size={784,128,10};

double random_double(double l,double r)
{
    uniform_real_distribution<double> dist(l,r);
    return dist(rng);
}

void init_b()
{
    ofstream bias("../mainspace/bias.inp");

    for(int i=1;i<layer_size.size();i++)
    {
        for(int j=0;j<layer_size[i];j++)
        {
            bias<<fixed<<setprecision(6)
                <<random_double(-1.0,1.0)<<' ';
        }
        bias<<'\n';
    }

    bias.close();
}

void init_w()
{
    ofstream weight("../mainspace/weight.inp");

    for(int i=1;i<layer_size.size();i++)
    {
        double limit=sqrt(2.0/layer_size[i-1]);

        for(int j=0;j<layer_size[i];j++)
        {
            for(int k=0;k<layer_size[i-1];k++)
            {
                weight<<fixed<<setprecision(6)<<random_double(-limit,limit)<<' ';
            }
            weight<<'\n';
        }
    }

    weight.close();
}

signed main()
{
    ios_base::sync_with_stdio(0);
    cin.tie(0);

    init_b();
    init_w();
}