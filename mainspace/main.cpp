#include<bits/stdc++.h>
#define int long long
#define pb push_back
#define x first
#define y second
#define getbit(u,i) ((u>>i)&1)
#define all(x) x.begin(),x.end()
#define N 200001
using namespace std;

typedef pair<int,int> ii;
typedef pair<double,double> dd;
typedef vector<vector<double>> mat;

const int layer=3;
const int data_train_size=60000;
const int epochs=200;
const int data_test_size=10000;
const int mini_batch_size=32;
const int batch_number=data_train_size/mini_batch_size;
const double eta=0.01;

const vector<int> layer_size={784,128,10};

mat biases[layer],weights[layer],nabla_b[layer],nabla_w[layer],delta_nabla_b[layer],delta_nabla_w[layer];
pair<mat,mat> train[data_train_size+1],test[data_test_size+1];

mt19937 rng((unsigned)chrono::steady_clock::now().time_since_epoch().count());

double random_double(double l,double r)
{
    uniform_real_distribution<double> dist(l,r);
    return dist(rng);
}

mat operator + (mat a,const mat &b)
{
    assert(a.size()==b.size()&&a[0].size()==b[0].size());

    for(int i=0;i<(int)a.size();i++)
    {
        for(int j=0;j<(int)a[i].size();j++)
        {
            a[i][j]+=b[i][j];
        }
    }

    return a;
}

mat operator - (mat a,const mat &b)
{
    assert(a.size()==b.size()&&a[0].size()==b[0].size());

    for(int i=0;i<(int)a.size();i++)
    {
        for(int j=0;j<(int)a[i].size();j++)
        {
            a[i][j]-=b[i][j];
        }
    }

    return a;
}

mat operator * (mat a,const mat &b)
{
    assert(a.size()==b.size()&&a[0].size()==b[0].size());

    for(int i=0;i<(int)a.size();i++)
    {
        for(int j=0;j<(int)a[i].size();j++)
        {
            a[i][j]*=b[i][j];
        }
    }

    return a;
}

mat operator * (double a,mat b)
{
    for(int i=0;i<(int)b.size();i++)
    {
        for(int j=0;j<(int)b[i].size();j++)
        {
            b[i][j]*=a;
        }
    }

    return b;
}

mat dot(const mat &a,const mat &b)
{
    assert(a[0].size()==b.size());

    mat c(a.size(),vector<double>(b[0].size(),0));

    for(int i=0;i<(int)a.size();i++)
    {
        for(int k=0;k<(int)b.size();k++)
        {
            for(int j=0;j<(int)b[0].size();j++)
            {
                c[i][j]+=a[i][k]*b[k][j];
            }
        }
    }

    return c;
}

mat transpose(const mat &a)
{
    mat b(a[0].size(),vector<double>(a.size()));

    for(int i=0;i<(int)a.size();i++)
    {
        for(int j=0;j<(int)a[0].size();j++)
        {
            b[j][i]=a[i][j];
        }
    }

    return b;
}

double relu_point(double a)
{
    return max(0.0,a);
}

double relu_prime_point(double a)
{
    return a>0 ? 1.0 : 0.0;
}

mat relu(mat a)
{
    for(int i=0;i<(int)a.size();i++)
    {
        for(int j=0;j<(int)a[i].size();j++)
        {
            a[i][j]=relu_point(a[i][j]);
        }
    }
    return a;
}

mat relu_prime(mat a)
{
    for(int i=0;i<(int)a.size();i++)
    {
        for(int j=0;j<(int)a[i].size();j++)
        {
            a[i][j]=relu_prime_point(a[i][j]);
        }
    }
    return a;
}

int Max(mat a)
{
    int cur=0;
    double val=-1e100;
    for(int i=0;i<(int)a.size();i++)
    {
        if(a[i][0]>val)
        {
            val=a[i][0];
            cur=i;
        }
    }
    return cur;
}

void init()
{
    ifstream bias("bias.inp");
    ifstream weight("weight.inp");

    bool okb=bias.good();
    bool okw=weight.good();
    
    if(okb&&okw)
    {
        for(int i=1;i<layer;i++)
        {
            biases[i].assign(layer_size[i],vector<double>(1,0));
            for(int j=0;j<layer_size[i];j++)
            {
                double u; bias>>u;
                biases[i][j][0]=u;
            }
        }

        for(int i=1;i<layer;i++)
        {
            weights[i].assign(layer_size[i],vector<double>(layer_size[i-1],0));
            for(int j=0;j<layer_size[i];j++)
            {
                for(int k=0;k<layer_size[i-1];k++)
                {
                    double u; weight>>u;
                    weights[i][j][k]=u;
                }
            }
        }

        bias.close();
        weight.close();
    }
    else
    {
        for(int i=1;i<layer;i++)
        {
            biases[i].assign(layer_size[i],vector<double>(1,0));
            weights[i].assign(layer_size[i],vector<double>(layer_size[i-1],0));

            double limit=sqrt(2.0/layer_size[i-1]);

            for(int j=0;j<layer_size[i];j++)
            {
                for(int k=0;k<layer_size[i-1];k++)
                {
                    weights[i][j][k]=random_double(-limit,limit);
                }
            }
        }
    }
}

void read_train()
{
    ifstream inp("train.inp");
    for(int t=1;t<=data_train_size;t++)
    {
        mat x,y;
        x.resize(layer_size[0],vector<double>(1,0));
        y.resize(layer_size[layer-1],vector<double>(1,0));

        for(int i=0;i<layer_size[0];i++)
        {
            double u; inp>>u;
            x[i][0]=u;
        }
        for(int i=0;i<layer_size[layer-1];i++)
        {
            double u; inp>>u;
            y[i][0]=u;
        }
        train[t]={x,y};
    }
    inp.close();
}

void read_test()
{
    ifstream inp("test.inp");
    for(int t=1;t<=data_test_size;t++)
    {
        mat x,y;
        x.resize(layer_size[0],vector<double>(1,0));
        y.resize(layer_size[layer-1],vector<double>(1,0));

        for(int i=0;i<layer_size[0];i++)
        {
            double u; inp>>u;
            x[i][0]=u;
        }
        for(int i=0;i<layer_size[layer-1];i++)
        {
            double u; inp>>u;
            y[i][0]=u;
        }
        test[t]={x,y};
    }
    inp.close();
}

mat feed_forward(mat a)
{
    for(int i=1;i<layer;i++)
    {
        mat z=dot(weights[i],a)+biases[i];
        if(i==layer-1) a=z;
        else a=relu(z);
    }
    return a;
}

void print(mat a)
{
    for(int i=0;i<(int)a.size();i++)
    {
        for(int j=0;j<(int)a[0].size();j++)
        {
            cout<<a[i][j]<<' ';
        }
        cout<<'\n';
    }
}

void P(mat a)
{
    for(int i=0;i<(int)a.size();i++)
    {
        cout<<a[i][0]<<' ';
    }
    cout<<'\n';
}

void backprop(mat x,mat y)
{
    for(int i=1;i<layer;i++)
    {
        delta_nabla_b[i].assign(biases[i].size(),vector<double>(1,0));
        delta_nabla_w[i].assign(weights[i].size(),vector<double>(weights[i][0].size(),0));
    }

    mat activation=x;
    vector<mat> activations,zs;
    activations.pb(x);

    for(int i=1;i<layer;i++)
    {
        mat z=dot(weights[i],activation)+biases[i];
        zs.pb(z);
        if(i==layer-1) activation=z;
        else activation=relu(z);
        activations.pb(activation);
    }

    mat delta=(activations.back()-y);
    delta_nabla_b[layer-1]=delta;
    delta_nabla_w[layer-1]=dot(delta,transpose(activations[layer-2]));

    for(int l=layer-2;l>=1;l--)
    {
        mat sp=relu_prime(zs[l-1]);
        delta=dot(transpose(weights[l+1]),delta);
        delta=delta*sp;
        delta_nabla_b[l]=delta;
        delta_nabla_w[l]=dot(delta,transpose(activations[l-1]));
    }
}

void update_batch(int k)
{
    for(int i=1;i<layer;i++)
    {
        nabla_b[i].assign(biases[i].size(),vector<double>(1,0));
        nabla_w[i].assign(weights[i].size(),vector<double>(weights[i][0].size(),0));
    }

    for(int i=(k-1)*mini_batch_size+1;i<=k*mini_batch_size;i++)
    {
        backprop(train[i].x,train[i].y);
        for(int j=1;j<layer;j++)
        {
            nabla_b[j]=nabla_b[j]+delta_nabla_b[j];
            nabla_w[j]=nabla_w[j]+delta_nabla_w[j];
        }
    }

    for(int i=1;i<layer;i++)
    {
        weights[i]=weights[i]-((1.0*eta/mini_batch_size)*nabla_w[i]);
        biases[i]=biases[i]-((1.0*eta/mini_batch_size)*nabla_b[i]);
        assert(!isnan(weights[i][0][0]));
    }
}

int evaluate()
{
    int sum=0;
    for(int t=1;t<=data_test_size;t++)
    {
        mat x=test[t].x;
        mat y=test[t].y;
        mat c=feed_forward(x);
        int val1=Max(c);
        int val2=Max(y);
        sum+=(val1==val2);
    }
    return sum;
}

void SGD()
{
    for(int t=1;t<=epochs;t++)
    {
        shuffle(train+1,train+data_train_size+1,rng);
        for(int k=1;k<=batch_number;k++) update_batch(k);
        cout<<"Epoch "<<t<<' '<<evaluate()<<'/'<<data_test_size<<'\n';
    }
}

void rewrite()
{
    ofstream bias("bias.inp");
    for(int i=1;i<layer;i++)
    {
        for(int j=0;j<layer_size[i];j++)
        {
            bias<<fixed<<setprecision(10)<<biases[i][j][0]<<' ';
        }
        bias<<'\n';
    }
    bias.close();

    ofstream weight("weight.inp");
    for(int i=1;i<layer;i++)
    {
        for(int j=0;j<layer_size[i];j++)
        {
            for(int k=0;k<layer_size[i-1];k++)
            {
                weight<<fixed<<setprecision(10)<<weights[i][j][k]<<' ';
            }
            weight<<'\n';
        }
    }
    weight.close();
}

signed main()
{
    init();
    read_train();
    read_test();
    cout << "done loading\n";
    SGD();
    rewrite();
}