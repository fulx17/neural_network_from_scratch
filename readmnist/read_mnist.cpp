#include <bits/stdc++.h>
using namespace std;
int layer_size=15,a[41][41],b[29][29];
bool inside(int r,int c)
{
    return r<=40&&c<=40;
}
void col(int i,int j)
{
    if(inside(i,j)) a[i][j]=1;
}
void read()
{
    ifstream image("train-images.idx3-ubyte",ios::binary);
    ifstream label("train-labels.idx1-ubyte",ios::binary);
    ofstream out("train.inp");
    char number;
    for (int i = 1; i <= 16; ++i) {
        image.read(&number, sizeof(char));
	}
    for (int i = 1; i <= 8; ++i) {
        label.read(&number, sizeof(char));
	}
	for(int t=1;t<=layer_size;t++)
    {
        for(int i=1;i<=28;i++)
        {
            for(int j=1;j<=28;j++)
            {
                image.read(&number, sizeof(char));
                int r=i*1.43;
                int c=j*1.43;
                if(((int)number!=0))
                {
                    col(r-1,c-1);
                    col(r,c-1);
                    col(r-1,c);
                    col(r-1,c);
                }
            }
        }
        for(int i=1;i<=40;i++)
        {
            for(int j=1;j<=40;j++)
            {
                int sum=a[i-1][j]+a[i][j-1]+a[i+1][j]+a[i][j+1];
                if(sum>=3) a[i][j]=1;
                out<<a[i][j];
                a[i][j]=0;
            }
            out<<'\n';
        }
        label.read(&number, sizeof(char));
        for(int i=0;i<10;i++) out<<(i==(int)number)<<' ';
        out<<'\n';
    }
    out.close();
    label.close();
    image.close();
}
int main()
{
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    read();
}
