#include <stdio.h>
#include <stdlib.h>
#include <time.h>
clock_t start,stop;

#define N 10000
void merge(int a[],int b[],int start,int mid,int end)
{
    int i=start,j=mid+1,k=start;
    while(i!=mid+1&&j!=end+1)
    {
        if(a[i]>a[j])
            b[k++]=a[j++];
        else b[k++]=a[i++];
    }
    while(i!=mid+1)
        b[k++]=a[i++];
    while(j!=end+1)
        b[k++]=a[j++];
    for(i=start;i<=end;i++)
        a[i]=b[i];
}
void mergesort(int a[],int b[],int start,int end)
{
    int mid;
    if(start<end)
    {
        mid=(start+end)/2;
        mergesort(a,b,start,mid);
        mergesort(a,b,mid+1,end);
        merge(a,b,start,mid,end);
    }
}
int main()
{
    int a[N],b[N],i;
    double time1;
    srand((unsigned int)time(0));
      for(i=0;i<N;i++)
    {
        a[i]=rand()%10000;
    }
    start=clock();
    mergesort(a,b,0,N-1);
    stop=clock();
    for(i=0;i<N;i++)
    printf("%d ",a[i]);
    printf("\n");
    time1=((double)(stop-start))/CLK_TCK;;
    printf("%f",time1);
    return 0;
}
