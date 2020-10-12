#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 2
void quicksort(int a[],int n,int left,int right)
{
    int i,j,t;
    if(left<right)
    {
        i=left;
        j=right+1;
        while(1)
        {
            while(i+1<n&&a[++i]<a[left]);
            while(j-1>-1&&a[--j]>a[left]);
            if(i>=j)break;
            t=a[i],a[i]=a[j],a[j]=t;
        }
        t=a[left],a[left]=a[j],a[j]=t;
        quicksort(a,n,left,j-1);
        quicksort(a,n,j+1,right);
    }
}
int main()
{
    int a[N],i;
    //srand((unsigned int)time(0));
      for(i=0;i<N;i++)
    {
scanf("%d",&a[i]);

    }

    quicksort(a,N,0,N-1);
   for(i=0;i<N;i++)
    printf("%d ",a[i]);

    return 0;
}
//56 89 12 6 4 36 99 48 15 42
