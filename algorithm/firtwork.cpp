// quick_sort.cpp : Defines the entry point for the console application.
//

//#include "stdafx.h"
#include <stdio.h>
#include <windows.h>

#include<time.h>

#define    NUM   100001

//swap: exchange a[i] <=> a[j]
void  swap(int a[], int i, int j)
{
	if(a[i]==a[j])
	{
		return;
	}

	int tmp;

	tmp = a[i];
	a[i] = a[j];
	a[j] = tmp;

}


// partition：   对下标left开始right结束的数组a中的数进划分
// a:            array  to sort, 需要排序的数组
// left,right:   数组a的需要排序数列的下标
// 返回值：      划分后处于中间数（基准点位（轴点））的下标
int partition(int a[], int left, int right)
{
	int i,j,pivot;

    if(left>=right)
       return left;

    pivot = a[left]; //save the pivot data
    i=left;          // i point, from left(low)
    j=right;         // j point, form right(high)


	// ------- 不断交换错位的数字，i最终遇到了j  -------
    while(i!=j)
    {

		// search from right: （j对应的数字比基点大或者等）&& （j没有碰见i）则：j往左走
        while(a[j]>=pivot && i<j)
			j--;

		//search from left: （i对应的数字比基点小或者等）&&（没有碰见j） 则：i往右走
		// i从基点开始走，最多走到j！
        while(a[i]<=pivot && i<j)
			i++;

		// change the number： 比基准大的a[i] 与 比 基准小的a[j] 进行交换
		if(i<j)
        {
				swap(a, i, j);
		}
    }

    // ------- put the  pivot to correct place: i  -------
	swap(a, left, i);

	// ------- return the place of pivot -------
	return(i);
}

//quicksort：对下标left开始right结束的数组a中的数进行排序
// a:            array  to sort, 需要排序的数组
//left,right:   数组a的需要排序数列的下标
void quicksort(int a[], int left, int right)
{

	if (left < right)
	{
		int pivot = partition(a,left, right);

		quicksort(a, left,pivot-1);  //handle the left part
		quicksort(a, pivot+1,right); //handle the right part
	}

}





//mergesort



void swap_it(int*a, int pos1, int pos2)
{
	int tmp;

	tmp = a[pos1];
	a[pos1] = a[pos2];
	a[pos2] = tmp;

}

// print_it: print the numbers of array one by one
// a:        array name;
// num:      the number amount which saved in the array a
void print_it(int *a, int num)
{
		 for (int i=0; i<num;i++)
		 {
			 printf("%d ",a[i]);
		 }
		 printf("\n");
//		 getchar();

}

// Merge：将有序的a[low..mid]和a[mid+1..high]归并为有序的a[low..high]
// a: array to merge
// low...mid...high:  需要归并的子数列下标
void Merge(int a[], int low, int mid, int high)
{
	int *tmp;
	int	i= low;     // traverse left set
	int	j= mid+1;   // traverse right set
    int	k = 0;      // traverse all set

   // ---- malloc the buffer which save the merged numbers
	tmp=(int *)malloc((high-low+1)*sizeof(int));

   // ---- traverse the two set, and put the smaller number to the tmp buffer
	while (i<=mid && j<=high)
	{
		// 将小的那个数，放到tmp[k]中。之后，下标递增1
		(a[i]<=a[j])?(tmp[k++]=a[i++]):(tmp[k++]=a[j++]);
		/*
		if (a[i]<=a[j])
		{
			tmp[k++]=a[i++];
		}else
		{
			tmp[k++]=a[j++];
		}
		*/
	}

    // ---- put the remained numbers to the tmp buffer
	// note:  the below two case can not exsit at the same time. there is only one case:  (i<=mid) or (j<=high)
	while (i<=mid)
	{
		tmp[k++]=a[i++];
	}

	while (j<=high)
	{
		tmp[k++]=a[j++];
	}

	// ---- copy the sorted number to the array a
   for (k=0,i=low; i<high+1; k++,i++)
   {
		a[i]=tmp[k];
   }

	free(tmp);

}



// MergeSort：Sort the array by merge algorithm
// a: array,  a[low...high]
// n: the amout of a
void MergeSort(int a[], int low, int high)
{
    int mid;

    //printf("low:%d, high:%d\n", low, high); //debug 可以在这里观察递归如何进行的！

	if(low < high)
	{
		mid = (low + high)/2;

		MergeSort(a, low, mid);
		MergeSort(a, mid+1, high);

		Merge(a, low, mid, high);
	}
}


void sort(int array[], int size)
{
    int temp;
    bool madeAswap;
    do
    {
        madeAswap = false;
        for (int count = 0; count < (size - 1); count++)
        {
            if (array[count] > array[count + 1])
            {
                temp = array[count];
                array[count] = array[count + 1];
                array[count +1] = temp;
                madeAswap = true;
            }
        }
    } while (madeAswap) ;    //如果出现交换则再次循环
}

void time(int a[],int n){

	int b[NUM];
	int i;
	for(i=0;i<n;i++){
		b[i]=a[i];
	}

	clock_t startTime,endTime;

	startTime = clock();
	quicksort(b, 0,n-1);
	endTime = clock();
	printf("quicksorttime: %f\n",(double)(endTime - startTime) / CLOCKS_PER_SEC);

	for(i=0;i<n;i++){
		b[i]=a[i];
	}
	startTime = clock();
	MergeSort(b, 0, n-1);
	endTime = clock();
	printf("mergesorttime: %f\n",(double)(endTime - startTime) / CLOCKS_PER_SEC);

	for(i=0;i<n;i++){
		b[i]=a[i];
	}
	startTime = clock();
	sort(b,n);
	endTime = clock();
	printf("sorttime: %f\n",(double)(endTime - startTime) / CLOCKS_PER_SEC);
}


int main(int argc, char* argv[])
{

    int i;

    int a[NUM];  // 存储需要排序的数
    int n;       // 有多少个需要排序的数，不要超过NUM



	//read the input  读入需要排序的数据
	printf("pls input how many numbers to sort:\n");
    scanf("%d",&n);
	srand((int)time(0));

	printf("pls input numbers one by one\n");
    for(i=0;i<n;i++)
	{
		a[i]=rand();
	}

/*	for(i=0; i<n; i++)
	{
        printf("%d ",a[i]);
	}*/

	time(a,n);


    //print the sorted result
/*	printf("******sorted result******\n");
    for(i=0; i<n; i++)
	{
        printf("%d ",a[i]);
	}*/

	printf("\n");
    getchar();
    return 0;
}
