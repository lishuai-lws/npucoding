// quick_sort.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <stdio.h>
#include <windows.h>

#define    NUM   101

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


int main(int argc, char* argv[])
{

    int i; 

    int a[NUM];  // 存储需要排序的数
    int n;       // 有多少个需要排序的数，不要超过NUM

    
	//read the input  读入需要排序的数据
	printf("pls input how many numbers to sort:\n");
    scanf("%d",&n); 

	printf("pls input numbers one by one\n");
    for(i=0;i<n;i++) 
	{
		scanf("%d",&a[i]); 
	}

	// call the qucick sort function
    quicksort(a, 0,n-1); 
                             
    //print the sorted result
	printf("******sorted result******\n"); 
    for(i=0; i<n; i++) 
	{
        printf("%d ",a[i]); 
	}

	printf("\n"); 
    getchar();
    return 0; 
} 
