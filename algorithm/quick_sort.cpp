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


// partition��   ���±�left��ʼright����������a�е���������
// a:            array  to sort, ��Ҫ���������
// left,right:   ����a����Ҫ�������е��±�
// ����ֵ��      ���ֺ����м�������׼��λ����㣩�����±�
int partition(int a[], int left, int right)
{
	int i,j,pivot; 

    if(left>=right) 
       return left; 
                                
    pivot = a[left]; //save the pivot data
    i=left;          // i point, from left(low)
    j=right;         // j point, form right(high)


	// ------- ���Ͻ�����λ�����֣�i����������j  ------- 
    while(i!=j) 
    { 
		
		// search from right: ��j��Ӧ�����ֱȻ������ߵȣ�&& ��jû������i����j������
        while(a[j]>=pivot && i<j) 
			j--; 

		//search from left: ��i��Ӧ�����ֱȻ���С���ߵȣ�&&��û������j�� ��i������
		// i�ӻ��㿪ʼ�ߣ�����ߵ�j��
        while(a[i]<=pivot && i<j) 
			i++; 		

		// change the number�� �Ȼ�׼���a[i] �� �� ��׼С��a[j] ���н���
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

//quicksort�����±�left��ʼright����������a�е�����������
// a:            array  to sort, ��Ҫ���������
//left,right:   ����a����Ҫ�������е��±�
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

    int a[NUM];  // �洢��Ҫ�������
    int n;       // �ж��ٸ���Ҫ�����������Ҫ����NUM

    
	//read the input  ������Ҫ���������
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
