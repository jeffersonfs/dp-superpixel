//============================================================================
// Name        : Teste.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <stdio.h>
#include <vector>
#include <limits.h>

using namespace std;

void b(int v[], int n, int k){

	if ( n == 0){
		return;
	}else{
		int k;
		for(int i = 0; i < n; i++){
				printf("%d ", v[i]);
				k = 0;
				int vAux[n-1];
				for(int j =0; j < n; j++){
					if(v[i] != v[j]){
						vAux[k++] = v[j];
					}
				}
				b(vAux, n-1, k);
		}
	}
	if ( n == k){
		printf("\n");
	}


}

void  c(int k){

	int n[k];
	for (int i = 0; i < k; i++){
		n[i] = i;
		//printf("%d ", n[i]);
	}
	b(n, k, k);
}

int cN[] = {10,12,8,10};
int cL[2][4] = {{3, 4, 5, 6},
			  {2, 1, 3, 5},
			 };

const int k = 2;
int l[k];
int memo[4][4];

int graphCut(int v, int nivel){

	printf("%d\n", nivel);

	int r;
	if (v != -1 && memo[v][nivel] != -1 ){
		return memo[v][nivel];
	}
	if (v == 0){
		r = 0;
	}else{
		int min = INT_MAX;
		int e;
		for(int i = 0; i < k; i++){
			e = cL[i][v] + cN[v] +  graphCut(v-1, nivel + 1);
			if ( min > e){
				min = e;
			}
		}
		r = min;
	}
	memo[v][nivel] = r;

	return r;
}

int main() {

	//int k;
	//scanf("%d", &k);
	//c(k);
	for(int i = 0; i < 4; i++){
		for(int j = 0; j < 4; j++){
			memo[i][j] = -1;
		}
	}
	printf("%d", graphCut(3, 0));

	return 0;
}
