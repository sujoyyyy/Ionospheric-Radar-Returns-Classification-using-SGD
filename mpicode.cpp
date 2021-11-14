#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <bits/stdc++.h>
#include <iostream>
#include <string>
#include <iomanip>
#include <sstream>
#include <fstream>
#define NRA 351
#define MASTER 0               /* taskid of first task */
#define FROM_MASTER 1          /* setting a message type */
#define FROM_WORKER 2          /* setting a message type */
using namespace std;
int op=0;
//Variables for obtaining line of best fit
double b0=0,b1=0,b2=0,b3=0,err=0;

//Swapping function
void swap(double *xp, double *yp) 
{ 
    double temp = *xp; 
    *xp = *yp; 
    *yp = temp; 
} 
string convertToString(char* a, int size)
{
    int i;
    string s = "";
    for (i = 0; i < size; i++) {
        s = s + a[i];
        op++;
    }
    return s;
}

// A function to implement bubble sort 
void bubbleSort(double arr[], int n) 
{ 
    int i, j; 
    for (i = 0; i < n; i++)     
    //Absolute swapping mechanism
    for (j = 0; j < n-i-1; j++) 
        if (abs(arr[j]) > abs(arr[j+1])) 
            swap(&arr[j], &arr[j+1]); 
} 

int main (int argc, char *argv[])
{
	std::cout << std::fixed;
    std::cout << std::setprecision(6);
	int	numtasks,              /* number of tasks in partition */
		taskid,                /* a task identifier */
		numworkers,            /* number of worker tasks */
		source,                /* task id of message source */
		dest,                  /* task id of message destination */
		mtype,                 /* message type */
		rows,                  /* rows of matrix A sent to each worker */
		averow, extra, offset,rc; /* used to determine rows sent to each worker */
	double x1[351];
	double x2[351];
	double x3[351];
	double y[351];
	MPI_Status status;
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
	MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
	if (numtasks < 2 ) 
	{
		printf("Need at least two MPI tasks. Quitting...\n");
		MPI_Abort(MPI_COMM_WORLD, rc);
		exit(1);
	}
	numworkers = numtasks-1;
	double start = MPI_Wtime();
	if(taskid == MASTER)
	{
		FILE* fp = fopen("ionosphere_data.csv", "r");
	    char buffer[1024]; int i=0;
	    int row = 0; int column = 0;
	    while (fgets(buffer,1024, fp)) 
	    {
	       column = 0;
	       row++;
	       if (row == 1)
	           continue;

	       // Splitting the data
	       char* value = strtok(buffer, ",");
	       while (value) 
	       {
	           // Column 1
	           if (column == 0) 
	           {
	               x1[i]=stod(value);
	           }
	           // Column 2
	           if (column == 1) 
	           {
	               x2[i]=stod(value);
	           }
	           // Column 3
	           if (column ==2)
	           {
	               x3[i]=stod(value);
	           }
	           // Column 4
	           if (column == 3) 
	           {
	                string str = convertToString(value,1);
	                if (str.compare("g")==0)
	                {
	                    y[i]=1.0;  
	                }
	                else
	                {
	                    y[i]=0.0;
	                }
	                i++;
	           }
	           value = strtok(NULL, ",");
	           column++;
	       }
		}
	     
	    //Close the file
	    fclose(fp);
		averow = NRA/numworkers;
		extra = NRA%numworkers;
		offset = 0;
		mtype = FROM_MASTER;
		for (dest=1; dest<=numworkers; dest++)
      	{
			rows = (dest <= extra) ? averow+1 : averow;   	
			MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
			MPI_Send(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
			MPI_Send(&x1[offset], rows, MPI_DOUBLE, dest, mtype,MPI_COMM_WORLD);
			MPI_Send(&x2[offset], rows, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
			MPI_Send(&x3[offset], rows, MPI_DOUBLE, dest, mtype,MPI_COMM_WORLD);
			MPI_Send(&y[offset], rows, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
			offset = offset + rows;
      	}
      	/* Receive results from worker tasks */
      	mtype = FROM_WORKER;
      	for (i=1; i<=numworkers; i++)
      	{
			source = i;
			MPI_Recv(&offset, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&rows, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&b0, 1, MPI_DOUBLE, source, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&b1, 1, MPI_DOUBLE, source, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&b2, 1, MPI_DOUBLE, source, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&b3, 1, MPI_DOUBLE, source, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&err, 1, MPI_DOUBLE, source, mtype, MPI_COMM_WORLD, &status);
			//printf("Received results from task %d\n",source);
      	}

      	cout << "Final Values are: " << "\tB0=" << b0 << " " << "\tB1=" << b1 << " " << "\tB2=" << b2 << "\tB3=" << b3 <<"\tError=" << abs(err)<<endl;
      	//make prediction
	    double pred = b0 + b1 * 1 + b2 * 0.93035 + b3*-0.10868;
	    char ch;
	    cout << "The value predicted by the model= " << pred << endl;
	    if (pred > 0.5)
	    {
	        pred = 1;
	        ch='g';
	    }
	    else
	    {
	        pred = 0;
	        ch='b';
	    }
	    cout << "The class predicted by the model= " << ch<<endl;
	    double finish = MPI_Wtime();
      	printf("Done in %f seconds.\n", finish - start);
      	exit(0); 
	}


	if(taskid > MASTER)
	{
		mtype = FROM_MASTER;
		MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
		MPI_Recv(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
		MPI_Recv(&x1, rows, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
		MPI_Recv(&x2, rows, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
		MPI_Recv(&x3, rows, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
		MPI_Recv(&y, rows, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
	    int i,idx=0;
	    double error[rows*16];  // for storing the error values
	    err=y[0]-0.5; // for calculating error on each stage
	    double alpha = 0.001; // initializing our learning rate
	    double p;
	    double e = 2.718281828,pred=0;
		for(idx=1;idx<=16;idx++)
    	{
        	for (i = 1; i < rows; i++) 
        	{   
	            p = -1 * (b0 + b1*x1[idx] + b2*x2[idx] + b3*x3[idx]);//making the prediction
	            pred = 1 / (1 + pow(e, p));
	            err = y[idx]-pred; //calculating the error
	            error[i*idx]=err;
	            b0=b0 - alpha * err * pred * (1-pred);    
	            b1=b1 - alpha * err * pred * (1-pred) * x1[idx];
	            b2=b2 - alpha * err * pred * (1-pred) * x2[idx];
	            b3=b3 - alpha * err * pred * (1-pred) * x3[idx];
       		}
        	bubbleSort(error,i);   
    	}
    	mtype = FROM_WORKER;
     	MPI_Send(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
      	MPI_Send(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
      	MPI_Send(&b0, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
      	MPI_Send(&b1, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
      	MPI_Send(&b2, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
      	MPI_Send(&b3, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
      	MPI_Send(&err, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
	}
	MPI_Finalize();
	return 0;
}