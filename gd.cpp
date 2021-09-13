#include <bits/stdc++.h>
#include <iostream>
#include <string>
#include <iomanip>
#include <omp.h>
#include <sstream>
#include <fstream>
using namespace std;

//Variables for obtaining line of best fit
double b[4][10530]={};
//Swapping function
void swap(double *xp, double *yp) 
{ 
    double temp = *xp; 
    *xp = *yp; 
    *yp = temp; 
} 


// A function to implement bubble sort 
void bubbleSort(double arr[], int n) 
{ 
    int i, j; 
    for (i = 0; i < n-1; i++)     
    //Absolute swapping mechanism
    for (j = 0; j < n-i-1; j++) 
        if (abs(arr[j]) > abs(arr[j+1])) 
            swap(&arr[j], &arr[j+1]); 
} 


//Training using the obtained data set
void train(double *x1,double *x2,double *x3,double *y) 
{   
    double start,end;
    int i,idx;double error[10530]; // for storing the error values
    //Since there are 351 values in our dataset and we want to run for 50 batches so total for loop run 17550 times
    double err[10530]={0};          // for calculating error on each stage
    double alpha = 0.01; // initializing our learning rate
    double e = 2.718281828;
    double p[10530]={0},pred[10530]={0.5};

    start=omp_get_wtime();
    #pragma omp parallel for 
    for (i = 0; i < 10530; i++) 
    {   
        idx = i % 10;//for accessing index after every batch
        p[i] = -(b[0][i] + b[1][i] * x1[idx] + b[2][i] * x2[idx] + b[3][i] * x3[idx]);//making the prediction
        pred[i] = 1 / (1 + pow(e, p[i])); //calculating final prediction applying sigmoid 
        err[i] = y[idx] - pred[i]; //calculating the error
        for(int j=0;j<100000;j++)
        {
            b[0][i]=b[0][i] - alpha * err[i] * pred[i] * (1 - pred[i]) * 1.0;     //updating b0
            b[1][i]=b[1][i] + alpha * err[i] * pred[i] * (1 - pred[i]) * x1[idx]; //updating b1
            b[2][i]=b[2][i] + alpha * err[i] * pred[i] * (1 - pred[i]) * x2[idx]; //updating b2
            b[3][i]=b[3][i] + alpha * err[i] * pred[i] * (1 - pred[i]) * x3[idx]; //updating b3
        }
        //cout << "\tB0= " << b[0][i] << " " << "\t\tB1= " << b[1][i] << " " << "\t\tB2= " << b[2][i] << "\t\tB3= " << b[3][i] << "\t\tError=" << err << endl;
        error[i]=err[i];
    } 
    bubbleSort(error,i);
    //custom sort based on absolute error difference
    end=omp_get_wtime();
    //Time Taken
    cout<<"Time- "<<end-start<<endl;   
    cout << "Final Values are: " << "\tB0=" << b[0][10529] << " " << "\tB1=" << b[1][10529] << " " << "\tB2=" << b[2][10529] << "\tB3=" << b[3][10529] <<"\tMinimum Error=" << abs(error[0])<<endl;
}

//Testing the trained Stochastic Model
void test(double test1, double test2, double test3) 
{
    //make prediction
    double pred = b[0][10529] + b[1][10529] * test1 + b[2][10529] * test2 + b[3][10529]*test3;
    char ch;

    //cout << "The value predicted by the model= " << pred << endl;
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
}

int main() 
{
    //Input dataset arrays
    double x1[1053];
    double x2[1053];
    double x3[1053];
    double y[1053];

    //Reading the data file
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
                if (value=="g")
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


    //Training Phase
    train(x1, x2,x3, y);

    //Testing Phase
    double test1=0.35346, test2=0.69387, test3=0.68195; 
    test(test1, test2, test3);


    return 0;
}