#include <bits/stdc++.h>
#include <iostream>
#include <string>
#include <iomanip>
#include <omp.h>
#include <sstream>
#include <fstream>
using namespace std;

//Variables for obtaining line of best fit
double b0 = 0; 
double b1 = 0; 
double b2 = 0; 
double b3 = 0; 

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
    double error[17550]; // for storing the error values
    double err;          // for calculating error on each stage
    double alpha = 0.01; // initializing our learning rate
    double e = 2.718281828;

    /*Training Phase*/
    for (int i = 0; i < 17550; i++) 
    { //Since there are 350 values in our dataset and we want to run for 50 batches so total for loop run 17550 times

        //for accessing index after every batch
        int idx = i % 50; 

        //making the prediction
        double p = -(b0 + b1 * x1[idx] + b2 * x2[idx] + b3 * x3[idx]);

        //calculating final prediction applying sigmoid 
        double pred = 1 / (1 + pow(e, p)); 

        err = y[idx] - pred; //calculating the error

        //obtaining the line of best fit
        b0 = b0 - alpha * err * pred * (1 - pred) * 1.0;     //updating b0
        b1 = b1 + alpha * err * pred * (1 - pred) * x1[idx]; //updating b1
        b2 = b2 + alpha * err * pred * (1 - pred) * x2[idx]; //updating b2
        b3 = b3 + alpha * err * pred * (1 - pred) * x3[idx]; //updating b3


        //printing values for each training step
        cout << "\tB0= " << b0 << " " << "\t\tB1= " << b1 << " " << "\t\tB2= " << b2 << "\t\tB3= " << b3 << "\t\tError=" << err << endl; 
        error[i]=err;
    }

    //custom sort based on absolute error difference
    bubbleSort(error,17550); 


    cout << "Final Values are: " << "\tB0=" << b0 << " " << "\tB1=" << b1 << " " << "\tB2=" << b2 << "\tB3=" << b3 <<"\tError=" << error[0]<<endl;

}

//Testing the trained Stochastic Model
void test(double test1, double test2, double test3) 
{
    //make prediction
    double pred = b0 + b1 * test1 + b2 * test2 + b3*test3;
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
}

int main() 
{
    //Input dataset arrays
    double x1[351];
    double x2[351];
    double x3[351];
    double y[351];

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


    double start,end;
    start=omp_get_wtime();
    //Training Phase
    train(x1, x2,x3, y);
    end=omp_get_wtime();

    //Testing Phase
    double test1=0.5131, test2=-0.00015, test3=0.52099; 
    test(test1, test2, test3);

    //Time Taken
    cout<<"Time "<<end-start<<" seconds"<<endl;
    return 0;
}