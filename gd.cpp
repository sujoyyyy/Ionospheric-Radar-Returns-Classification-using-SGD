#include <bits/stdc++.h>
#include <iostream>
#include <string>
#include <iomanip>
#include <omp.h>
#include <sstream>
#include <fstream>
using namespace std;
int op=0;
//Variables for obtaining line of best fit
double b0=0,b1=0,b2=0,b3=0;
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
    for (i = 0; i < n-1; i++)     
    //Absolute swapping mechanism
    for (j = 0; j < n-i-1; j++) 
        if (abs(arr[j]) > abs(arr[j+1])) 
            swap(&arr[j], &arr[j+1]); 
} 


//Training using the obtained data set
void train(double x1[351],double x2[351],double x3[351],double y[351]) 
{   
    double start,end;
    int i,idx=0;
    double error[5616];  // for storing the error values
    double err=y[0]-0.5; // for calculating error on each stage
    double alpha = 0.001; // initializing our learning rate
    double p;
    double e = 2.718281828,pred=0;

    start=omp_get_wtime();
    #pragma omp parallel for 
    for(idx=0;idx<=16;idx++)
    {
        for (i = 0; i < 351; i++) 
        {   
            p = -1 * (b0 + b1*x1[idx] + b2*x2[idx] + b3*x3[idx]);//making the prediction
            op+=7;
            pred = 1 / (1 + pow(e, p));
            op+=5;
            err = y[idx]-pred; //calculating the error
            op++;
            error[i*idx]=err;
            b0=b0 - alpha * err * pred * (1-pred);    
            b1=b1 - alpha * err * pred * (1-pred) * x1[idx];
            b2=b2 - alpha * err * pred * (1-pred) * x2[idx];
            b3=b3 - alpha * err * pred * (1-pred) * x3[idx];
            op+=23;
            //#pragma omp critical
            //cout << "\tB0= " << b0 << " " << "\t\t\tB1= " << b1 << " " << "\t\t\tB2= " << b2 << "\t\t\tB3= " << b3 << "\t\t\tError=" << err << endl;
        }
        bubbleSort(error,i*idx);   
    }
    end=omp_get_wtime();

    //Time Taken
    cout<<end-start<<endl;
    op++;   
    //cout << "Final Values are: " << "\tB0=" << b0 << " " << "\tB1=" << b1 << " " << "\tB2=" << b2 << "\tB3=" << b3 <<"\tMinimum Error=" << abs(error[0])<<endl;
}

//Testing the trained Stochastic Model
void test(double test1, double test2, double test3) 
{
    //make prediction
    double pred = b0 + b1 * test1 + b2 * test2 + b3*test3;
    op+=6;
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
    //cout << "The class predicted by the model= " << ch<<endl;
}

int main() 
{
    std::cout << std::fixed;
    std::cout << std::setprecision(6);
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


    //Training Phase
    train(x1, x2,x3, y);

    //Testing Phase
    double test1=1, test2=0.93035, test3=-0.10868; 
    test(test1, test2, test3);
    //cout<<"Floating point operations= "<<op<<endl;

    return 0;
}