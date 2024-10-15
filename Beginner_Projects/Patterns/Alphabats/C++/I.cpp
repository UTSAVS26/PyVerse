#include <iostream>
#include <conio.h>
using namespace std;

int main()
{
    for (int rows = 0; rows < 5; rows++)
    {
        for (int cols = 0; cols < 5; cols++)
        {
            if (rows == 0 || rows == 4 || cols == 2)
            {
                cout << "*";
            }
            else
            {
                cout << " ";
            }
        }
        cout << endl;
    }
    getch();
}

// Created by Salfi Hacker Mansoor Bukhari