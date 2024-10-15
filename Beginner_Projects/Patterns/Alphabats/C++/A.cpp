#include <iostream>
#include <conio.h>
using namespace std;

int main()
{
    int row = 0;
    for (; row < 5; row++)
    {
        int col = 0;
        for (; col < 5; col++)
        {
            if (((row == 0 || row == 2) and (col != 0 and col != 4)) ||
                ((col == 0 || col == 4) and (row != 0)))
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