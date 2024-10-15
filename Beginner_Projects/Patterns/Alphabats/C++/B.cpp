#include <iostream>
#include <conio.h>
using namespace std;

int main()
{
    for (int row = 0; row < 5; row++)
    {
        for (int col = 0; col < 5; col++)
        {

            if (
                (row * col == 0 && col != 4) ||
                ((row == 2 || row == 4) && (col != 4)) ||
                ((row == 1 || row == 3) && (col == 4)))
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