#include <conio.h>
#include <iostream>
using namespace std;

int main()
{
    int row = 0;
    while (row < 5)
    {
        int col = 0;
        while (col < 6)
        {
            if ((row * col == 0) || (row == 4 && (col != 0 && col != 4)) ||
                (row == 3 && col > 1))
            {
                cout << "*";
            }
            else
            {
                cout << " ";
            }
            col++;
        }
        cout << endl;
        row++;
    }
    getch();
}

// Created by Salfi Hacker Mansoor Bukhari