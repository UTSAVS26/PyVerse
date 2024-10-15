#include <conio.h>
#include <iostream>

using namespace std;

int main()
{
    int row = 0;
    while (row < 5)
    {
        int col = 0;
        while (col < 5)
        {
            if ((row * col == 0) || (row == 2 & (col != 4 && col != 0)))
            {
                cout << "*";
            }
            else
            {
                cout << " ";
            }
            col++;
        }
        row++;
        cout << endl;
    }
    getch();
}

// Created by Salfi Hacker Mansoor Bukhari