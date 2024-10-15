#include <iostream>
#include <conio.h>
using namespace std;

int main()
{
    for (int row = 0; row < 7; row++)
    {
        for (int col = 0; col < 6; col++)
        {
            if ((row * col == 0 and col < 5 and row + col != 0) or (col == 5 and row > 0 and row < 3) or
                (row == 3 and col < 5) or (row > 3 and col + 2 == row))
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