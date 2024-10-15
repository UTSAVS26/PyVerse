#include <iostream>
#include <conio.h>
using namespace std;

int main()
{
    for (int row = 0; row < 7; row++)
    {
        for (int col = 0; col < 5; col++)
        {
            if ((row == 6 and col != 4) or (row == 0 and col != 0) or (col == 0 and row != 0 and row < 3) or (row == 3 and col != 0 and col != 4) or col == 4 and row > 3 and row != 6)
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