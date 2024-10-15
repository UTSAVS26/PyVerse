#include <iostream>
#include <conio.h>
using namespace std;

int main()
{
    for (int row = 0; row < 4; row++)
    {
        for (int col = 0; col < 5; col++)
        {
            if (((col == 0 or col == 4) and row != 3) or (row == 3 and (col != 0 and col != 4)))
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