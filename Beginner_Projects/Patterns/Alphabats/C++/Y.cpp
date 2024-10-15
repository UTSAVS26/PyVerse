#include <iostream>
#include <conio.h>
using namespace std;

int main()
{
    for (int row = 0; row < 4; row++)
    {
        for (int col = 0; col < 5; col++)
        {
            if ((row < 3 and (row + col == 4 or row - col == 0)) or (col == 2 and row > 2))
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