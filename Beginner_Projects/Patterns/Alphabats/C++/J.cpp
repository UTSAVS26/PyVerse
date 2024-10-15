#include <iostream>
#include <conio.h>
using namespace std;

int main()
{
    for (int row = 0; row < 7; row++)
    {
        for (int col = 0; col < 9; col++)
        {
            if (row == 0 || (col == 7 && row < 5) || (col == 6 && row == 5) || ((col == 4 || col == 3 || col == 2) && row == 6) || ((row == 5 || row == 4) && col == 0))
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