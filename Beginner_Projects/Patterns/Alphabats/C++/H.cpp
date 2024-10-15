#include <iostream>
using namespace std;
#include <conio.h>

int main()
{
    for (int row = 0; row < 5; row++)
    {
        for (int col = 0; col < 5; col++)
        {
            if (col == 0 || col == 4 || (row == 2 && (col != 0 && col != 4)))
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
    return 0;
}

// Created by Salfi Hacker Mansoor Bukhari