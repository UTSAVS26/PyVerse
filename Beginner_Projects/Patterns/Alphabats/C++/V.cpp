#include <iostream>
#include <conio.h>
using namespace std;

int main()
{
    for (int row = 0; row < 4; row++)
    {
        for (int col = 0; col < 7; col++)
        {
            if (col - row == 0 or col + row == 6)
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