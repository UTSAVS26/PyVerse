#include <iostream>
#include <conio.h>

using namespace std;

int main()
{
    int row = 0;
    while (row < 4)
    {
        int col = 0;
        while (col < 4)
        {
            if ((row * col == 0 && col != 3) || (row == 3 && col != 3) || (col == 3 && (row > 0 && row < 3)))
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