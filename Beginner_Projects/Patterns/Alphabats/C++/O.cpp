#include <iostream>
#include <conio.h>
using namespace std;

int main()
{
    for (int row = 0; row < 5; row++)
    {
        for (int col = 0; col < 6; col++)
        {
            if (((row * col == 0 && (col + row != 0) && col != 5 && row != 4)) || (col == 5 && row != 0 && row != 4) || (row == 4 && col != 0 && col != 5))
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


//Created by Salfi Hacker Mansoor Bukhari