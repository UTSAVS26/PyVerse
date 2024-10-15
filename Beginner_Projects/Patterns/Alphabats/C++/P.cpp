#include <iostream>
#include <conio.h>
using namespace std;

int main()
{
    for (int row = 0; row < 7; row++)
    {
        for (int col = 0; col < 6; col++)
        {
            if ((row * col == 0 && col != 5) || (col == 5 && (row > 0 && row < 3) || (row == 3 && col < 4)))
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