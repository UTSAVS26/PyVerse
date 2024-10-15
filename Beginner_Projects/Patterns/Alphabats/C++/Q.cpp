#include <iostream>
#include <conio.h>
using namespace std;

int main()
{
    for (int row = 0; row < 6; row++)
    {
        for (int col = 0; col < 7; col++)
        {
            if (((row * col == 0 && (col + row != 0) && col != 6 && row < 4)) || (col == 6 && row != 0 && row != 4) || (row == 4 && col != 0 && col < 6) || (row == 3 && col == 4))

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