def TowerOfHanoi(n, from_rod, to_rod, aux_rod):
    if n == 0:
        return
    TowerOfHanoi(n - 1, from_rod, aux_rod, to_rod)
    print(f"Move disk {n} from rod {from_rod} to rod {to_rod}")
    TowerOfHanoi(n - 1, aux_rod, to_rod, from_rod)

def getNumberOfDisks():
    while True:
        try:
            n = int(input("Enter the number of disks: "))
            if n <= 0:
                print("Number of disks must be greater than zero. Please try again.")
            else:
                return n
        except ValueError:
            print("Invalid input! Please enter a valid integer.")

if __name__ == "__main__":
    N = getNumberOfDisks()
    print(f"\nSolving Tower of Hanoi for {N} disks:")
    TowerOfHanoi(N, 'A', 'C', 'B')
