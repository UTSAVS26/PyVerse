def TowerOfHanoi(n, from_rod, to_rod, aux_rod):
    # Base case: No moves needed if no disks
    if n == 0:
        return
    
    # Move n-1 disks from the source rod to the auxiliary rod
    TowerOfHanoi(n - 1, from_rod, aux_rod, to_rod)
    
    # Move the nth disk from the source rod to the target rod
    print(f"Move disk {n} from rod {from_rod} to rod {to_rod}")
    
    # Move the n-1 disks from the auxiliary rod to the target rod
    TowerOfHanoi(n - 1, aux_rod, to_rod, from_rod)

def getNumberOfDisks():
    # Input validation loop to ensure the user enters a valid number of disks
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
    # Get the number of disks from the user
    N = getNumberOfDisks()
    
    # Print the solution for Tower of Hanoi with N disks
    print(f"\nSolving Tower of Hanoi for {N} disks:")
    
    # Call the TowerOfHanoi function with rods labeled A, B, and C
    TowerOfHanoi(N, 'A', 'C', 'B')
