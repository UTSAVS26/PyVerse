import itertools
import string
import hashlib
import time

def hash_password(password):
    """ Hasheing the password using SHA-256 """
    return hashlib.sha256(password.encode()).hexdigest()

def brute_force_crack(hash_to_crack, min_length, max_length):
    chars = string.ascii_letters + string.digits + string.punctuation
    attempts = 0

    # Starting time to calculate how long it takes
    start_time = time.time()

    for length in range(min_length, max_length + 1):  # Generate combinations within the specified range
        # Generating all possible combinations of the given length
        for guess in itertools.product(chars, repeat=length):
            attempts += 1
            guess_password = ''.join(guess)
            guess_hash = hash_password(guess_password)

            if guess_hash == hash_to_crack:
                end_time = time.time()
                time_taken = end_time - start_time
                print(f"Password '{guess_password}' cracked in {attempts} attempts and {time_taken:.2f} seconds!")
                return

    print("Password not found.")

# Driver Code
if __name__ == "__main__":
    min_length = 4  # Minimum length of the password
    max_length = 6  # Maximum length of the password
    while True:
        password = str(input(" Enter the password: "))
        if len(password) >= min_length and len(password) <= max_length:
            hashed_password = hash_password(password)
            brute_force_crack(hashed_password, min_length, max_length)
        else:
            print("Re-enter password")
