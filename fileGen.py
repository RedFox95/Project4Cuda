import random
import string

def generate_random_string(length):
    """Generate a random string of specified length."""
    characters = string.ascii_letters + string.digits + string.punctuation
    return ''.join(random.choice(characters) for _ in range(length))

def write_random_strings_to_file(file_path, n):
    """Write n random strings of length n to a file."""
    with open(file_path, 'w') as file:
        for _ in range(n):
            random_string = generate_random_string(n)
            file.write(random_string + '\n')

# Example usage:
file_path = 'random_strings.txt'
n = 5000  # Number of random strings and their lengths

write_random_strings_to_file(file_path, n)
print(f"{n} random strings of length {n} have been written to '{file_path}'.")
