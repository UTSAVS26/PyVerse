'''
Given an array of characters chars, compress it using the following algorithm:
Begin with an empty string s. For each group of consecutive repeating characters in chars:
    -If the group's length is 1, append the character to s.
    -Otherwise, append the character followed by the group's length.

'''
class Compressor:
    def compress(self, chars: list[str]) -> int:
        write = 0
        read = 0

        while read < len(chars):
            char = chars[read]
            count = 0

            while read < len(chars) and chars[read] == char:
                read += 1
                count += 1

            chars[write] = char
            write += 1

            if count > 1:
                for digit in str(count):
                    chars[write] = digit
                    write += 1

        return write


if __name__ == '__main__':
    compressor = Compressor()
    
    user_input = input("Enter characters (no spaces, e.g. aabbccc): ").strip()
    chars = list(user_input)
    
    new_length = compressor.compress(chars)
    
    print("New length:", new_length)
    print("Compressed array:", chars[:new_length])
