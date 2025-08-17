'''
multiply two complex numbers, those with a real and an imaginary part.
'''
class ComplexNumberMultiplier:
    def multiply(self, num1: str, num2: str) -> str:
        def parse_complex(num: str):
            # find the position of '+' or '-' (not at start) that splits real and imaginary part
            # excluding the first character in case the number starts with '-'
            for i in range(1, len(num)):
                if num[i] in ['+', '-'] and num[i-1] != 'e':  # avoid splitting on scientific notation accidentally
                    split_pos = i
                    break
            else:
                raise ValueError("Invalid complex number format")

            real = int(num[:split_pos])
            imag = int(num[split_pos:-1])  # skip last character 'i'
            return real, imag

        a, b = parse_complex(num1)
        c, d = parse_complex(num2)

        real = a * c - b * d
        imag = a * d + b * c

        return f"{real}+{imag}i"


if __name__ == '__main__':
    multiplier = ComplexNumberMultiplier()

    num1 = input("Enter the first complex number (format: a+bi): ").strip()
    num2 = input("Enter the second complex number (format: a+bi): ").strip()

    result = multiplier.multiply(num1, num2)

    print("Result of multiplication:", result)
