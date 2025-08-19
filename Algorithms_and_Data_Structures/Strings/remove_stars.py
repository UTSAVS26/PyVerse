'''
remove character to the left of star and the star itself
'''
class StarRemover:
    def remove_stars(self, s: str) -> str:
        stack = []
        for char in s:
            if char == '*':
                # Remove the closest non-star character to the left
                if stack:
                    stack.pop()
            else:
                stack.append(char)
        return ''.join(stack)


if __name__ == '__main__':
    remover = StarRemover()

    user_input = input("Enter string with stars '*': ").strip()
    result = remover.remove_stars(user_input)

    print("String after removing stars and corresponding characters:", result)
