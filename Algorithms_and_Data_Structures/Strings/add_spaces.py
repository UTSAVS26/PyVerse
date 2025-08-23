'''
add space at the desired locations in a string
'''

class SpaceInserter:
    def add_spaces(self, s: str, spaces: list[int]) -> str:
        result = []
        space_set = set(spaces)  # For O(1) lookup
        
        for i, char in enumerate(s):
            if i in space_set:
                result.append(' ')
            result.append(char)
        
        return ''.join(result)


if __name__ == '__main__':
    inserter = SpaceInserter()

    s = input("Enter the string: ").strip()
    spaces_str = input("Enter space indices (comma separated): ").strip()
    spaces = list(map(int, spaces_str.split(','))) if spaces_str else []

    modified_string = inserter.add_spaces(s, spaces)

    print("Modified string:", modified_string)
