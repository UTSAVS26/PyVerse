def input_array():
    n = int(input("Enter number of days: "))
    temps = []
    for i in range(n):
        temp = int(input(f"Enter temperature for day {i + 1}: "))
        temps.append(temp)
    return temps

def display_array(arr):
    print("\nResult array:")
    for x in arr:
        print(x, end=" ")
    print()

def daily_temperatures(temperatures):
    n = len(temperatures)
    answer = [0] * n
    stack = []
    for i, temp in enumerate(temperatures):
        while stack and temperatures[stack[-1]] < temp:
            prev_index = stack.pop()
            answer[prev_index] = i - prev_index
        stack.append(i)
    return answer

if __name__ == "__main__":
    temperatures = input_array()
    result = daily_temperatures(temperatures)
    display_array(result)
