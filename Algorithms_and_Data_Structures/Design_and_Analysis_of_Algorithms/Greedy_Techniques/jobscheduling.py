def printJobScheduling(arr, t):
    n = len(arr)

    arr.sort(key=lambda x: x[2], reverse=True)

    result = [False] * t
    job = ['-1'] * t

    for i in range(len(arr)):
        for j in range(min(t - 1, arr[i][1] - 1), -1, -1):
            if not result[j]:
                result[j] = True
                job[j] = arr[i][0]
                break

    print(job)

if __name__ == '__main__':
    n = int(input("Enter the number of jobs: "))
    arr = []

    print("Enter job details in the format: JobName Deadline Profit (space-separated):")
    for _ in range(n):
        job_detail = input().split()
        job_name = job_detail[0]
        deadline = int(job_detail[1])
        profit = int(job_detail[2])
        arr.append([job_name, deadline, profit])

    t = int(input("Enter the maximum number of time slots: "))
    print("Following is the maximum profit sequence of jobs:")
    printJobScheduling(arr, t)
