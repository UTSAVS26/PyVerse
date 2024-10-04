def printJobScheduling(arr, t):
    n = len(arr)  # Get the number of jobs

    # Sort jobs based on profit in descending order
    arr.sort(key=lambda x: x[2], reverse=True)

    result = [False] * t  # To track which time slots are filled
    job = ['-1'] * t  # To store the job names for the time slots

    # Iterate over each job
    for i in range(len(arr)):
        # Find a free slot for this job, starting from the last possible time slot
        for j in range(min(t - 1, arr[i][1] - 1), -1, -1):
            if not result[j]:  # Check if the slot is free
                result[j] = True  # Mark the slot as filled
                job[j] = arr[i][0]  # Assign the job to this slot
                break  # Move to the next job

    print(job)  # Print the scheduled jobs

if __name__ == '__main__':
    n = int(input("Enter the number of jobs: "))  # Input the number of jobs
    arr = []  # List to hold job details

    print("Enter job details in the format: JobName Deadline Profit (space-separated):")
    # Input job details
    for _ in range(n):
        job_detail = input().split()
        job_name = job_detail[0]  # Job name
        deadline = int(job_detail[1])  # Deadline for the job
        profit = int(job_detail[2])  # Profit of the job
        arr.append([job_name, deadline, profit])  # Append job details to the list

    t = int(input("Enter the maximum number of time slots: "))  # Input the maximum number of time slots
    print("Following is the maximum profit sequence of jobs:")
    printJobScheduling(arr, t)  # Schedule and print jobs
