#!/usr/bin/env python
# coding: utf-8

import subprocess
from multiprocessing import Pool, cpu_count

# Define the ranges for mob_of_the_pandemic and number_of_the_pandemic
mobility_range = range(4)  # 0 to 3
number_range = range(81)  # 0 to 80

# Generate all combinations of mob_of_the_pandemic and number_of_the_pandemic
tasks = [(mob, num) for mob in mobility_range for num in number_range]

# Define a function to run your script
def run_task(task):
    mob, num = task
    command = [
        "python",
        "global_comparison.py",  # Replace with your script's filename
        str(mob),
        str(num),
    ]
    print(f"Running for mob_of_the_pandemic={mob}, number_of_the_pandemic={num}")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Success: mob={mob}, num={num}")
        return f"Success: mob={mob}, num={num}, output: {result.stdout}"
    else:
        print(f"Error: mob={mob}, num={num}, error: {result.stderr}")
        return f"Error: mob={mob}, num={num}, error: {result.stderr}"

# Run tasks in parallel
if __name__ == "__main__":
    # Determine the number of processes (use all available cores)
    num_processes = min(cpu_count(), len(tasks))  # Use at most the number of tasks
    print(f"Running with {num_processes} parallel processes...")

    with Pool(num_processes) as pool:
        results = pool.map(run_task, tasks)

    # Optionally, save results to a file
    with open("parallel_results.txt", "w") as f:
        for result in results:
            f.write(result + "\n")
