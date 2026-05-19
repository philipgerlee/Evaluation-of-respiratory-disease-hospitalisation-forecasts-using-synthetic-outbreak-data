#!/usr/bin/env python
# coding: utf-8

import subprocess
import os
import numpy as np
from multiprocessing import Pool, cpu_count

reach=7
# Define the ranges for mob_of_the_pandemic and number_of_the_pandemic
mobility_range = range(4)  # 0 to 3
number_range = range(81)  # 0 to 80

#thinning
#param_range = range(1,32,5)

#shift
#param_range = range(1,32,5)

#mob noise
param_range=np.logspace(-3,-1,7)

#inc noise
#param_range=np.logspace(1,3,7)


# Generate all combinations of mob_of_the_pandemic and number_of_the_pandemic
#tasks = [(mob, num, param) for mob in mobility_range for num in number_range for param in param_range]
outdir = "./results/thinning_evaluation"  # adjust to your directory


# Define a function to run your script
def run_task(task):
    mob, num, param = task
    command = [
        "python",
        #"shift_comparison.py",  # Replace with your script's filename
        "noisemob_comparison.py",
        #"noiseinc_comparison.py",
        #"thinning_comparison.py",
        str(mob),
        str(num),
        str(param),
    ]
    print(f"Running for mob_of_the_pandemic={mob}, number_of_the_pandemic={num}, parameter={param}")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Success: mob={mob}, num={num}")
        return f"Success: mob={mob}, num={num}, output: {result.stdout}"
    else:
        print(f"Error: mob={mob}, num={num}, error: {result.stderr}")
        return f"Error: mob={mob}, num={num}, error: {result.stderr}"

# Run tasks in parallel
if __name__ == "__main__":
    tasks = []
    for mob in mobility_range:
        for num in number_range:
            for param in param_range:   # param = shift
                #fname = f"evaluation_with_WIS_on_pandemic_{mob}_{num}_and_reach_={reach}_shift_={param}.json"
                fname = f"evaluation_with_WIS_on_pandemic_{mob}_{num}_and_reach_={reach}_noisemob_={param}.json"
                #fname = f"evaluation_with_WIS_on_pandemic_{mob}_{num}_and_reach_={reach}_noiseinc_={param}.json"
                #fname = f"evaluation_with_WIS_on_pandemic_{mob}_{num}_and_reach_={reach}_freq_={param}.json"
                fpath = os.path.join(outdir, fname)
                if not os.path.exists(fpath):
                    tasks.append((mob, num, param))

    print(f"{len(tasks)} tasks left to run")
    # Determine the number of processes (use all available cores)
    num_processes = min(cpu_count(), len(tasks))  # Use at most the number of tasks
    print(f"Running with {num_processes} parallel processes...")

    with Pool(num_processes) as pool:
        results = pool.map(run_task, tasks)

    # Optionally, save results to a file
    with open("parallel_results.txt", "w") as f:
        for result in results:
            f.write(result + "\n")
