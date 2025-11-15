# accelerate_test.py
from accelerate import Accelerator
import os
print("MADE IT HERE")
# Force unbuffered printing
import sys; sys.stdout.reconfigure(line_buffering=True)  

acc = Accelerator()
print(acc.num_processes )
print(
    f"[host {os.uname().nodename}] "
    f"global rank {acc.process_index}/{acc.num_processes}, "
    f"local rank {acc.local_process_index}"
)

# Print out assigned CUDA device
print(f"Device: {acc.device}")
