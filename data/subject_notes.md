# Subject: Operating Systems (B.Tech CSE Core)

## Topic 1: Process vs Thread
A Process is an independent program in execution that has its own isolated virtual address space, code segment, data segment, and operating system resources such as open files. Because processes are isolated, switching between two processes (Context Switching) is heavy and slow because the CPU cache and Memory Management Unit (MMU) page tables must be flushed.
A Thread, often called a lightweight process, is the smallest unit of CPU execution within a process. Multiple threads inside the same process share the same code, data, and heap memory segments, while each thread keeps its own private stack and program counter (registers). Sharing memory makes inter-thread communication fast and context switching much cheaper than processes.

## Topic 2: Deadlock and the Four Coffman Conditions
A Deadlock is a permanent blocking state in an Operating System where a set of processes are stalled because each process holds at least one resource and waits to acquire another resource currently held by another process in the same set.
A deadlock can occur if and only if all four Coffman conditions hold simultaneously:
1. Mutual Exclusion: At least one resource must be held in a non-shareable mode (only one process can use it at a time).
2. Hold and Wait: A process must be currently holding at least one resource while requesting additional resources held by other processes.
3. No Preemption: Resources cannot be forcibly taken away from a process; they can only be released voluntarily by the process holding them after it finishes its task.
4. Circular Wait: A closed chain of processes exists such that Process P0 waits for a resource held by P1, P1 waits for P2, and Pn waits for a resource held by P0.

## Topic 3: Virtual Memory, Paging, and Page Faults
Virtual Memory is a memory management technique that gives each process the illusion of having a large, contiguous private memory space, even when physical RAM is smaller. The OS divides virtual memory into fixed-size blocks called Pages (typically 4 KB) and physical RAM into fixed-size blocks called Frames.
A Page Table maps virtual page numbers to physical frame numbers. When a CPU requests a virtual address whose page is not currently loaded in physical RAM, the hardware triggers a Page Fault trap. The OS pauses the process, locates the required page on the secondary disk (swap space), loads it into a free RAM frame, updates the Page Table, and restarts the instruction. If the system spends more time swapping pages between disk and RAM than executing instructions, it enters a severe performance collapse called Thrashing.

## Topic 4: CPU Scheduling Algorithms (FCFS, SJF, and Round Robin)
CPU Scheduling determines which process in the Ready Queue gets allocated to the CPU next.
1. First-Come, First-Served (FCFS): A non-preemptive queue where the first process to arrive executes until completion. FCFS suffers from the Convoy Effect, where one long CPU-bound process forces many short I/O-bound processes to wait, increasing average waiting time.
2. Shortest Job First (SJF): Selects the process with the smallest next CPU burst time. While SJF is mathematically optimal for minimizing average waiting time, it can cause Starvation for long processes if short processes keep arriving continuously.
3. Round Robin (RR): Designed for time-sharing interactive systems. Each process gets a fixed small unit of CPU time called a Time Quantum (typically 10 to 100 milliseconds). If the process does not finish within the time quantum, the OS preempts it and moves it to the back of the Ready Queue.