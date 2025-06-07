import matplotlib
import matplotlib.pyplot as plt
import numpy as np

threads = [1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32]
times = [42.62, 29.16, 18.35, 14.73, 11.78, 10.20, 8.83, 8.20, 7.82, 6.17, 4.26, 3.54]

# Compute speedup
speedup = times[0] / np.array(times)

# Plot time vs thread
plt.figure()
plt.plot(threads, times, marker='o')
plt.xlabel('Number of Threads')
plt.ylabel('Training Time (s)')
plt.title('Strong Scaling: Training Time vs Threads')
plt.xscale('log', base=2)
plt.xticks(threads, threads)
plt.grid(True)
plt.tight_layout()
plt.savefig('MNIST_time_threads.png')
plt.clf()

# Plot speedup vs thread
plt.figure()
plt.plot(threads, speedup, marker='o')
plt.xlabel('Number of Threads')
plt.ylabel('Speedup (T1 / TN)')
plt.title('Strong Scaling: Speedup vs Threads')
plt.xscale('log', base=2)
plt.xticks(threads, threads)
plt.grid(True)
plt.tight_layout()
plt.savefig('MNIST_speedup_threads.png')
plt.clf()