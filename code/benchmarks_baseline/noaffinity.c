#define _GNU_SOURCE
#include <sched.h>
#include <errno.h>

// Override the glibc/OS sched_setaffinity
int sched_setaffinity(pid_t pid, size_t cpusetsize, const cpu_set_t *mask) {
    return 0;
}

