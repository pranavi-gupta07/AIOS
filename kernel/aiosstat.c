#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include "aios_uapi.h"

int main(int argc, char **argv)
{
	const char *path = argc > 1 ? argv[1] : "/dev/aios";
	struct aios_stats s;
	int fd = open(path, O_RDWR | O_CLOEXEC);

	if (fd < 0) {
		fprintf(stderr, "aiosstat: open %s: %s\n", path, strerror(errno));
		return 1;
	}
	memset(&s, 0, sizeof(s));
	if (ioctl(fd, AIOS_IOC_STATS, &s) < 0) {
		fprintf(stderr, "aiosstat: AIOS_IOC_STATS: %s\n", strerror(errno));
		close(fd);
		return 1;
	}
	printf("daemon_pid %d\n", s.daemon_pid);
	printf("submitted  %llu\n", (unsigned long long)s.submitted);
	printf("completed  %llu\n", (unsigned long long)s.completed);
	printf("failed     %llu\n", (unsigned long long)s.failed);
	printf("pending    %llu\n", (unsigned long long)s.pending);
	printf("inflight   %llu\n", (unsigned long long)s.inflight);
	printf("bytes_in   %llu\n", (unsigned long long)s.bytes_in);
	printf("bytes_out  %llu\n", (unsigned long long)s.bytes_out);
	close(fd);
	return 0;
}
