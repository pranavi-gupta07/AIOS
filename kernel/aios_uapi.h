#ifndef _UAPI_AIOS_H
#define _UAPI_AIOS_H

#ifdef __KERNEL__
#include <linux/types.h>
#include <linux/ioctl.h>
#else
#include <stdint.h>
#include <sys/ioctl.h>
typedef uint64_t __u64;
typedef uint32_t __u32;
typedef int32_t __s32;
#endif

#define AIOS_IOC_MAGIC 'A'
#define AIOS_COMM_LEN 16

struct aios_submit {
	__u64 buf;
	__u32 len;
	__u32 flags;
	__u64 id;
};

struct aios_wait {
	__u64 id;
	__u64 buf;
	__u32 cap;
	__u32 len;
	__s32 status;
	__u32 timeout_ms;
};

struct aios_fetch {
	__u64 id;
	__u64 buf;
	__u32 cap;
	__u32 len;
	__s32 pid;
	__u32 uid;
	__u32 timeout_ms;
	__u32 reserved;
	char comm[AIOS_COMM_LEN];
};

struct aios_complete {
	__u64 id;
	__u64 buf;
	__u32 len;
	__s32 status;
};

struct aios_stats {
	__u64 submitted;
	__u64 completed;
	__u64 failed;
	__u64 pending;
	__u64 inflight;
	__u64 bytes_in;
	__u64 bytes_out;
	__s32 daemon_pid;
	__u32 reserved;
};

#define AIOS_IOC_SUBMIT   _IOWR(AIOS_IOC_MAGIC, 1, struct aios_submit)
#define AIOS_IOC_WAIT     _IOWR(AIOS_IOC_MAGIC, 2, struct aios_wait)
#define AIOS_IOC_REGISTER _IO(AIOS_IOC_MAGIC, 3)
#define AIOS_IOC_FETCH    _IOWR(AIOS_IOC_MAGIC, 4, struct aios_fetch)
#define AIOS_IOC_COMPLETE _IOW(AIOS_IOC_MAGIC, 5, struct aios_complete)
#define AIOS_IOC_STATS    _IOR(AIOS_IOC_MAGIC, 6, struct aios_stats)

#endif
