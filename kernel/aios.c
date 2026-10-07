#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/miscdevice.h>
#include <linux/fs.h>
#include <linux/slab.h>
#include <linux/mm.h>
#include <linux/uaccess.h>
#include <linux/wait.h>
#include <linux/mutex.h>
#include <linux/list.h>
#include <linux/sched.h>
#include <linux/cred.h>
#include <linux/uidgid.h>
#include <linux/capability.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>
#include <linux/jiffies.h>
#include <linux/ktime.h>
#include <linux/string.h>
#include "aios_uapi.h"

static unsigned int max_msg_mb = 64;
module_param(max_msg_mb, uint, 0644);
MODULE_PARM_DESC(max_msg_mb, "largest request or response accepted, in MiB");

static int daemon_uid = -1;
module_param(daemon_uid, int, 0644);
MODULE_PARM_DESC(daemon_uid, "uid allowed to register as the placement daemon, -1 for any");

enum aios_state {
	AIOS_PENDING,
	AIOS_FETCHED,
	AIOS_DONE,
	AIOS_ABANDONED,
};

static const char *const aios_state_names[] = { "pending", "fetched", "done", "abandoned" };

struct aios_req {
	struct list_head node;
	struct list_head queue;
	u64 id;
	enum aios_state state;
	struct file *owner;
	void *req_buf;
	u32 req_len;
	void *resp_buf;
	u32 resp_len;
	s32 status;
	pid_t pid;
	uid_t uid;
	char comm[TASK_COMM_LEN];
	ktime_t submitted;
	bool waiting;
	wait_queue_head_t wq;
};

static DEFINE_MUTEX(aios_lock);
static LIST_HEAD(aios_all);
static LIST_HEAD(aios_pending);
static DECLARE_WAIT_QUEUE_HEAD(aios_daemon_wq);
static struct file *aios_daemon;
static pid_t aios_daemon_pid;
static u64 aios_next_id = 1;
static struct aios_stats aios_st;

static size_t aios_max_msg(void)
{
	return (size_t)READ_ONCE(max_msg_mb) << 20;
}

static struct aios_req *aios_find(u64 id)
{
	struct aios_req *r;

	list_for_each_entry(r, &aios_all, node)
		if (r->id == id)
			return r;
	return NULL;
}

static void aios_free_req(struct aios_req *r)
{
	list_del(&r->node);
	if (!list_empty(&r->queue))
		list_del_init(&r->queue);
	kvfree(r->req_buf);
	kvfree(r->resp_buf);
	kfree(r);
}

static long aios_timeout(u32 ms)
{
	return ms ? (long)msecs_to_jiffies(ms) : MAX_SCHEDULE_TIMEOUT;
}

static long aios_submit(struct file *f, void __user *arg)
{
	struct aios_submit s;
	struct aios_req *r;
	void *buf;

	if (copy_from_user(&s, arg, sizeof(s)))
		return -EFAULT;
	if (!s.len || s.len > aios_max_msg())
		return -EMSGSIZE;
	buf = kvmalloc(s.len, GFP_KERNEL);
	if (!buf)
		return -ENOMEM;
	if (copy_from_user(buf, u64_to_user_ptr(s.buf), s.len)) {
		kvfree(buf);
		return -EFAULT;
	}
	r = kzalloc(sizeof(*r), GFP_KERNEL);
	if (!r) {
		kvfree(buf);
		return -ENOMEM;
	}
	init_waitqueue_head(&r->wq);
	INIT_LIST_HEAD(&r->queue);
	r->req_buf = buf;
	r->req_len = s.len;
	r->owner = f;
	r->pid = task_tgid_vnr(current);
	r->uid = from_kuid_munged(current_user_ns(), current_uid());
	get_task_comm(r->comm, current);
	r->submitted = ktime_get();

	mutex_lock(&aios_lock);
	if (!aios_daemon) {
		mutex_unlock(&aios_lock);
		kvfree(buf);
		kfree(r);
		return -EHOSTDOWN;
	}
	r->id = aios_next_id++;
	r->state = AIOS_PENDING;
	list_add_tail(&r->node, &aios_all);
	list_add_tail(&r->queue, &aios_pending);
	aios_st.submitted++;
	aios_st.bytes_in += s.len;
	WRITE_ONCE(aios_st.pending, aios_st.pending + 1);
	s.id = r->id;
	wake_up_interruptible(&aios_daemon_wq);
	mutex_unlock(&aios_lock);

	if (copy_to_user(arg, &s, sizeof(s)))
		return -EFAULT;
	return 0;
}

static long aios_wait(struct file *f, void __user *arg)
{
	struct aios_wait w;
	struct aios_req *r;
	long ret;

	if (copy_from_user(&w, arg, sizeof(w)))
		return -EFAULT;

	mutex_lock(&aios_lock);
	r = aios_find(w.id);
	if (!r || r->owner != f) {
		mutex_unlock(&aios_lock);
		return -ENOENT;
	}
	if (r->waiting) {
		mutex_unlock(&aios_lock);
		return -EBUSY;
	}
	r->waiting = true;
	mutex_unlock(&aios_lock);

	ret = wait_event_interruptible_timeout(r->wq, READ_ONCE(r->state) == AIOS_DONE, aios_timeout(w.timeout_ms));

	mutex_lock(&aios_lock);
	r->waiting = false;
	if (ret == 0) {
		mutex_unlock(&aios_lock);
		return -ETIMEDOUT;
	}
	if (ret < 0) {
		mutex_unlock(&aios_lock);
		return ret;
	}
	w.status = r->status;
	w.len = r->resp_len;
	if (w.cap < r->resp_len) {
		mutex_unlock(&aios_lock);
		if (copy_to_user(arg, &w, sizeof(w)))
			return -EFAULT;
		return -ENOSPC;
	}
	if (r->resp_len && copy_to_user(u64_to_user_ptr(w.buf), r->resp_buf, r->resp_len)) {
		mutex_unlock(&aios_lock);
		return -EFAULT;
	}
	aios_free_req(r);
	mutex_unlock(&aios_lock);

	if (copy_to_user(arg, &w, sizeof(w)))
		return -EFAULT;
	return 0;
}

static long aios_register(struct file *f)
{
	int want = READ_ONCE(daemon_uid);

	if (want >= 0 && from_kuid(&init_user_ns, current_uid()) != (uid_t)want && !capable(CAP_SYS_ADMIN))
		return -EPERM;
	mutex_lock(&aios_lock);
	if (aios_daemon && aios_daemon != f) {
		mutex_unlock(&aios_lock);
		return -EBUSY;
	}
	aios_daemon = f;
	aios_daemon_pid = task_tgid_vnr(current);
	mutex_unlock(&aios_lock);
	pr_info("aios: placement daemon registered (pid %d)\n", aios_daemon_pid);
	return 0;
}

static long aios_fetch(struct file *f, void __user *arg)
{
	struct aios_fetch q;
	struct aios_req *r;
	long ret;

	if (copy_from_user(&q, arg, sizeof(q)))
		return -EFAULT;
	if (f != READ_ONCE(aios_daemon))
		return -EPERM;

	ret = wait_event_interruptible_timeout(aios_daemon_wq, READ_ONCE(aios_st.pending) > 0, aios_timeout(q.timeout_ms));
	if (ret == 0)
		return -EAGAIN;
	if (ret < 0)
		return ret;

	mutex_lock(&aios_lock);
	if (list_empty(&aios_pending)) {
		mutex_unlock(&aios_lock);
		return -EAGAIN;
	}
	r = list_first_entry(&aios_pending, struct aios_req, queue);
	q.id = r->id;
	q.len = r->req_len;
	q.pid = r->pid;
	q.uid = r->uid;
	memset(q.comm, 0, sizeof(q.comm));
	strscpy(q.comm, r->comm, sizeof(q.comm));
	if (q.cap < r->req_len) {
		mutex_unlock(&aios_lock);
		if (copy_to_user(arg, &q, sizeof(q)))
			return -EFAULT;
		return -ENOSPC;
	}
	if (copy_to_user(u64_to_user_ptr(q.buf), r->req_buf, r->req_len)) {
		mutex_unlock(&aios_lock);
		return -EFAULT;
	}
	list_del_init(&r->queue);
	r->state = AIOS_FETCHED;
	kvfree(r->req_buf);
	r->req_buf = NULL;
	WRITE_ONCE(aios_st.pending, aios_st.pending - 1);
	aios_st.inflight++;
	mutex_unlock(&aios_lock);

	if (copy_to_user(arg, &q, sizeof(q)))
		return -EFAULT;
	return 0;
}

static long aios_complete(struct file *f, void __user *arg)
{
	struct aios_complete c;
	struct aios_req *r;
	void *buf = NULL;

	if (copy_from_user(&c, arg, sizeof(c)))
		return -EFAULT;
	if (f != READ_ONCE(aios_daemon))
		return -EPERM;
	if (c.len > aios_max_msg())
		return -EMSGSIZE;
	if (c.len) {
		buf = kvmalloc(c.len, GFP_KERNEL);
		if (!buf)
			return -ENOMEM;
		if (copy_from_user(buf, u64_to_user_ptr(c.buf), c.len)) {
			kvfree(buf);
			return -EFAULT;
		}
	}

	mutex_lock(&aios_lock);
	r = aios_find(c.id);
	if (!r || (r->state != AIOS_FETCHED && r->state != AIOS_ABANDONED)) {
		mutex_unlock(&aios_lock);
		kvfree(buf);
		return -ENOENT;
	}
	aios_st.inflight--;
	if (r->state == AIOS_ABANDONED) {
		aios_free_req(r);
		mutex_unlock(&aios_lock);
		kvfree(buf);
		return 0;
	}
	r->resp_buf = buf;
	r->resp_len = c.len;
	r->status = c.status;
	if (c.status < 0)
		aios_st.failed++;
	else
		aios_st.completed++;
	aios_st.bytes_out += c.len;
	WRITE_ONCE(r->state, AIOS_DONE);
	wake_up_interruptible(&r->wq);
	mutex_unlock(&aios_lock);
	return 0;
}

static long aios_stats_ioctl(void __user *arg)
{
	struct aios_stats s;

	mutex_lock(&aios_lock);
	s = aios_st;
	s.daemon_pid = aios_daemon ? aios_daemon_pid : 0;
	mutex_unlock(&aios_lock);
	if (copy_to_user(arg, &s, sizeof(s)))
		return -EFAULT;
	return 0;
}

static long aios_ioctl(struct file *f, unsigned int cmd, unsigned long arg)
{
	void __user *uarg = (void __user *)arg;

	switch (cmd) {
	case AIOS_IOC_SUBMIT:
		return aios_submit(f, uarg);
	case AIOS_IOC_WAIT:
		return aios_wait(f, uarg);
	case AIOS_IOC_REGISTER:
		return aios_register(f);
	case AIOS_IOC_FETCH:
		return aios_fetch(f, uarg);
	case AIOS_IOC_COMPLETE:
		return aios_complete(f, uarg);
	case AIOS_IOC_STATS:
		return aios_stats_ioctl(uarg);
	default:
		return -ENOTTY;
	}
}

static int aios_open(struct inode *inode, struct file *f)
{
	return nonseekable_open(inode, f);
}

static int aios_release(struct inode *inode, struct file *f)
{
	struct aios_req *r, *tmp;

	mutex_lock(&aios_lock);
	if (f == aios_daemon) {
		aios_daemon = NULL;
		aios_daemon_pid = 0;
		list_for_each_entry_safe(r, tmp, &aios_all, node) {
			if (r->state == AIOS_ABANDONED) {
				aios_st.inflight--;
				aios_free_req(r);
				continue;
			}
			if (r->state == AIOS_FETCHED) {
				aios_st.inflight--;
				r->status = -EPIPE;
			} else if (r->state == AIOS_PENDING) {
				list_del_init(&r->queue);
				WRITE_ONCE(aios_st.pending, aios_st.pending - 1);
				r->status = -EHOSTDOWN;
			} else {
				continue;
			}
			r->resp_len = 0;
			aios_st.failed++;
			WRITE_ONCE(r->state, AIOS_DONE);
			wake_up_interruptible(&r->wq);
		}
		pr_info("aios: placement daemon unregistered\n");
	}
	list_for_each_entry_safe(r, tmp, &aios_all, node) {
		if (r->owner != f)
			continue;
		if (r->state == AIOS_PENDING) {
			WRITE_ONCE(aios_st.pending, aios_st.pending - 1);
			aios_free_req(r);
		} else if (r->state == AIOS_FETCHED) {
			r->state = AIOS_ABANDONED;
			r->owner = NULL;
		} else if (r->state == AIOS_DONE) {
			aios_free_req(r);
		}
	}
	mutex_unlock(&aios_lock);
	return 0;
}

static const struct file_operations aios_fops = {
	.owner = THIS_MODULE,
	.open = aios_open,
	.release = aios_release,
	.unlocked_ioctl = aios_ioctl,
	.compat_ioctl = compat_ptr_ioctl,
};

static struct miscdevice aios_misc = {
	.minor = MISC_DYNAMIC_MINOR,
	.name = "aios",
	.fops = &aios_fops,
	.mode = 0666,
};

static int aios_proc_show(struct seq_file *m, void *v)
{
	struct aios_req *r;
	ktime_t now = ktime_get();

	mutex_lock(&aios_lock);
	seq_printf(m, "daemon_pid %d\n", aios_daemon ? aios_daemon_pid : 0);
	seq_printf(m, "submitted %llu\n", aios_st.submitted);
	seq_printf(m, "completed %llu\n", aios_st.completed);
	seq_printf(m, "failed %llu\n", aios_st.failed);
	seq_printf(m, "pending %llu\n", aios_st.pending);
	seq_printf(m, "inflight %llu\n", aios_st.inflight);
	seq_printf(m, "bytes_in %llu\n", aios_st.bytes_in);
	seq_printf(m, "bytes_out %llu\n", aios_st.bytes_out);
	seq_puts(m, "\nid state pid uid comm age_us bytes\n");
	list_for_each_entry(r, &aios_all, node)
		seq_printf(m, "%llu %s %d %u %s %lld %u\n", r->id, aios_state_names[r->state], r->pid, r->uid,
			   r->comm, ktime_us_delta(now, r->submitted), r->req_len);
	mutex_unlock(&aios_lock);
	return 0;
}

static int __init aios_init(void)
{
	int ret = misc_register(&aios_misc);

	if (ret)
		return ret;
	if (!proc_create_single("aios", 0444, NULL, aios_proc_show)) {
		misc_deregister(&aios_misc);
		return -ENOMEM;
	}
	pr_info("aios: inference request broker ready at /dev/aios\n");
	return 0;
}

static void __exit aios_exit(void)
{
	struct aios_req *r, *tmp;

	remove_proc_entry("aios", NULL);
	misc_deregister(&aios_misc);
	mutex_lock(&aios_lock);
	list_for_each_entry_safe(r, tmp, &aios_all, node)
		aios_free_req(r);
	mutex_unlock(&aios_lock);
}

module_init(aios_init);
module_exit(aios_exit);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("AIOS: OS-level AI inference request broker");
MODULE_VERSION("0.1");
