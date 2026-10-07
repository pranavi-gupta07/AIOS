import ctypes
import errno
try:
    import fcntl
except ImportError:
    fcntl = None
import os

DEFAULT_DEVICE = "/dev/aios"
MAGIC = ord("A")
IOC_WRITE = 1
IOC_READ = 2


def _ioc(direction, nr, size):
    return (direction << 30) | (size << 16) | (MAGIC << 8) | nr


class Submit(ctypes.Structure):
    _fields_ = [("buf", ctypes.c_uint64), ("len", ctypes.c_uint32), ("flags", ctypes.c_uint32), ("id", ctypes.c_uint64)]


class Wait(ctypes.Structure):
    _fields_ = [("id", ctypes.c_uint64), ("buf", ctypes.c_uint64), ("cap", ctypes.c_uint32), ("len", ctypes.c_uint32),
                ("status", ctypes.c_int32), ("timeout_ms", ctypes.c_uint32)]


class Fetch(ctypes.Structure):
    _fields_ = [("id", ctypes.c_uint64), ("buf", ctypes.c_uint64), ("cap", ctypes.c_uint32), ("len", ctypes.c_uint32),
                ("pid", ctypes.c_int32), ("uid", ctypes.c_uint32), ("timeout_ms", ctypes.c_uint32),
                ("reserved", ctypes.c_uint32), ("comm", ctypes.c_char * 16)]


class Complete(ctypes.Structure):
    _fields_ = [("id", ctypes.c_uint64), ("buf", ctypes.c_uint64), ("len", ctypes.c_uint32), ("status", ctypes.c_int32)]


class Stats(ctypes.Structure):
    _fields_ = [("submitted", ctypes.c_uint64), ("completed", ctypes.c_uint64), ("failed", ctypes.c_uint64),
                ("pending", ctypes.c_uint64), ("inflight", ctypes.c_uint64), ("bytes_in", ctypes.c_uint64),
                ("bytes_out", ctypes.c_uint64), ("daemon_pid", ctypes.c_int32), ("reserved", ctypes.c_uint32)]


IOC_SUBMIT = _ioc(IOC_READ | IOC_WRITE, 1, ctypes.sizeof(Submit))
IOC_WAIT = _ioc(IOC_READ | IOC_WRITE, 2, ctypes.sizeof(Wait))
IOC_REGISTER = _ioc(0, 3, 0)
IOC_FETCH = _ioc(IOC_READ | IOC_WRITE, 4, ctypes.sizeof(Fetch))
IOC_COMPLETE = _ioc(IOC_WRITE, 5, ctypes.sizeof(Complete))
IOC_STATS = _ioc(IOC_READ, 6, ctypes.sizeof(Stats))


def available(path=DEFAULT_DEVICE):
    if fcntl is None:
        return False
    return os.path.exists(path) and os.access(path, os.R_OK | os.W_OK)


class KernelChannel:
    def __init__(self, path=DEFAULT_DEVICE):
        self.path = path
        self.fd = os.open(path, os.O_RDWR | os.O_CLOEXEC)
        self._cap = 1 << 20

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def stats(self):
        s = Stats()
        fcntl.ioctl(self.fd, IOC_STATS, s, True)
        return {name: getattr(s, name) for name, _ in Stats._fields_ if name != "reserved"}

    def submit(self, blob):
        buf = ctypes.create_string_buffer(blob, len(blob))
        s = Submit(ctypes.addressof(buf), len(blob), 0, 0)
        fcntl.ioctl(self.fd, IOC_SUBMIT, s, True)
        return s.id

    def wait(self, req_id, timeout_ms=0):
        cap = self._cap
        while True:
            buf = ctypes.create_string_buffer(cap)
            w = Wait(req_id, ctypes.addressof(buf), cap, 0, 0, timeout_ms)
            try:
                fcntl.ioctl(self.fd, IOC_WAIT, w, True)
            except OSError as exc:
                if exc.errno == errno.ENOSPC:
                    cap = max(cap * 2, w.len)
                    self._cap = cap
                    continue
                raise
            return w.status, buf.raw[:w.len]

    def call(self, blob, timeout_ms=0):
        return self.wait(self.submit(blob), timeout_ms)

    def register(self):
        fcntl.ioctl(self.fd, IOC_REGISTER, 0)

    def fetch(self, timeout_ms=500):
        cap = self._cap
        while True:
            buf = ctypes.create_string_buffer(cap)
            f = Fetch(0, ctypes.addressof(buf), cap, 0, 0, 0, timeout_ms, 0, b"")
            try:
                fcntl.ioctl(self.fd, IOC_FETCH, f, True)
            except OSError as exc:
                if exc.errno == errno.EAGAIN:
                    return None
                if exc.errno == errno.ENOSPC:
                    cap = max(cap * 2, f.len)
                    self._cap = cap
                    continue
                raise
            return {"id": f.id, "blob": buf.raw[:f.len], "pid": f.pid, "uid": f.uid,
                    "comm": f.comm.decode(errors="replace")}

    def complete(self, req_id, blob, status=0):
        buf = ctypes.create_string_buffer(blob, len(blob))
        c = Complete(req_id, ctypes.addressof(buf), len(blob), status)
        fcntl.ioctl(self.fd, IOC_COMPLETE, c, True)
