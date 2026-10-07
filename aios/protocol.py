import json
import socket
import struct

import numpy as np

PREFIX = struct.Struct(">II")


class ProtocolError(Exception):
    pass


def encode(header, arrays=None):
    arrays = arrays or {}
    meta = []
    chunks = []
    offset = 0
    for name, arr in arrays.items():
        arr = np.ascontiguousarray(arr)
        data = arr.tobytes()
        meta.append({"name": name, "dtype": arr.dtype.str, "shape": list(arr.shape), "offset": offset, "nbytes": len(data)})
        chunks.append(data)
        offset += len(data)
    body = dict(header)
    body["arrays"] = meta
    head = json.dumps(body, separators=(",", ":")).encode()
    return PREFIX.pack(len(head), offset) + head + b"".join(chunks)


def decode(blob):
    if len(blob) < PREFIX.size:
        raise ProtocolError("short message")
    hlen, plen = PREFIX.unpack_from(blob, 0)
    if len(blob) < PREFIX.size + hlen + plen:
        raise ProtocolError("truncated message")
    header = json.loads(blob[PREFIX.size:PREFIX.size + hlen])
    payload = memoryview(blob)[PREFIX.size + hlen:PREFIX.size + hlen + plen]
    return header, _unpack_arrays(header, payload)


def _unpack_arrays(header, payload):
    arrays = {}
    for m in header.pop("arrays", []):
        raw = payload[m["offset"]:m["offset"] + m["nbytes"]]
        arrays[m["name"]] = np.frombuffer(raw, dtype=np.dtype(m["dtype"])).reshape(m["shape"]).copy()
    return arrays


def recv_exact(sock, n):
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("peer closed connection")
        got += k
    return bytes(buf)


def send(sock, header, arrays=None):
    blob = encode(header, arrays)
    sock.sendall(blob)
    return len(blob)


def recv(sock):
    prefix = recv_exact(sock, PREFIX.size)
    hlen, plen = PREFIX.unpack(prefix)
    rest = recv_exact(sock, hlen + plen)
    header, arrays = decode(prefix + rest)
    header["_wire_bytes"] = PREFIX.size + hlen + plen
    return header, arrays


def message_size(header, arrays=None):
    return len(encode(header, arrays))


def connect_tcp(host, port, timeout=5.0):
    s = socket.create_connection((host, port), timeout=timeout)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    s.settimeout(None)
    return s
