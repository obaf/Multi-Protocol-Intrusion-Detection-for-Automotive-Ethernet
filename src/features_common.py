"""Unified per-packet feature extraction for multi-protocol automotive Ethernet IDS.

Extracts a protocol-agnostic feature vector from any Ethernet frame (raw pcap
bytes or the 58-byte SOME/IP dataset rows), so that detectors can be trained
across protocol families (AVTP, gPTP, UDP/CAN-over-UDP, SOME/IP) and evaluated
in cross-protocol settings.

Feature groups:
  1. Parsed header fields (ethertype one-hot, VLAN, IP, UDP, gPTP, AVTP, SOME/IP)
  2. Length + byte statistics of the first bytes of the frame
  3. Order-based rolling context (no timestamps: SOME/IP rows carry none),
     e.g. distinct source MACs in the last 100 frames (MAC-flooding signal)
"""
import numpy as np

N_RAW_BYTES = 46          # raw leading bytes used for the byte view
FEATURE_DIM = None        # set at build time

ETHERTYPE_ONEHOT = [0x0800, 0x8100, 0x88F7, 0x22EA, 0x22F0, 0x0806, 0x86DD]
GPTP_TYPES = {0x00: 'sync', 0x02: 'follow_up', 0x08: 'follow_up', 0x0B: 'announce'}


def _entropy(b):
    if len(b) == 0:
        return 0.0
    counts = np.bincount(np.frombuffer(bytes(b), dtype=np.uint8), minlength=256)
    p = counts[counts > 0] / len(b)
    return float(-(p * np.log2(p)).sum())


def extract_static_features(frame: bytes) -> np.ndarray:
    """Static (per-packet) features from one Ethernet frame. Returns float32 vector."""
    f = []
    ln = len(frame)
    f.append(float(ln))

    # ---- Ethernet ----
    ethertype = int.from_bytes(frame[12:14], 'big') if ln >= 14 else 0
    et_oh = [1.0 if ethertype == t else 0.0 for t in ETHERTYPE_ONEHOT]
    f += et_oh
    f.append(1.0 if ethertype not in ETHERTYPE_ONEHOT else 0.0)   # other ethertype

    # VLAN handling: 0x8100 -> inner ethertype at 16:18, vlan id at 14:16
    vlan = (ethertype == 0x8100)
    f.append(1.0 if vlan else 0.0)
    inner_et = ethertype
    if vlan and ln >= 18:
        tci = int.from_bytes(frame[14:16], 'big')
        f.append(float(tci & 0x0FFF))          # vlan id
        f.append(float((tci >> 13) & 0x07))    # pcp
        inner_et = int.from_bytes(frame[16:18], 'big')
    else:
        f += [0.0, 0.0]
    f += [1.0 if inner_et == t else 0.0 for t in (0x0800, 0x88F7, 0x22EA, 0x22F0)]  # inner one-hot

    # MAC-based: locally-administered / multicast flags of src+dst
    if ln >= 12:
        dst0, src0 = frame[0], frame[6]
        f += [1.0 if dst0 & 1 else 0.0, 1.0 if dst0 & 2 else 0.0,
              1.0 if src0 & 2 else 0.0, 1.0 if src0 & 1 else 0.0]
    else:
        f += [0.0] * 4

    off = 18 if vlan else 14     # start of L3/L2 payload

    # ---- IPv4 ----
    ip_proto, ttl, ip_flags, ip_len, ip_df, ip_mf, src_last, dst_last = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    udp_sport, udp_dport, udp_len = 0.0, 0.0, 0.0
    someip_mtype, someip_rc, someip_len, someip_ver, someip_iver = 0.0, 0.0, 0.0, 0.0, 0.0
    someip_sid, someip_mid = 0.0, 0.0
    if inner_et == 0x0800 and ln >= off + 20:
        ip = frame[off:]
        ihl = (ip[0] & 0x0F) * 4
        ip_len = float(int.from_bytes(ip[2:4], 'big'))
        ttl = float(ip[8])
        ip_proto = float(ip[9])
        fo = int.from_bytes(ip[6:8], 'big')
        ip_df = 1.0 if fo & 0x4000 else 0.0
        ip_mf = 1.0 if fo & 0x2000 else 0.0
        src_last = float(ip[12 + ihl - 4]) if ihl >= 16 else 0.0
        src_last = float(ip[16]) if len(ip) >= 20 else 0.0
        dst_last = float(ip[19]) if len(ip) >= 20 else 0.0
        f += [ip_len, ttl, ip_proto, ip_df, ip_mf, src_last, dst_last]
        if ip_proto == 17 and len(ip) >= ihl + 8:
            udp = ip[ihl:]
            udp_sport = float(int.from_bytes(udp[0:2], 'big'))
            udp_dport = float(int.from_bytes(udp[2:4], 'big'))
            udp_len = float(int.from_bytes(udp[4:6], 'big'))
            f += [udp_sport, udp_dport, udp_len]
            # ---- SOME/IP header (AUTOSAR FO-18) ----
            pl = udp[8:]
            if len(pl) >= 16:
                someip_len = float(int.from_bytes(pl[0:4], 'big'))
                someip_ver = float(pl[4])
                someip_iver = float(pl[5])
                someip_mtype = float(pl[6])
                someip_rc = float(pl[7])
                someip_sid = float(int.from_bytes(pl[8:10], 'big'))
                someip_mid = float(int.from_bytes(pl[10:12], 'big'))
            f += [someip_len, someip_ver, someip_iver, someip_mtype,
                  someip_rc, someip_sid, someip_mid]
        else:
            f += [0.0] * 3 + [0.0] * 7
    else:
        f += [0.0] * 17          # ip(7) + udp(3) + someip(7) padding

    # ---- gPTP (IEEE 802.1AS, ethertype 0x88F7) ----
    gptp_mtype, gptp_seqid, gptp_log = 0.0, 0.0, 0.0
    gptp_corr = 0.0
    if inner_et == 0x88F7 and ln >= off + 20:
        p = frame[off:]
        gptp_mtype = float(p[0])
        gptp_seqid = float(int.from_bytes(p[30:32], 'big')) if len(p) >= 32 else 0.0
        gptp_log = float(p[6]) if len(p) > 6 else 0.0
        gptp_corr = float(int.from_bytes(p[8:16], 'big')) if len(p) >= 16 else 0.0
    f += [gptp_mtype, gptp_seqid, gptp_log, min(gptp_corr, 1e12) / 1e9]

    # ---- AVTP (ethertype 0x22EA / 0x22F0, incl. inside VLAN) ----
    avtp_subtype, avtp_ver, avtp_len = 0.0, 0.0, 0.0
    if inner_et in (0x22EA, 0x22F0) and ln >= off + 4:
        a = frame[off:]
        avtp_subtype = float((a[0] >> 4) & 0x0F) if a else 0.0
        avtp_ver = float(a[1] & 0x0F) if len(a) > 1 else 0.0
        avtp_len = float(int.from_bytes(a[2:4], 'big')) if len(a) >= 4 else 0.0
    f += [avtp_subtype, avtp_ver, avtp_len]

    # ---- payload byte statistics (L4+ payload or post-L2 bytes) ----
    if inner_et == 0x0800 and ip_proto == 17 and ln >= off + 20 + 8:
        payload = frame[off + 20 + 8:]          # crude UDP payload (no IP options in this data)
    elif inner_et == 0x0800:
        payload = frame[off + 20:]
    else:
        payload = frame[off:]
    pl_short = payload[:64]
    if len(pl_short) > 0:
        arr = np.frombuffer(bytes(pl_short), dtype=np.uint8).astype(np.float32)
        hist, _ = np.histogram(arr, bins=16, range=(0, 256))
        hist = hist / max(1, len(arr))
        f += [float(len(payload)), float(arr.mean()), float(arr.std()),
              float(arr.min()), float(arr.max()), float((arr == 0).sum()),
              _entropy(bytes(pl_short))]
        f += hist.tolist()
    else:
        f += [0.0] * 7 + [0.0] * 16

    # ---- raw leading bytes (normalized) ----
    raw = np.frombuffer(bytes(frame[:N_RAW_BYTES]).ljust(N_RAW_BYTES, b'\x00'),
                        dtype=np.uint8).astype(np.float32) / 255.0
    f += raw.tolist()

    return np.asarray(f, dtype=np.float32)


class StreamContext:
    """Order-based rolling context features (timestamp-free).

    Provides: position index, same-ethertype recent count, distinct source MACs
    in a sliding window of 100 frames, and same-source-MAC recent count.
    These generalise across datasets where timestamps are unavailable.
    """

    def __init__(self, window=100):
        self.window = window
        self.pos = 0
        self.et_counter = {}           # ethertype -> deque-free count approximation (cumulative)
        self.mac_hist = []             # list of src macs (bounded ring)
        self.et_hist = []              # ethertype history (bounded ring)

    def update(self, frame: bytes, ethertype: int):
        ln = len(frame)
        src = frame[6:12] if ln >= 12 else b''
        self.pos += 1
        self.et_counter[ethertype] = self.et_counter.get(ethertype, 0) + 1
        self.mac_hist.append(src)
        self.et_hist.append(ethertype)
        if len(self.mac_hist) > self.window:
            self.mac_hist.pop(0)
            self.et_hist.pop(0)

    def features(self) -> np.ndarray:
        recent = self.mac_hist
        n_recent = len(recent)
        n_distinct = float(len(set(recent)))
        # same-source repetition: max count of any one src mac in the window
        if n_recent:
            from collections import Counter
            c = Counter(recent)
            top = c.most_common(1)[0][1]
        else:
            top = 0
        et_last = self.et_hist[-1] if self.et_hist else 0
        et_frac = 0.0
        if n_recent:
            et_frac = sum(1 for e in self.et_hist if e == et_last) / n_recent
        return np.asarray([
            np.log1p(float(self.pos)),
            np.log1p(float(self.et_counter.get(et_last, 0))),
            n_distinct,
            float(top),
            et_frac,
            float(n_recent),
        ], dtype=np.float32)


CONTEXT_DIM = 6


def frame_ethertype(frame: bytes) -> int:
    ln = len(frame)
    if ln < 14:
        return 0
    et = int.from_bytes(frame[12:14], 'big')
    if et == 0x8100 and ln >= 18:
        et = int.from_bytes(frame[16:18], 'big')
    return et
