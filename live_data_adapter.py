from __future__ import annotations

import socket
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import psutil


COMMON_SERVICES = {
    21: "ftp",
    22: "ssh",
    23: "telnet",
    25: "smtp",
    53: "dns",
    80: "http",
    110: "pop3",
    143: "imap",
    443: "https",
    3306: "mysql",
    5432: "postgres",
    8080: "http",
}


def _proto_from_socktype(sock_type: int | None) -> str:
    if sock_type == socket.SOCK_STREAM:
        return "tcp"
    if sock_type == socket.SOCK_DGRAM:
        return "udp"
    return "tcp"


def _service_from_port(port: int | None) -> str:
    if port is None:
        return "-"
    return COMMON_SERVICES.get(port, "-")


def _state_from_status(status: str | None) -> str:
    s = (status or "").upper()
    if "ESTABLISHED" in s:
        return "CON"
    if "SYN" in s:
        return "SYN"
    if "TIME_WAIT" in s or "CLOSE" in s:
        return "FIN"
    if "LISTEN" in s:
        return "INT"
    return "CON"


def _baseline_row(schema_cols: List[str], idx: int) -> Dict[str, object]:
    row: Dict[str, object] = {}
    for col in schema_cols:
        row[col] = 0

    row.update(
        {
            "id": idx,
            "dur": 0.1,
            "proto": "tcp",
            "service": "-",
            "state": "CON",
            "spkts": 2,
            "dpkts": 2,
            "sbytes": 200,
            "dbytes": 200,
            "rate": 10.0,
            "sttl": 64,
            "dttl": 64,
            "sload": 1000.0,
            "dload": 1000.0,
            "sinpkt": 10.0,
            "dinpkt": 10.0,
            "sjit": 0.0,
            "djit": 0.0,
            "swin": 255,
            "dwin": 255,
            "smean": 100,
            "dmean": 100,
            "ct_srv_src": 1,
            "ct_state_ttl": 0,
            "ct_dst_ltm": 1,
            "ct_src_dport_ltm": 1,
            "ct_dst_sport_ltm": 1,
            "ct_dst_src_ltm": 1,
            "is_ftp_login": 0,
            "ct_ftp_cmd": 0,
            "ct_flw_http_mthd": 0,
            "ct_src_ltm": 1,
            "ct_srv_dst": 1,
            "is_sm_ips_ports": 0,
            "attack_cat": "Normal",
            "label": 0,
        }
    )
    return row


def build_live_dataframe(
    schema_path: str,
    window_rows: int = 300,
) -> pd.DataFrame:
    """Builds flow-like rows from local live network metadata.

    This is a lightweight adapter for demonstration; it uses connection snapshots,
    not deep packet inspection.
    """
    schema_df = pd.read_csv(schema_path, nrows=1)
    cols = schema_df.columns.tolist()

    io_before = psutil.net_io_counters()
    time.sleep(0.15)
    io_after = psutil.net_io_counters()

    delta_sent = max(1, int(io_after.bytes_sent - io_before.bytes_sent))
    delta_recv = max(1, int(io_after.bytes_recv - io_before.bytes_recv))

    conns = psutil.net_connections(kind="inet")
    if not conns:
        conns = [None] * min(window_rows, 20)

    rows: List[Dict[str, object]] = []
    rng = np.random.default_rng(42)

    for i in range(window_rows):
        conn = conns[i % len(conns)]
        row = _baseline_row(cols, i + 1)

        if conn is not None:
            laddr_port = conn.laddr.port if conn.laddr else None
            raddr_port = conn.raddr.port if conn.raddr else None
            row["proto"] = _proto_from_socktype(conn.type)
            row["service"] = _service_from_port(raddr_port or laddr_port)
            row["state"] = _state_from_status(conn.status)

        jitter = float(rng.uniform(0.9, 1.1))
        row["spkts"] = int(max(1, delta_sent // 200 * jitter))
        row["dpkts"] = int(max(1, delta_recv // 200 * jitter))
        row["sbytes"] = int(max(100, delta_sent * jitter))
        row["dbytes"] = int(max(100, delta_recv * jitter))
        row["rate"] = float((row["spkts"] + row["dpkts"]) / 0.15)
        row["sload"] = float(row["sbytes"] / 0.15)
        row["dload"] = float(row["dbytes"] / 0.15)
        row["sinpkt"] = float(rng.uniform(2.0, 20.0))
        row["dinpkt"] = float(rng.uniform(2.0, 20.0))

        # Create small synthetic anomaly labels for online demo training stability.
        activity = row["rate"] + (row["sload"] + row["dload"]) / 5000.0
        if activity > 1200 or rng.uniform() > 0.96:
            row["label"] = 1
            row["attack_cat"] = "SyntheticAnomaly"
        else:
            row["label"] = 0
            row["attack_cat"] = "Normal"

        rows.append(row)

    live_df = pd.DataFrame(rows)
    for col in cols:
        if col not in live_df.columns:
            live_df[col] = 0

    return live_df[cols]
