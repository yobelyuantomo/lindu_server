"""Uji kesetaraan ekstraksi fitur: firmware node vs ``ml/feature_extractor.py``.

Model yang tertanam di node dilatih pada fitur hasil ``feature_extractor.py``.
Kalau firmware menghitung fitur yang sedikit berbeda, model menerima masukan
yang bukan dilatihkan padanya dan hasilnya salah TANPA gejala apa pun —
prediksi tetap keluar, confidence tetap terlihat wajar. Kesetaraan itu tidak
bisa dibuktikan saat kompilasi, jadi dibuktikan di sini, pada perangkat sungguhan.

Cara kerja::

    1. Skrip menyimak telemetri node selama ``--warmup`` detik.
    2. Skrip mengirim perintah ``ml_selftest`` ke node.
    3. Node menerbitkan vektor fitur atas jendela 2 detik terakhirnya, beserta
       rentang waktu jendela dan jumlah sampelnya.
    4. Skrip memilih baris telemetri yang persis di rentang itu, menghitung
       fitur dengan ``feature_extractor.py``, lalu membandingkan per fitur.

Guncang node setelah skrip mulai; perintah dikirim begitu guncangan terdeteksi. Pada jendela tenang (1 Hz, ~2 sampel) energi, durasi
di atas ambang, dan waktu-menuju-puncak nyaris nol di kedua sisi sehingga
perbandingannya lulus tanpa benar-benar menguji apa pun.

Pemakaian::

    python -m ml_training.ml_selftest --node node_a0025bd3 --broker localhost

Kode keluar: 0 = setara, 1 = ada fitur yang berbeda, 2 = tidak dapat
disimpulkan (paket hilang, node tidak menjawab, dst.).
"""

import argparse
import json
import math
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.feature_extractor import (  # noqa: E402
    FEATURE_NAMES,
    RULE_PGA_MIN,
    extract_features,
    features_to_vector,
)
from ml.inference_engine import payload_to_row  # noqa: E402

# Toleransi. Firmware menghitung dengan float32 dan menerima nilai masukan yang
# sudah dibulatkan ArduinoJson; server memakai float64. Selisih sebesar itu
# wajar. Selisih yang lebih besar berarti definisi fiturnya berbeda.
REL_TOL = 1e-3
ABS_TOL = 1e-5

#: Di bawah ini jendela dianggap tenang: pengujian terlalu lemah.
MIN_SAMPLES_TEGAS = 8

EXIT_SETARA, EXIT_BEDA, EXIT_TAK_TERSIMPULKAN = 0, 1, 2


def select_rows(rows, t_first, t_last, eps=5e-4):
    """Baris telemetri yang ``sensor_ts``-nya di dalam [t_first, t_last]."""
    return [
        r for r in rows
        if r.get("sensor_ts") is not None
        and t_first - eps <= r["sensor_ts"] <= t_last + eps
    ]


def compare_features(node_vector, server_vector, rel_tol=REL_TOL, abs_tol=ABS_TOL):
    """Bandingkan dua vektor fitur; kembalikan satu dict per fitur."""
    hasil = []
    for name, a, b in zip(FEATURE_NAMES, node_vector, server_vector):
        cocok = math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
        hasil.append({"feature": name, "node": a, "server": b, "match": cocok})
    return hasil


def evaluate(reply, rows):
    """Nilai satu balasan ``ml_selftest`` terhadap baris telemetri yang terkumpul.

    Mengembalikan ``(kode_keluar, ringkasan, tabel)``.
    """
    feats = reply.get("features") or []
    if len(feats) != len(FEATURE_NAMES):
        return (
            EXIT_BEDA,
            f"jumlah fitur berbeda: node={len(feats)}, server={len(FEATURE_NAMES)}",
            [],
        )

    window = select_rows(rows, reply["t_first"], reply["t_last"])
    if len(window) != reply["n_samples"]:
        return (
            EXIT_TAK_TERSIMPULKAN,
            f"jendela tidak sejajar: node memakai {reply['n_samples']} sampel, "
            f"skrip menerima {len(window)} pada rentang waktu yang sama "
            "(paket MQTT hilang, atau timestamp berbeda antara buffer dan payload)",
            [],
        )

    server_vector = features_to_vector(extract_features(window))
    tabel = compare_features(feats, server_vector)
    beda = [t["feature"] for t in tabel if not t["match"]]

    catatan = ""
    if reply["n_samples"] < MIN_SAMPLES_TEGAS or not any(
        (r.get("pga") or 0.0) >= RULE_PGA_MIN for r in window
    ):
        catatan = (
            " PERINGATAN: jendela tenang, fitur energi/durasi/puncak tidak "
            "teruji. Ulangi sambil mengguncang node."
        )

    if beda:
        return EXIT_BEDA, "fitur berbeda: " + ", ".join(beda) + "." + catatan, tabel
    return EXIT_SETARA, f"{len(tabel)}/{len(tabel)} fitur setara." + catatan, tabel


def _print_table(tabel):
    print(f"{'fitur':<24}{'node':>16}{'server':>16}  ok")
    for t in tabel:
        print(f"{t['feature']:<24}{t['node']:>16.6g}{t['server']:>16.6g}  "
              f"{'ya' if t['match'] else 'BEDA'}")


def run(broker, port, node_id, warmup, timeout, wait_shake=True, shake_timeout=30.0):
    import paho.mqtt.client as mqtt

    rows = []
    reply_box = {}
    got_reply = threading.Event()
    lock = threading.Lock()

    def on_connect(client, _userdata, _flags, _rc):
        client.subscribe(f"lindu/sensor/{node_id}/telemetry")
        client.subscribe(f"lindu/sensor/{node_id}/ml_selftest")

    def on_message(_client, _userdata, msg):
        try:
            payload = json.loads(msg.payload)
        except ValueError:
            return
        if msg.topic.endswith("/telemetry"):
            with lock:
                rows.append(payload_to_row(payload))
        elif msg.topic.endswith("/ml_selftest"):
            reply_box.update(payload)
            got_reply.set()

    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(broker, port, 60)
    client.loop_start()
    try:
        print(f"Menyimak telemetri {node_id} — GUNCANG NODE SEKARANG "
              f"(menunggu guncangan hingga {shake_timeout:.0f} detik).")
        time.sleep(warmup)
        if wait_shake:
            # Node hanya menganalisis 2 detik terakhir, jadi perintah dikirim
            # tepat setelah guncangan terdeteksi, bukan pada waktu tebakan.
            deadline = time.time() + shake_timeout
            while time.time() < deadline:
                with lock:
                    terpicu = any((r.get("pga") or 0.0) >= RULE_PGA_MIN
                                  for r in rows[-8:])
                if terpicu:
                    time.sleep(0.6)  # biarkan burst terisi, tetap di dalam jendela
                    break
                time.sleep(0.1)
        client.publish(
            "lindu/actuator/cmd/all",
            json.dumps({"cmd": "ml_selftest", "target_node": node_id}),
        )
        if not got_reply.wait(timeout):
            return EXIT_TAK_TERSIMPULKAN, "node tidak menjawab ml_selftest", []
        time.sleep(0.3)  # beri kesempatan paket telemetri terakhir tiba
        with lock:
            snapshot = list(rows)
        return evaluate(dict(reply_box), snapshot) + (dict(reply_box),)
    finally:
        client.loop_stop()
        client.disconnect()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--node", required=True, help="node_id yang diuji")
    ap.add_argument("--broker", default=os.getenv("MQTT_BROKER", "localhost"))
    ap.add_argument("--port", type=int, default=int(os.getenv("MQTT_PORT", 1883)))
    ap.add_argument("--warmup", type=float, default=5.0)
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--no-wait-shake", action="store_true",
                    help="kirim perintah setelah warmup tanpa menunggu guncangan")
    ap.add_argument("--shake-timeout", type=float, default=30.0)
    args = ap.parse_args(argv)

    hasil = run(args.broker, args.port, args.node, args.warmup, args.timeout,
                wait_shake=not args.no_wait_shake, shake_timeout=args.shake_timeout)
    kode, ringkasan, tabel = hasil[0], hasil[1], hasil[2]
    reply = hasil[3] if len(hasil) > 3 else {}
    if reply:
        print(f"Model node: {reply.get('model')}  |  sampel: {reply.get('n_samples')}")
    if tabel:
        _print_table(tabel)
    print(("LULUS: " if kode == EXIT_SETARA else "GAGAL: " if kode == EXIT_BEDA
           else "TIDAK TERSIMPULKAN: ") + ringkasan)
    return kode


if __name__ == "__main__":
    sys.exit(main())
