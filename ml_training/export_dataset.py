"""Ekspor telemetri mentah dari PostgreSQL ke CSV untuk pelatihan model.

Sumber dataset **hanya** tabel ``sensor_telemetry`` (ditulis oleh ingester).
Sengaja tidak join dengan ``tb_sensor_telemetry``: kedua tabel ditulis oleh
proses berbeda dengan semantik waktu berbeda (ingester memakai waktu terima di
server, consensus memakai epoch dari sensor), sehingga jendela sinyalnya tidak
akan sejajar. ``sensor_telemetry`` juga satu-satunya yang memuat fitur
lingkungan (suhu, tekanan, gas) sekaligus fitur seismik.

Pemakaian::

    python -m ml_training.export_dataset --out data/raw/telemetry.csv
    python -m ml_training.export_dataset --out x.csv --since 2026-09-01 --node NODE_A

Catatan penting: baris tanpa ``freq_hz`` dibuang. Frekuensi dominan adalah
fitur wajib classifier dan tidak dapat direkonstruksi dari kolom lain, jadi
mengimputasinya sama saja mengarang data. Seluruh baris yang direkam sebelum
kolom ``freq_hz`` ditambahkan ke ingester otomatis ikut terbuang di sini.
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.feature_extractor import rule_based_verdict  # noqa: E402

#: Kolom CSV hasil ekspor, berurutan.
#:
#: ``sta_lta`` sengaja dipetakan dari kolom ``rms`` di database: ingester
#: menulis nilai ``sta_lta`` dari payload MQTT ke kolom bernama ``rms``
#: (lihat ingester.py). Penamaan di database menyesatkan, jadi diluruskan
#: di sini agar cocok dengan kunci yang diharapkan feature_extractor.
EXPORT_COLUMNS = [
    "sensor_ts",
    "recv_time",
    "node_id",
    "pga",
    "sta_lta",
    "freq_hz",
    "accel_x",
    "accel_y",
    "accel_z",
    "temperature",
    "pressure",
    "humidity",
    "gas_raw",
    "gas_alert",
    "valve_status",
    "passed_rule",
]

_QUERY = """
    SELECT
        sensor_ts,
        time         AS recv_time,
        node_id,
        pga,
        rms          AS sta_lta,
        freq_hz,
        accel_x,
        accel_y,
        accel_z,
        temperature,
        pressure,
        humidity,
        gas_raw,
        gas_alert,
        valve_status
    FROM sensor_telemetry
    WHERE freq_hz IS NOT NULL
      {extra_where}
    ORDER BY node_id, COALESCE(sensor_ts, EXTRACT(EPOCH FROM time))
"""

_COUNT_QUERY = """
    SELECT
        COUNT(*)                                        AS total,
        COUNT(*) FILTER (WHERE freq_hz IS NULL)         AS tanpa_freq,
        COUNT(*) FILTER (WHERE freq_hz IS NOT NULL)     AS dengan_freq
    FROM sensor_telemetry
"""


def build_query(since=None, until=None, nodes=None):
    """Susun query beserta parameternya.

    Dipisah dari eksekusi agar bisa diuji tanpa database.
    """
    clauses, params = [], []
    if since:
        clauses.append("AND time >= %s")
        params.append(since)
    if until:
        clauses.append("AND time < %s")
        params.append(until)
    if nodes:
        clauses.append("AND node_id = ANY(%s)")
        params.append(list(nodes))
    return _QUERY.format(extra_where=" ".join(clauses)), params


def annotate_row(row):
    """Lengkapi satu baris hasil query dengan kolom turunan ``passed_rule``.

    Verdict rule-based sengaja dihitung di sini, bukan disimpan di database:
    ia adalah fungsi murni dari pga/sta_lta/freq_hz yang ketiganya sudah ada,
    sehingga mempersistensinya hanya menciptakan kemungkinan tidak sinkron.
    """
    out = dict(row)
    out["passed_rule"] = rule_based_verdict(
        row.get("pga"), row.get("sta_lta"), row.get("freq_hz")
    )
    return out


def write_csv(rows, path):
    """Tulis baris ke CSV. Mengembalikan jumlah baris yang ditulis."""
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)

    written = 0
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPORT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: annotate_row(row).get(col) for col in EXPORT_COLUMNS})
            written += 1
    return written


def _connect(dsn):
    import psycopg2
    from psycopg2.extras import RealDictCursor

    conn = psycopg2.connect(dsn)
    return conn, RealDictCursor


def export(dsn, out_path, since=None, until=None, nodes=None):
    conn, cursor_factory = _connect(dsn)
    try:
        with conn.cursor(cursor_factory=cursor_factory) as cur:
            cur.execute(_COUNT_QUERY)
            counts = cur.fetchone()
            print(
                f"[DB] sensor_telemetry: {counts['total']} baris total | "
                f"{counts['dengan_freq']} punya freq_hz | "
                f"{counts['tanpa_freq']} TANPA freq_hz (dibuang)"
            )
            if counts["dengan_freq"] == 0:
                print(
                    "[!] Tidak ada satu pun baris dengan freq_hz. Ingester versi "
                    "baru kemungkinan belum ter-deploy — jalankan "
                    "`docker-compose up -d --build` di prototype/grafana-stack."
                )

            query, params = build_query(since, until, nodes)
            cur.execute(query, params)
            rows = cur.fetchall()

        written = write_csv(rows, out_path)
        print(f"[OK] {written} baris ditulis ke {out_path}")

        per_node = {}
        for row in rows:
            per_node[row["node_id"]] = per_node.get(row["node_id"], 0) + 1
        for node_id, jumlah in sorted(per_node.items()):
            print(f"      {node_id}: {jumlah} baris")
        return written
    finally:
        conn.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dsn",
        default=os.getenv(
            "LINDU_DSN", "host=localhost dbname=lindu_db user=postgres password=postgres"
        ),
        help="DSN PostgreSQL (atau set env LINDU_DSN)",
    )
    parser.add_argument("--out", required=True, help="Path CSV keluaran")
    parser.add_argument("--since", help="Batas bawah waktu, mis. 2026-09-01")
    parser.add_argument("--until", help="Batas atas waktu (eksklusif)")
    parser.add_argument("--node", action="append", dest="nodes", help="Batasi node (boleh diulang)")
    args = parser.parse_args(argv)

    export(args.dsn, args.out, args.since, args.until, args.nodes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
