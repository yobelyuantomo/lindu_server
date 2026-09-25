"""Ekspor classifier menjadi kode C untuk dijalankan di ESP32.

--------------------------------------------------------------------------
Kenapa BUKAN TensorFlow Lite Micro seperti rencana awal
--------------------------------------------------------------------------
Proposal menyebut "kuantisasi ke int8 dan konversi ke TFLite Micro". Itu tidak
berlaku untuk model yang akhirnya terpilih: **Random Forest adalah ensemble
pohon keputusan, bukan jaringan saraf.** TFLite bekerja pada graf operasi
tensor; tidak ada jalur konversi dari ``sklearn.ensemble`` ke sana. Memaksakan
TFLite berarti harus mengganti modelnya dengan jaringan saraf, yang pada
dataset sebesar ini hampir pasti kalah dari model berbasis pohon sekaligus
jauh lebih sulit dijelaskan.

Untuk pohon keputusan di mikrokontroler, pendekatan yang lazim dan lebih baik
adalah **generasi kode**: struktur pohon ditulis langsung sebagai array C.
Keunggulannya nyata:

- Tidak ada runtime yang perlu ditanam. TFLite Micro sendiri memakan puluhan KB
  flash sebelum modelnya dihitung.
- Inferensi hanya berupa penelusuran array — tanpa perkalian matriks, tanpa
  alokasi memori, waktu eksekusinya dapat diprediksi.
- Tidak ada kehilangan akurasi. Kuantisasi int8 selalu menggeser hasil;
  di sini nilai ambangnya disalin apa adanya.
- Tidak membutuhkan TensorFlow sama sekali untuk membangunnya.

Formatnya sengaja berupa **array datar**, bukan rantai if/else. Array bisa
dibaca ulang dan diuji kesetaraannya dengan model Python, sedangkan kode
if/else yang dihasilkan otomatis praktis mustahil diverifikasi.

Pemakaian::

    python -m ml_training.export_edge_model \\
        --model ml/models/classifier.joblib \\
        --meta ml/models/model_meta.json \\
        --out ../esp32_sensor_node/src/EdgeModel.h
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.feature_extractor import FEATURE_NAMES  # noqa: E402

#: Penanda daun pada kolom feature: node tanpa anak.
LEAF = -1


def extract_trees(model):
    """Ubah ensemble sklearn menjadi struktur array datar.

    Tiap pohon menjadi empat larik sejajar: ``feature``, ``threshold``,
    ``left``, ``right``, ditambah ``value`` berisi proporsi kelas di tiap daun.
    """
    pohon = []
    for est in model.estimators_:
        t = est.tree_
        nilai = []
        for i in range(t.node_count):
            baris = t.value[i][0]
            total = float(sum(baris)) or 1.0
            nilai.append([float(v) / total for v in baris])
        pohon.append({
            "feature": [int(f) if t.children_left[i] != -1 else LEAF
                        for i, f in enumerate(t.feature)],
            "threshold": [float(v) for v in t.threshold],
            "left": [int(v) for v in t.children_left],
            "right": [int(v) for v in t.children_right],
            "value": nilai,
        })
    return pohon


def predict_with_trees(pohon, classes, vektor):
    """Evaluasi ensemble dari struktur array — semantik yang sama dengan C.

    Fungsi ini adalah **acuan kebenaran** untuk pengujian kesetaraan: kalau
    hasilnya cocok dengan sklearn, berarti array yang diekspor memang memuat
    pohon yang sama. Kode C menelusuri array yang persis sama dengan cara ini.
    """
    akumulasi = [0.0] * len(classes)
    for t in pohon:
        node = 0
        while t["feature"][node] != LEAF:
            if vektor[t["feature"][node]] <= t["threshold"][node]:
                node = t["left"][node]
            else:
                node = t["right"][node]
        for i, v in enumerate(t["value"][node]):
            akumulasi[i] += v

    total = sum(akumulasi) or 1.0
    proba = [v / total for v in akumulasi]
    terbaik = max(range(len(proba)), key=lambda i: proba[i])
    return classes[terbaik], proba[terbaik]


def _c_float(nilai):
    """Format float sebagai literal C yang sah.

    ``f"{-2.0:.6g}"`` menghasilkan ``"-2"``, dan ``-2f`` adalah konstanta
    bilangan bulat dengan akhiran ``f`` — ditolak compiler. Nilai ambang daun
    di sklearn memang tepat -2.0, jadi kasus ini pasti muncul di setiap model.
    """
    teks = f"{nilai:.6g}"
    if "." not in teks and "e" not in teks and "E" not in teks and "inf" not in teks:
        teks += ".0"
    return teks + "f"


def _c_array(nama, tipe, nilai, per_baris=12):
    baris = [f"static const {tipe} {nama}[] = {{"]
    for i in range(0, len(nilai), per_baris):
        potong = ", ".join(nilai[i:i + per_baris])
        baris.append(f"    {potong},")
    baris.append("};")
    return "\n".join(baris)


def generate_header(pohon, classes, model_version, feature_names):
    """Susun isi berkas ``EdgeModel.h``."""
    offsets, feature, threshold, left, right, value = [], [], [], [], [], []
    for t in pohon:
        offsets.append(len(feature))
        feature += [str(v) for v in t["feature"]]
        threshold += [_c_float(v) for v in t["threshold"]]
        left += [str(v) for v in t["left"]]
        right += [str(v) for v in t["right"]]
        for baris in t["value"]:
            value += [_c_float(v) for v in baris]
    offsets.append(len(feature))

    n_kelas = len(classes)
    total_node = len(feature)

    bagian = [
        "// BERKAS INI DIHASILKAN OTOMATIS - JANGAN DISUNTING DENGAN TANGAN.",
        "//",
        f"// Sumber   : ml_training/export_edge_model.py",
        f"// Model    : {model_version}",
        f"// Dibuat   : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"// Pohon    : {len(pohon)}   Node: {total_node}   Kelas: {n_kelas}",
        "//",
        "// Ensemble pohon keputusan sebagai array datar. Inferensi hanya berupa",
        "// penelusuran array: tanpa perkalian matriks, tanpa alokasi memori,",
        "// dan waktu eksekusinya dapat diprediksi.",
        "",
        "#pragma once",
        "#include <stdint.h>",
        "",
        f"#define EDGE_MODEL_VERSION \"{model_version}\"",
        f"#define EDGE_N_TREES   {len(pohon)}",
        f"#define EDGE_N_NODES   {total_node}",
        f"#define EDGE_N_CLASSES {n_kelas}",
        f"#define EDGE_N_FEATURES {len(feature_names)}",
        "",
        "// Urutan fitur WAJIB sama dengan FEATURE_NAMES di ml/feature_extractor.py.",
        "// Firmware harus mengisi vektor masukan dengan urutan ini, tidak boleh lain.",
        "static const char* const EDGE_FEATURE_NAMES[] = {",
    ]
    bagian += [f'    "{n}",' for n in feature_names]
    bagian += ["};", ""]
    bagian.append("static const char* const EDGE_CLASS_NAMES[] = {")
    bagian += [f'    "{c}",' for c in classes]
    bagian += ["};", ""]

    bagian.append(_c_array("EDGE_TREE_OFFSET", "int16_t", [str(v) for v in offsets]))
    bagian.append("")
    bagian.append(_c_array("EDGE_FEATURE", "int8_t", feature))
    bagian.append("")
    bagian.append(_c_array("EDGE_THRESHOLD", "float", threshold, 8))
    bagian.append("")
    bagian.append(_c_array("EDGE_LEFT", "int16_t", left))
    bagian.append("")
    bagian.append(_c_array("EDGE_RIGHT", "int16_t", right))
    bagian.append("")
    bagian.append(_c_array("EDGE_VALUE", "float", value, 8))
    bagian.append("")

    bagian += [
        "// Menelusuri seluruh pohon dan mengembalikan indeks kelas dengan",
        "// proporsi tertinggi. `confidence` boleh NULL bila tidak dibutuhkan.",
        "static inline int edge_model_predict(const float* x, float* confidence) {",
        "    float akumulasi[EDGE_N_CLASSES] = {0};",
        "    for (int t = 0; t < EDGE_N_TREES; t++) {",
        "        int base = EDGE_TREE_OFFSET[t];",
        "        int node = 0;",
        "        while (EDGE_FEATURE[base + node] != -1) {",
        "            if (x[EDGE_FEATURE[base + node]] <= EDGE_THRESHOLD[base + node]) {",
        "                node = EDGE_LEFT[base + node];",
        "            } else {",
        "                node = EDGE_RIGHT[base + node];",
        "            }",
        "        }",
        "        const float* v = &EDGE_VALUE[(base + node) * EDGE_N_CLASSES];",
        "        for (int c = 0; c < EDGE_N_CLASSES; c++) akumulasi[c] += v[c];",
        "    }",
        "",
        "    float total = 0.0f;",
        "    for (int c = 0; c < EDGE_N_CLASSES; c++) total += akumulasi[c];",
        "    if (total <= 0.0f) total = 1.0f;",
        "",
        "    int terbaik = 0;",
        "    for (int c = 1; c < EDGE_N_CLASSES; c++) {",
        "        if (akumulasi[c] > akumulasi[terbaik]) terbaik = c;",
        "    }",
        "    if (confidence) *confidence = akumulasi[terbaik] / total;",
        "    return terbaik;",
        "}",
        "",
    ]
    return "\n".join(bagian)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="classifier.joblib")
    parser.add_argument("--meta", required=True, help="model_meta.json")
    parser.add_argument("--out", required=True, help="path EdgeModel.h")
    parser.add_argument("--verify-samples", type=int, default=2000)
    args = parser.parse_args(argv)

    import joblib

    model = joblib.load(args.model)
    with open(args.meta, encoding="utf-8") as h:
        meta = json.load(h)

    if meta.get("feature_names") != list(FEATURE_NAMES):
        raise SystemExit(
            "[!] Urutan fitur model tidak cocok dengan feature_extractor. "
            "Latih ulang sebelum mengekspor."
        )
    if not hasattr(model, "estimators_"):
        raise SystemExit(
            f"[!] {type(model).__name__} bukan ensemble pohon. "
            "Eksportir ini hanya mendukung model berbasis pohon."
        )

    classes = [str(c) for c in model.classes_]
    pohon = extract_trees(model)
    total_node = sum(len(t["feature"]) for t in pohon)
    print(f"[OK] {len(pohon)} pohon, {total_node} node, {len(classes)} kelas")

    # Kesetaraan terhadap sklearn, diuji pada vektor acak. Ini yang membuktikan
    # array yang diekspor memuat pohon yang sama, bukan sekadar terlihat mirip.
    import random

    rng = random.Random(0)
    vektor = [[rng.uniform(-5, 5) for _ in FEATURE_NAMES]
              for _ in range(args.verify_samples)]
    # Satu panggilan batch, bukan per sampel: jauh lebih cepat dan tidak
    # membanjiri keluaran dengan peringatan joblib di setiap iterasi.
    label_sklearn = [str(v) for v in model.predict(vektor)]
    label_array = [predict_with_trees(pohon, classes, v)[0] for v in vektor]

    beda = [i for i, (a, b) in enumerate(zip(label_sklearn, label_array)) if a != b]
    if beda:
        raise SystemExit(
            f"[!] Kesetaraan GAGAL pada {len(beda)}/{len(vektor)} vektor "
            f"(indeks pertama: {beda[0]}). Jangan sebarkan berkas ini."
        )
    print(f"[OK] Kesetaraan dengan sklearn: {len(vektor)}/{len(vektor)} vektor acak")

    isi = generate_header(pohon, classes, meta.get("model_version", "unknown"), FEATURE_NAMES)
    direktori = os.path.dirname(os.path.abspath(args.out))
    if direktori:
        os.makedirs(direktori, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as h:
        h.write(isi)

    ukuran = os.path.getsize(args.out)
    print(f"[OK] {args.out} ({ukuran / 1024:.1f} KB sumber)")

    # Perkiraan jejak flash: tiap node menyimpan int8 + float + 2 x int16,
    # ditambah tabel nilai per kelas di setiap node.
    per_node = 1 + 4 + 2 + 2 + 4 * len(classes)
    print(f"[i] Perkiraan data di flash: {total_node * per_node / 1024:.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
