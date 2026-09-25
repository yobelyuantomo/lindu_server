"""Susun laporan perbandingan ML vs rule-based dari artefak pelatihan.

Dibuat sebagai generator, bukan dokumen statis, supaya angka di laporan selalu
berasal langsung dari artefak yang dihasilkan pelatihan. Menyalin angka dengan
tangan ke dalam dokumen adalah cara paling mudah membuat laporan berbeda dari
kenyataan tanpa ada yang menyadarinya.

Pemakaian::

    python -m ml_training.generate_report \\
        --model-meta ml/models/model_meta.json \\
        --baseline reports/baseline.json \\
        --out ../../docs/08_AIOT_EVALUATION_RESULTS.md

Jalankan ulang setiap kali model dilatih ulang.
"""

import argparse
import json
import os
from datetime import datetime, timezone


def _pct(nilai):
    return "—" if nilai is None else f"{nilai:.4f}"


def _delta(ml, rule, lebih_besar_lebih_baik=True):
    """Selisih beserta arahnya, dari sudut pandang 'apakah ML lebih baik'."""
    if ml is None or rule is None:
        return "—"
    selisih = ml - rule
    menang = selisih > 0 if lebih_besar_lebih_baik else selisih < 0
    tanda = "+" if selisih > 0 else ""
    penanda = " ✅" if menang and abs(selisih) > 1e-9 else (
        " ❌" if not menang and abs(selisih) > 1e-9 else ""
    )
    return f"{tanda}{selisih:.4f}{penanda}"


def bagian_ringkasan(model_meta, baseline):
    uji = model_meta.get("test_report", {})
    baris = [
        "## 1. Ringkasan Perbandingan",
        "",
        "Kedua sistem dievaluasi pada **split uji yang sama persis**, memakai",
        "dataset dan pembagian yang identik.",
        "",
        "| Metrik | Rule-based (Assignment 2) | ML (Assignment 3) | Selisih |",
        "|---|---|---|---|",
    ]
    metrik = [
        ("Accuracy", "accuracy", True),
        ("Macro F1", "macro_f1", True),
        ("False positive rate", "false_positive_rate", False),
    ]
    for nama, kunci, naik_baik in metrik:
        m, r = uji.get(kunci), baseline.get(kunci)
        arah = "lebih tinggi lebih baik" if naik_baik else "lebih rendah lebih baik"
        baris.append(
            f"| {nama} <br><sub>({arah})</sub> | {_pct(r)} | {_pct(m)} | "
            f"{_delta(m, r, naik_baik)} |"
        )
    baris += [
        "",
        f"Jumlah sampel uji: **{uji.get('n_samples', '—')}** jendela.",
        "",
        "> **False positive rate adalah metrik yang paling menentukan** untuk sistem",
        "> ini. Sirine yang berbunyi karena truk lewat merusak kepercayaan pengguna",
        "> jauh lebih cepat daripada akurasi yang turun sedikit — dan begitu",
        "> kepercayaan hilang, peringatan yang benar pun akan diabaikan.",
        "",
    ]
    return baris


def bagian_per_kelas(model_meta, baseline):
    baris = ["## 2. Rincian per Kelas", ""]
    for judul, laporan in (("Rule-based", baseline), ("ML", model_meta.get("test_report", {}))):
        per_kelas = laporan.get("per_class", {})
        if not per_kelas:
            continue
        baris += [
            f"### {judul}", "",
            "| Kelas | Precision | Recall | F1 | Sampel |",
            "|---|---|---|---|---|",
        ]
        for kelas in sorted(per_kelas):
            m = per_kelas[kelas]
            baris.append(
                f"| `{kelas}` | {_pct(m.get('precision'))} | {_pct(m.get('recall'))} "
                f"| {_pct(m.get('f1'))} | {m.get('support', '—')} |"
            )
        baris.append("")
    return baris


def bagian_confusion(laporan, judul):
    matriks = laporan.get("confusion_matrix") or {}
    labels = laporan.get("labels") or []
    if not matriks or not labels:
        return []
    baris = [f"### Confusion matrix — {judul}", "",
             "| asli \\ prediksi | " + " | ".join(f"`{l}`" for l in labels) + " |",
             "|---" * (len(labels) + 1) + "|"]
    for asli in labels:
        sel = " | ".join(str(matriks.get(f"{asli}->{pred}", 0)) for pred in labels)
        baris.append(f"| **`{asli}`** | {sel} |")
    baris.append("")
    return baris


def bagian_model(model_meta):
    penting = list((model_meta.get("feature_importance") or {}).items())[:10]
    baris = [
        "## 3. Model", "",
        f"- **Jenis**: {model_meta.get('model_type', '—')}",
        f"- **Versi**: `{model_meta.get('model_version', '—')}`",
        f"- **Dipilih berdasarkan**: {model_meta.get('selection_metric', '—')}",
        f"- **Seed**: {model_meta.get('seed', '—')}",
        f"- **Jumlah fitur**: {model_meta.get('n_features', '—')}",
        f"- **Baris latih / val / uji**: {model_meta.get('n_train_rows', '—')} / "
        f"{model_meta.get('n_val_rows', '—')} / {model_meta.get('n_test_rows', '—')}",
        "",
    ]
    if penting:
        baris += ["### Fitur paling berpengaruh", "",
                  "| Fitur | Kepentingan |", "|---|---|"]
        baris += [f"| `{n}` | {v:.4f} |" for n, v in penting]
        baris.append("")
    return baris


def bagian_magnitudo(mag_meta):
    if not mag_meta:
        return []
    model = mag_meta.get("model_report", {})
    dasar = (mag_meta.get("baselines") or {}).get("persistensi", {})
    return [
        "## 4. Estimasi Dini Puncak Guncangan", "",
        f"Dari **{mag_meta.get('lead_window_s', '—')} detik pertama** sebuah kejadian,",
        "seberapa kuat guncangan ini akan menjadi?",
        "",
        "| Metrik | Persistensi | Model | Selisih |",
        "|---|---|---|---|",
        f"| MAE (PGA, G) | {_pct(dasar.get('pga_mae'))} | {_pct(model.get('pga_mae'))} "
        f"| {_delta(model.get('pga_mae'), dasar.get('pga_mae'), False)} |",
        f"| RMSE (PGA, G) | {_pct(dasar.get('pga_rmse'))} | {_pct(model.get('pga_rmse'))} "
        f"| {_delta(model.get('pga_rmse'), dasar.get('pga_rmse'), False)} |",
        f"| MAE (magnitudo) | {_pct(dasar.get('magnitudo_mae'))} "
        f"| {_pct(model.get('magnitudo_mae'))} "
        f"| {_delta(model.get('magnitudo_mae'), dasar.get('magnitudo_mae'), False)} |",
        "",
        f"Kejadian latih / uji: {mag_meta.get('n_train_events', '—')} / "
        f"{mag_meta.get('n_test_events', '—')}",
        "",
        "**Pembandingnya persistensi** — anggap puncak sama dengan PGA tertinggi yang",
        "sudah terlihat di jendela awal. Itulah yang secara efektif dilakukan sistem",
        "lama. Pembanding ini dipilih karena tidak diturunkan dari model mana pun,",
        "berbeda dengan `tb_system_alerts.magnitude` yang justru dihasilkan formula",
        "rule-based sendiri sehingga tidak sah dijadikan target.",
        "",
        f"**Model mengalahkan persistensi:** {'ya' if mag_meta.get('beats_persistence') else 'TIDAK'}",
        "",
    ]


def bagian_limitasi(model_meta):
    return [
        "## 5. Keterbatasan", "",
        "Hal-hal berikut membatasi sejauh mana angka di atas boleh ditafsirkan.",
        "Ditulis terus terang karena menyembunyikannya justru lebih merugikan saat",
        "dipertanyakan.",
        "",
        "1. **Kelas `earthquake` berasal dari peragaan**, bukan gempa tektonik",
        "   sungguhan — meja getar dan simulasi Wokwi. Tidak ada gempa nyata yang",
        "   terjadi selama masa proyek. Metrik sebagus apa pun tidak membuktikan",
        "   sistem akan bekerja pada gempa sesungguhnya.",
        "",
        "2. **Klasifikasi hanya dilakukan pada jendela yang terpicu** (ada sampel",
        "   dengan PGA ≥ 0,12 G). Firmware mengirim telemetri 10 Hz saat terpicu dan",
        "   1 Hz saat tenang, sehingga kepadatan sampel sendiri sudah mengkodekan",
        "   lapisan pertama filter rule-based. Membatasi populasi ini mencegah model",
        "   'menang' hanya dengan menghafal laju sampling. Konsekuensinya, angka di",
        "   atas tidak berlaku untuk getaran tenang.",
        "",
        "3. **Dataset berukuran kecil.** Split dilakukan per kejadian, bukan per",
        "   baris, sehingga tidak ada kebocoran — tetapi jumlah kejadian yang sedikit",
        "   membuat metrik uji punya ketidakpastian yang besar.",
        "",
        "4. **Magnitudo tidak diprediksi secara langsung.** Tidak ada label magnitudo",
        "   independen; satu-satunya yang tersedia dihasilkan oleh formula rule-based",
        "   sendiri, sehingga melatih model untuk menirunya tidak bermakna.",
        "",
        "5. **Model belum diberi wewenang atas aktuator.** Sistem berjalan dalam",
        "   shadow mode: prediksi dicatat penuh tetapi jalur rule-based tetap yang",
        "   menentukan. Lihat `ml_training/README.md` untuk prosedur melepasnya.",
        "",
    ]


def susun(model_meta, baseline, mag_meta, catatan=None):
    baris = [
        "# Hasil Evaluasi Sistem AIoT (Assignment 3)",
        "",
        f"*Dihasilkan otomatis oleh `ml_training/generate_report.py` pada "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.*",
        "",
        "*Jangan menyunting berkas ini dengan tangan — jalankan ulang generatornya",
        "setiap kali model dilatih ulang, supaya angkanya tidak pernah berbeda dari",
        "artefak yang sesungguhnya.*",
        "",
    ]
    if catatan:
        baris += [f"> ⚠️ **{catatan}**", ""]
    baris += bagian_ringkasan(model_meta, baseline)
    baris += bagian_per_kelas(model_meta, baseline)
    baris += bagian_confusion(baseline, "Rule-based")
    baris += bagian_confusion(model_meta.get("test_report", {}), "ML")
    baris += bagian_model(model_meta)
    baris += bagian_magnitudo(mag_meta)
    baris += bagian_limitasi(model_meta)
    return "\n".join(baris)


def _muat(path, wajib=True):
    if not path or not os.path.exists(path):
        if wajib:
            raise SystemExit(f"[!] Berkas tidak ditemukan: {path}")
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-meta", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--magnitude-meta", help="opsional")
    parser.add_argument("--out", required=True)
    parser.add_argument("--note", help="catatan peringatan di bagian atas laporan")
    args = parser.parse_args(argv)

    isi = susun(
        _muat(args.model_meta),
        _muat(args.baseline),
        _muat(args.magnitude_meta, wajib=False),
        args.note,
    )

    direktori = os.path.dirname(os.path.abspath(args.out))
    if direktori:
        os.makedirs(direktori, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(isi + "\n")
    print(f"[OK] Laporan ditulis ke {args.out} ({len(isi.splitlines())} baris)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
