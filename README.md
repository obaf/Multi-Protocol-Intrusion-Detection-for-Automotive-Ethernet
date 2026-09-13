# Multi-Protocol Intrusion Detection for Automotive Ethernet

Code, result tables, per-model predictions and figures for the manuscript

> *Multi-Protocol Intrusion Detection for Automotive Ethernet: A Unified Machine Learning Benchmark across AVTP, gPTP, CAN-over-UDP and SOME/IP* (under review, 2026; author list withheld for double-blind review).

The benchmark maps the two public labelled automotive Ethernet datasets, **TOW-IDS** (AVTP, gPTP, CAN-over-UDP) and the **SOME/IP dataset of Alkhatib et al.**, onto one timestamp-free schema of 119 per-packet features, trains ten learning algorithms on it and scores them in four evaluation suites:

| Suite | Question | Output |
|---|---|---|
| R1 | How well does each model detect attacks on each dataset alone (binary and multiclass)? | `results/r1_benchmark.csv` |
| R2 | What does one detector trained on all four protocol families cost relative to specialised detectors? | `results/r2_unified.csv` |
| R3 | Can a detector recognise attacks on a protocol family it has never seen under attack (leave-one-protocol-family-out)? | `results/r3_lofo.csv` |
| R4 | What do the models cost to deploy (size, throughput, detection latency)? | `results/r4_cost.csv`, `results/latency.json` |

Everything needed to regenerate the tables and figures from the raw datasets is here. The datasets themselves are not redistributed (see [Data](#data)).

## Contents

- [Why this work matters](#why-this-work-matters)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Data](#data)
- [Reproducing the experiments](#reproducing-the-experiments)
- [Expected results](#expected-results)
- [Design decisions](#design-decisions)
- [Result file formats](#result-file-formats)
- [Known limitations](#known-limitations)
- [Citation, licence and ethics](#citation-licence-and-ethics)

## Why this work matters

Published intrusion detection systems (IDS) for automotive Ethernet have each been trained and evaluated on a single protocol family, so nobody had measured whether one detector can protect a whole zonal vehicle network, or what that costs. This repository provides that measurement and, along the way, several findings that change how results in this area should be read:

1. **First multi-protocol benchmark from open data.** AVTP, gPTP, CAN-over-UDP and SOME/IP are brought into one feature space and one evaluation protocol, so detectors can be compared across protocol families for the first time (to our knowledge).
2. **A timestamp-free, protocol-agnostic feature schema.** 119 per-packet features (44 parsed header fields, 23 payload byte statistics, 46 raw leading bytes, 6 stream-context features) that need no flow reassembly, no decryption and no timestamps, so the same code runs on pcaps and on the 58-byte SOME/IP rows, and is suitable for an Ethernet-tap deployment.
3. **Unification is cheap for the lower protocol families.** A single detector trained on all families keeps 0.906 macro-F1 on TOW-IDS (logistic regression) and costs extra trees nothing measurable (0.890 vs 0.891). Multi-protocol defence is practical with current methods.
4. **Zero-day transfer across protocol families does not happen.** With one family held out of training, attack recall is 0.00 for four of five families. The one exception (switch MAC flooding) is caught only because of an engineered source-MAC-diversity feature. Supervised packet-level detectors learn family-specific signatures, so vehicle programmes need attack data for every protocol family they deploy.
5. **A leakage finding for the SOME/IP dataset.** Under a capture-disjoint split, no algorithm (tabular, deep or sequential) detects the SOME/IP attack classes from packet-level features. The published F1 scores above 0.8 for this dataset were obtained with a random window split and appear to measure within-capture memorisation. Detecting these attacks needs service-layer protocol state.
6. **A within-dataset distribution shift in the official TOW-IDS split.** The denial-of-service attacker's UDP source port differs between the training and test captures, so any model that fingerprints the port silently misses the whole class. Port-based features are fragile under attacker reconnection.
7. **Deployment-oriented reporting.** False-positive rates on the full unbalanced traffic, operating points at fixed recall, detection latency in packets and milliseconds, model size and throughput, plus every per-model test prediction, so others can re-score the same predictions with their own metrics.

## Repository layout

```
src/
  features_common.py            unified 119-feature extractor (static + stream context)
  build_towids.py               TOW-IDS pcaps + label CSVs -> data/towids.npz
  build_someip.py               SOME/IP pickles -> data/someip.npz (file-level 70/30 split)
  extract_towids_timestamps.py  per-packet timestamps -> data/towids_ts_{train,test}.npy (for latency)
  sanity_check.py               verifies label <-> protocol alignment before training
  models.py                     model zoo: LR, RF, ET, XGB, LGBM, MLP, 1D-CNN, GRU, VoteEns, StackEns
  exp_utils.py                  metrics bundle and sequence windowing
  run_experiments.py            suites R1-R4 -> results/*.csv and results/preds/*.npz
  analysis_latency.py           attack-segment detection latency -> results/latency.json
  analysis_operating_points.py  FPR at 99/95/90 % recall -> results/operating_points.json
results/                        all result tables and JSON files reported in the paper
results/preds/                  per-model test predictions (42 .npz files, 24 MB)
requirements.txt, package.json  Python and Node dependencies
data/                           empty; populated locally (see Data)
```

## Requirements

**Hardware.** Everything was run on an 8-core CPU without a GPU. Allow about 16 GB of RAM (both feature arrays are loaded at once, ~2 GB, plus model training), 2 GB of disk for the raw datasets and 0.2 GB for the derived `.npz` files. The full experiment run takes about 3 hours on 8 cores.

**Python 3.11 or later** (tested with 3.14.3) and the packages in `requirements.txt`:

```bash
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The CPU build of PyTorch is sufficient (`pip install torch --index-url https://download.pytorch.org/whl/cpu`). `dpkt` is only needed for the two TOW-IDS scripts that read pcaps.

**Optional, for the document generators only:** Node.js 18 or later (`npm install` in the repository root installs `docx` and `image-size`) and, for `papers/check_references.py`, `pandoc` on the PATH.

## Data

Both datasets are public research datasets released by their authors under their own terms. They are **not** redistributed here; please download them from the original sources and cite the dataset papers.

### TOW-IDS (AVTP, gPTP, CAN-over-UDP)

- Dataset: *TOW-IDS Automotive Ethernet Intrusion Dataset*, M. L. Han, B. I. Kwak and H. K. Kim (Korea University HCRL / DCRL). IEEE DataPort, DOI [10.21227/bz0w-zc12](https://dx.doi.org/10.21227/bz0w-zc12) (free IEEE account required), also mirrored from the [OCS Lab dataset page](https://ocslab.hksecurity.net/Datasets/tow-ids-automotive-ethernet-intrusion-dataset).
- Paper to cite: M. L. Han, B. I. Kwak, H. K. Kim, "TOW-IDS: Intrusion detection system based on three overlapped wavelets for automotive Ethernet," *IEEE Trans. Inf. Forensics Security*, vol. 18, pp. 411-422, 2023, doi:10.1109/TIFS.2022.3221893.
- Download `TOW-IDS_DATASET.zip` (about 207 MB) and extract it into `data/towids_extracted/`. The build script expects exactly these four files (ignore the `__MACOSX` folder):

  | File | Size (bytes) |
  |---|---|
  | `Automotive_Ethernet_with_Attack_original_10_17_19_50_training.pcap` | 198,979,326 |
  | `Automotive_Ethernet_with_Attack_original_10_17_20_04_test.pcap` | 134,309,632 |
  | `y_train.csv` | 26,326,022 |
  | `y_test.csv` | 17,173,503 |

- Label CSV row *i* is the label of the *i*-th packet in capture order (Normal, F_I, P_I, M_F, C_D, C_R). Expected counts: 1,203,737 training packets (954,912 Normal) and 791,611 test packets (660,777 Normal). `src/sanity_check.py` verifies the alignment: every F_I packet is a VLAN-tagged AVTP frame, every P_I packet a gPTP frame, every M_F/C_D/C_R packet IPv4.

### SOME/IP (Alkhatib, Ghauch and Danger)

- Dataset and code: <https://github.com/Alkhatibnatasha/supervised_detection_some_ip>. The README of that repository links the data as a Dropbox folder (about 1.1 GB zipped).
- Paper to cite: N. Alkhatib, H. Ghauch, J.-L. Danger, "SOME/IP intrusion detection using deep learning-based sequential models in automotive Ethernet networks," arXiv:2108.08262, 2021.
- Extract the archive into `data/someip_data/` so that it contains the four attack directories `Error_on_error` (35 captures), `Error_on_event` (43), `Missing_request` (40) and `Missing_response` (40). Each capture is a pair `<name>_<i>_x.pickle` (an N x 58 array of leading frame bytes) and `<name>_<i>_y.pickle` (N labels, 1 = attack). Expected total: 2,374,996 messages, of which 145,668 (6.1 %) are attacks. The dataset has no timestamps, which is why the shared feature schema is timestamp-free.

`data/README.md` shows the final directory layout.

## Reproducing the experiments

All commands are run from the repository root. Step numbers match the pipeline in the paper.

**1. Build the feature arrays** (single-threaded Python over 4.4 million frames; expect tens of minutes)

```bash
python src/build_towids.py             # -> data/towids.npz   (1,995,348 x 119)
python src/build_someip.py             # -> data/someip.npz   (2,374,996 x 119), file-level 70/30 split
python src/extract_towids_timestamps.py   # -> data/towids_ts_train.npy, data/towids_ts_test.npy
```

Both build scripts print per-class counts; they should match the numbers in the Data section (`build_towids.py`: `split 0 ... labels=[954912, 35112, 64635, 33765, 85466, 29847]`, `split 1 ... labels=[660777, 16962, 26013, 16809, 41203, 29847]`; `build_someip.py`: `label counts: [2229328, 27364, 28942, 44681, 44681]`).

**2. Check label alignment**

```bash
python src/sanity_check.py
```

**3. Run the experiment suites** (about 3 hours on 8 cores for `all`)

```bash
python src/run_experiments.py all          # or: r1 | r2 | r3 | r4
python src/run_experiments.py all --smoke  # quick end-to-end check with reduced training caps (skips StackEns)
```

Outputs: `results/r1_benchmark.csv`, `results/r2_unified.csv`, `results/r3_lofo.csv`, `results/r4_cost.csv` and one `results/preds/<task>_<model>.npz` per model and task. Seeds are fixed (42), so the tabular models are deterministic for a given set of library versions; deep-model results can differ slightly across PyTorch builds, and throughput and training time vary by up to about 30 % from run to run on a shared CPU.

**4. Post-hoc analyses**

```bash
python src/analysis_latency.py            # -> results/latency.json      (needs data/towids_ts_test.npy)
python src/analysis_operating_points.py   # -> results/operating_points.json
```


## Expected results

Values a successful reproduction should return (all from `results/`):

| Check | Expected |
|---|---|
| R1, TOW-IDS binary, StackEns macro-F1 / PR-AUC / FPR | 0.942 / 0.973 / 0.153 % |
| R1, TOW-IDS binary, tree-model FPR range (RF ... LGBM) | 0.009 % ... 0.106 % |
| R1, SOME/IP binary, best tabular model (RF) macro-F1 / PR-AUC | 0.508 / 0.068 (attack base rate 0.06) |
| R1, SOME/IP binary, GRU (window level) macro-F1 | 0.609 |
| R1, TOW-IDS multiclass, MLP accuracy / macro-F1 | 0.936 / 0.817 |
| R2, unified -> TOW-IDS, LR / ET / StackEns macro-F1 | 0.906 / 0.890 / 0.473 |
| R3, held-out switch family, XGB / LGBM / VoteEns attack recall | 1.00 (all other families 0.00) |
| R4, LightGBM size / throughput | 1.5 MB / ~483 k packets/s (8-core CPU) |
| Latency, StackEns segments alerted / median latency | 90.8 % of 92,409 / 0 packets |
| Operating point, LightGBM FPR at 99 % / 95 % recall | 7.14 % / 6.39 % |

## Design decisions

- **Splits.** The official TOW-IDS train/test split (two separate captures) is preserved. The SOME/IP dataset has no official split; captures are sorted per attack directory and the first 70 % go to training, so no capture contributes to both partitions. This is stricter than the random window split of the original SOME/IP evaluation and is the reason the SOME/IP numbers here are lower than the published ones.
- **Feature schema.** Timestamp-free by design so that both datasets share one representation (the SOME/IP release has no timestamps). Groups: 44 parsed header fields (Ethernet/VLAN, IPv4/UDP, SOME/IP header, gPTP, AVTP), 23 payload byte statistics over the first 64 payload bytes, 46 raw leading bytes scaled to [0, 1], and 6 order-based context features over a rolling window of 100 frames (including distinct source MACs, the MAC-flooding signal).
- **Training caps.** Training sets are stratified-subsampled to 500,000 rows per fit (minority classes kept proportionally); sequence models train on 150,000 windows. Test sets are never subsampled.
- **Imbalance.** Class weighting, not resampling, so false-positive rates reflect the true benign base rate.
- **Models.** Forests depth-limited to 20 with 250 trees; XGBoost and LightGBM 300 rounds; MLP 256-128-64; 1D-CNN and bidirectional GRU over 16-packet windows (stride 8, window positive if any packet is an attack); soft-voting and stacking ensembles of RF + XGB + LGBM. Deep models: early stopping, at most 25 epochs, CPU only.
- **Protocol families for R3.** AVTP = F_I; gPTP = P_I; switch = M_F; CAN-over-UDP = C_D + C_R; SOME/IP = its four classes.

## Result file formats

- `r1_benchmark.csv`, `r2_unified.csv`, `r3_lofo.csv`: one row per model and task with `acc`, `f1_macro`, `f1_weighted`, `precision_macro`, `recall_macro`, `pr_auc`, `roc_auc`, `fpr` (binary tasks), `confusion` (nested list), `per_class_recall`, `model`, `train_s`, `test_s`, `n_train`, `n_test`, `size_mb`, `thr_kmsg_s`, `task` and, for R2/R3, `test_set` / `held_out`. Throughput for the sequence models counts 16-packet windows per second.
- `r4_cost.csv`: size, training time and throughput of LR, RF, ET, XGB and LGBM **retrained on the unified binary corpus**. Table VI of the paper uses the per-model cost columns of the R1 TOW-IDS rows; this file quantifies the additional cost of unification (forests grow to 104-174 MB).
- `latency.json`: per model, the number of contiguous attack segments in the TOW-IDS test capture (92,409), how many were alerted, and the median latency from segment start to first alert in packets and milliseconds, with per-class segment counts.
- `operating_points.json`: false-positive rate at 99 %, 95 % and 90 % attack recall for the binary models.
- `preds/<task>_<model>.npz`: `y_true`, `y_pred` and, for binary tasks, `score` (attack probability) for every test packet or window, in test-set order. These allow any metric to be recomputed without retraining.

## Known limitations

- Both datasets come from testbeds (a physical bench for TOW-IDS, a generator for SOME/IP) and cover one vehicle configuration each; fleet traffic may differ.
- The schema deliberately omits timing features; adding them would likely raise TOW-IDS scores.
- Deep models were trained on a CPU with a small epoch budget; their ranking may change with more compute.
- Throughput and latency are CPU proxies, not measurements on automotive hardware. The relative ordering is the contribution, not the absolute numbers.
- The SOME/IP "missing message" labels mark the position at which an expected message fails to appear; this definition is inherited from the dataset.

## Citation and ethics

**Citation.** If you use this benchmark, the feature schema or the released predictions, please cite the manuscript (bibliographic details will be added on publication) together with the two dataset papers listed in the Data section.

**Ethics.** This is defensive security research on publicly released testbed data. No attacks were generated, modified or replayed against any real vehicle.
