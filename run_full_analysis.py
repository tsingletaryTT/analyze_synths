"""
Full library analysis on TT Blackhole hardware.

Uses the single-threaded AudioAnalyzer so the TT JAX backend initializes once
without thread-lock contention. The hardware STFT kernel handles its own
internal parallelism across the 4 P300C chips.

Run with:
    source bin/activate_tt
    python3 run_full_analysis.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from audio_analysis.api.analyzer import AudioAnalyzer

SAMPLE_DIR = Path("/tmp/samples_canonical")
OUTPUT_DIR = Path("/home/ttuser/code/analyze_synths/generated/full_library")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    n_files = len(list(SAMPLE_DIR.iterdir()))
    print(f"Analyzing {n_files} files in {SAMPLE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    t0 = time.time()
    analyzer = AudioAnalyzer(str(SAMPLE_DIR))
    df = analyzer.analyze_directory()
    t1 = time.time()
    print(f"\nFeature extraction complete: {len(df)} tracks in {t1-t0:.1f}s "
          f"({(t1-t0)/len(df):.1f}s per file)")

    cluster_labels, centers, features = analyzer.perform_clustering()
    sequence = analyzer.recommend_sequence()
    print(f"Clustering and sequencing complete")

    export_info = analyzer.export_comprehensive_analysis(
        export_dir=OUTPUT_DIR,
        show_plots=False,
        export_format="all",
        base_name="full_library",
    )
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    print("\nExported files:")
    for k, v in export_info.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                print(f"  {kk}: {vv}")
        else:
            print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
