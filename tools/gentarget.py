#!/usr/bin/env python3
"""
gentarget.py
Generates custom ISO 226:2023 equal-loudness contours.

Features:
- Standard ISO 226:2023 math (verified against the 2023 edition).
- Optional dynamic _LU derivation from your custom target curve.
- Normalization to 0 dB at a chosen reference frequency.
- Your energy-balanced curve is embedded as the default _LU table.
"""

import sys
import os
import argparse
import csv
import numpy as np

# ISO 226:2023 Table 1 parameters (exact values from the standard)
_F = np.array([20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0,
               200.0, 250.0, 315.0, 400.0, 500.0, 630.0, 800.0, 1000.0, 1250.0,
               1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0, 6300.0, 8000.0,
               10000.0, 12500.0, 16000.0, 20000.0])

_ALPHA = np.array([0.635, 0.602, 0.569, 0.537, 0.509, 0.482, 0.456, 0.433, 0.412,
                   0.391, 0.373, 0.357, 0.343, 0.330, 0.320, 0.311, 0.303, 0.300,
                   0.295, 0.292, 0.290, 0.290, 0.289, 0.289, 0.289, 0.293, 0.303,
                   0.323, 0.354, 0.354, 0.354])

_TF = np.array([78.1, 68.7, 59.5, 51.1, 44.0, 37.5, 31.5, 26.5, 22.1, 17.9, 14.4,
                11.4, 8.6, 6.2, 4.4, 3.0, 2.2, 2.4, 3.5, 1.7, -1.3, -4.2, -6.0,
                -5.4, -1.5, 6.0, 12.6, 13.9, 12.3, 12.3, 12.3])

# Your energy-balanced custom _LU table (derived from your target curve)
_LU_DEFAULT = np.array([7.8, 7.0, 7.6, 8.3, 8.8, 9.2, 9.4, 9.4, 9.1, 8.6, 8.1,
                        7.3, 6.4, 5.3, 4.3, 3.1, 1.6, 0.0, 2.1, 1.7, -3.1, -6.9,
                        -8.5, -6.9, -2.9, 3.1, 8.5, 10.0, 4.8, 5.6, 6.3])


def derive_lu_from_curve(csv_filepath: str, base_phon: float):
    """Derive _LU array from a custom target curve (reverses ISO formula)."""
    try:
        data = np.loadtxt(csv_filepath, delimiter=',', skiprows=1)
        target_freqs, target_spl = data[:, 0], data[:, 1]
    except Exception as e:
        print(f"Error loading input CSV '{csv_filepath}': {e}")
        sys.exit(1)

    target_spl_std = np.interp(_F, target_freqs, target_spl)

    # Normalize input curve to base_phon at 1 kHz
    idx_1k = np.where(_F == 1000.0)[0][0]
    offset = base_phon - target_spl_std[idx_1k]
    shifted_spl = target_spl_std + offset

    # Safeguard against threshold
    shifted_spl = np.maximum(shifted_spl, _TF + 0.05)

    # Reverse of Formula (1)
    term1 = (4e-10)**(0.3 - _ALPHA) * (10**(0.03 * base_phon) - 10**0.072)
    denominator = 10**(_ALPHA * shifted_spl / 10.0) - 10**(_ALPHA * _TF / 10.0)
    Y = term1 / denominator
    return (10.0 / _ALPHA) * np.log10(Y)


def calculate_iso_spl(phon: float, freqs: np.ndarray, lu_array: np.ndarray):
    """Compute SPL using ISO 226:2023 Formula (1)."""
    lf = np.log10(freqs)
    alpha_f = np.interp(lf, np.log10(_F), _ALPHA)
    lu = np.interp(lf, np.log10(_F), lu_array)
    tf = np.interp(lf, np.log10(_F), _TF)

    term1 = (4e-10)**(0.3 - alpha_f) * (10**(0.03 * phon) - 10**0.072)
    term2 = 10**(alpha_f * (tf + lu) / 10.0)
    Af = np.maximum(term1 + term2, 1e-12)

    return (10.0 / alpha_f) * np.log10(Af) - lu


def main():
    parser = argparse.ArgumentParser(
        description="Generate ISO 226:2023-based target curves.\n"
                    "Supports dynamic _LU derivation from your custom energy-balanced curve.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("-p", "--phon", type=float, default=85.0,
                        help="Target loudness level in phon (default: 85)")
    parser.add_argument("-r", "--ref", type=float, default=1000.0,
                        help="Reference frequency for 0 dB normalization (default: 1000 Hz)")

    parser.add_argument("-i", "--input", type=str, default=None,
                        help="Optional CSV to derive _LU table from your custom curve")
    parser.add_argument("-b", "--base-phon", type=float, default=85.0,
                        help="Phon level of the --input curve (default: 85)")

    args = parser.parse_args()

    if args.input:
        print(f"Deriving _LU from custom curve: {args.input}")
        print(f"(Assuming input represents {args.base_phon} phon)")
        active_lu = derive_lu_from_curve(args.input, args.base_phon)
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        output_file = f"{base_name}_iso226_{args.phon:.0f}phon_ref{args.ref:.0f}hz.csv"
    else:
        print("Using built-in energy-balanced _LU_DEFAULT table.")
        active_lu = _LU_DEFAULT
        output_file = f"iso226_{args.phon:.0f}phon_ref{args.ref:.0f}hz.csv"

    freqs = np.logspace(np.log10(20), np.log10(20000), 2048)
    spl = calculate_iso_spl(args.phon, freqs, active_lu)

    # Normalize
    ref_idx = np.argmin(np.abs(freqs - args.ref))
    spl -= spl[ref_idx]

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frequency", "raw"])
        for fr, s in zip(freqs, spl):
            writer.writerow([f"{fr:.6f}", f"{s:.6f}"])

    print(f"Saved: {output_file} (0 dB @ {args.ref:.0f} Hz)")


if __name__ == "__main__":
    main()