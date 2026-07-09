#!/usr/bin/env python3
import re
import math
import sys
from pathlib import Path
from typing import List, Tuple

class Filter:
    def __init__(self, fc: float, gain: float, q: float):
        self.fc = fc
        self.gain = gain
        self.q = q

def parse_eq_profile(text: str) -> Tuple[float, List[Filter]]:
    """Parse Preamp and PK filters."""
    preamp_match = re.search(r'Preamp:\s*([-\d.]+)\s*dB', text, re.IGNORECASE)
    preamp = float(preamp_match.group(1)) if preamp_match else 0.0

    filters = []
    filter_pattern = r'Filter \d+:\s*ON PK Fc ([\d.]+) Hz Gain ([-\d.]+) dB Q ([\d.]+)'
    for match in re.finditer(filter_pattern, text, re.IGNORECASE):
        fc, gain, q = map(float, match.groups())
        filters.append(Filter(fc, gain, q))
    return preamp, filters

def peaking_gain_at_freq(f: float, fc: float, gain_db: float, q: float) -> float:
    """Compute the exact magnitude response of an analog peaking EQ filter at frequency f (in dB)."""
    if abs(gain_db) < 0.001:
        return 0.0
    
    # Normalized frequency relative to the center frequency
    w = f / fc
    
    # Gain scaling factor (square root of linear gain)
    A = 10 ** (gain_db / 40.0)
    
    # Continuous-time magnitude squared evaluation
    c = (1.0 - w ** 2) ** 2
    num = c + (w * A / q) ** 2
    den = c + (w / (A * q)) ** 2
    
    if den < 1e-12:
        return gain_db
        
    mag_sq = num / den
    return 10 * math.log10(max(mag_sq, 1e-18))

def a_weighting_linear(f: float) -> float:
    """
    Computes the linear magnitude of the A-weighting curve at frequency f.
    Based on the standard ANSI S1.4 formula.
    """
    f2 = f ** 2
    
    # Numerator
    num = (12194.0 ** 2) * (f2 ** 2)
    # Denominator
    den = (
        (f2 + 20.6 ** 2)
        * math.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2))
        * (f2 + 12194.0 ** 2)
    )
    
    # Multiply by 1.2589 to normalize 1kHz to 0 dB (gain of 1.0)
    return (num / den) * 1.2589

def simulate_profile_gain(preamp: float, filters: List[Filter]) -> float:
    """
    Computes perceived loudness using linear power summation, 
    A-weighting, and an implicit pink-noise spectrum assumption.
    """
    num_points = 500  # High resolution for precise Riemann sum
    f_min, f_max = 20.0, 20000.0
    
    # Logarithmic step multiplier
    step = (f_max / f_min) ** (1.0 / (num_points - 1))
    
    total_weighted_power = 0.0
    
    f = f_min
    for _ in range(num_points):
        # 1. Calculate total EQ gain in dB at this frequency
        gain_db = preamp
        for filt in filters:
            gain_db += peaking_gain_at_freq(f, filt.fc, filt.gain, filt.q)
        
        # 2. Convert decibels to linear power multiplier (10^(dB/10))
        eq_power = 10 ** (gain_db / 10.0)
        
        # 3. Calculate A-Weighting linear power
        a_weight_power = a_weighting_linear(f) ** 2
        
        # 4. Integrate (Sum) the weighted power
        total_weighted_power += eq_power * a_weight_power
        
        # Advance to next log-spaced frequency
        f *= step
        
    # Convert the average linear power back to Decibels
    avg_power = total_weighted_power / num_points
    
    if avg_power <= 0:
        return -float('inf')
        
    return 10 * math.log10(avg_power)

def main():
    if len(sys.argv) != 3:
        print("Usage: python eq_volume_match.py <source_eq.txt> <target_eq.txt>")
        print("  The script will output the target profile with preamp adjusted to match source volume.")
        sys.exit(1)

    source_path = Path(sys.argv[1])
    target_path = Path(sys.argv[2])

    if not source_path.exists() or not target_path.exists():
        print("Error: One or both files do not exist.")
        sys.exit(1)

    source_text = source_path.read_text(encoding="utf-8")
    target_text = target_path.read_text(encoding="utf-8")

    source_preamp, source_filters = parse_eq_profile(source_text)
    target_preamp, target_filters = parse_eq_profile(target_text)

    source_loudness = simulate_profile_gain(source_preamp, source_filters)
    target_loudness = simulate_profile_gain(target_preamp, target_filters)

    delta = source_loudness - target_loudness
    new_preamp = round(target_preamp + delta, 1)  # typical precision

    # Rebuild target text with new preamp
    lines = target_text.splitlines()
    updated = False
    for i, line in enumerate(lines):
        if re.search(r'Preamp:', line, re.IGNORECASE):
            lines[i] = re.sub(r'Preamp:\s*[-\d.]+\s*dB', f'Preamp: {new_preamp} dB', line, flags=re.IGNORECASE)
            updated = True
            break

    if not updated:
        # Prepend if no Preamp line found
        lines.insert(0, f'Preamp: {new_preamp} dB')

    adjusted_text = '\n'.join(lines)

    print(f"Source ({source_path.name}) perceived gain ≈ {source_loudness:.2f} dB")
    print(f"Target ({target_path.name}) original gain ≈ {target_loudness:.2f} dB")
    print(f"Delta: {delta:+.2f} dB")
    print(f"New Preamp: {new_preamp} dB\n")

    output_path = target_path.with_name(target_path.stem + "_volume_matched" + target_path.suffix)
    output_path.write_text(adjusted_text, encoding="utf-8")
    print(f"Adjusted profile saved to: {output_path}")

if __name__ == "__main__":
    main()