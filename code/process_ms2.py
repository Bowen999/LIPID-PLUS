#!/usr/bin/env python3
import argparse
import pandas as pd
import sys
import mcid

def main():
    parser = argparse.ArgumentParser(description="Process MS2 feature table.")
    parser.add_argument("input_path", help="Input CSV file path")
    parser.add_argument("--output_path", default="feature_unfold_ms2.csv", help="Output CSV file")
    parser.add_argument("--decimal_point", default=0, type=int, help="Decimal point (integer)")
    parser.add_argument("--neutral_loss", action="store_true", help="Enable neutral loss")
    parser.add_argument("--no-neutral_loss", dest="neutral_loss", action="store_false", help="Disable neutral loss")
    parser.set_defaults(neutral_loss=True)

    parser.add_argument("--keep_intensity", action="store_true", help="Keep intensity column")
    parser.add_argument("--no-keep_intensity", dest="keep_intensity", action="store_false")
    parser.set_defaults(keep_intensity=False)

    args = parser.parse_args()

    # Load file
    try:
        df = pd.read_csv(args.input_path)
    except Exception as e:
        print(f"Error reading input file: {e}")
        sys.exit(1)

    # Check required columns
    missing = []
    if "precursor_mz" not in df.columns:
        missing.append("precursor_mz")
    if ("ion_mode" not in df.columns) and ("adduct" not in df.columns):
        missing.append("ion_mode or adduct")
    if "MS2" not in df.columns:
        missing.append("MS2")

    if missing:
        print("Error: Missing required column(s): " + ", ".join(missing))
        sys.exit(1)

    # Process MS2
    df = mcid.ms2_norm(df)
    df = mcid.process_ms2_df(
        df,
        decimal_point=args.decimal_point,
        neutral_loss=args.neutral_loss,
        keep_intensity=args.keep_intensity
    )
    df = mcid.ms2_norm(df)
    

    # Add index if missing
    if "index" not in df.columns:
        df.insert(0, "index", [f"F{i:06d}" for i in range(1, len(df) + 1)])

    if "ion_mode" not in df.columns and "adduct" in df.columns:
        def infer_mode(adduct):
            if not isinstance(adduct, str) or len(adduct) == 0:
                return None
            last = adduct.strip()[-1]
            if last in ["+", "]"]:
                return "Positive"
            if last == "-":
                return "Negative"
            return None

        df["ion_mode"] = df["adduct"].apply(infer_mode)

    # Save
    df.to_csv(args.output_path, index=False)
    print(f"Saved processed file to {args.output_path}")


if __name__ == "__main__":
    main()
