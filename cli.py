# cli.py
# Command‑line interface for interactive IADC code entry.

import argparse
from main import run_design
from iadc_mapper import decode_iadc

def interactive_mode():
    print("Enter IADC code (e.g., M431):")
    code = input("> ").strip().upper()
    try:
        params = decode_iadc(code)
        print("Decoded parameters:", params)
        run_design(params)
    except Exception as e:
        print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Drill Bit Optimizer CLI")
    parser.add_argument("--iadc", type=str, help="IADC code (e.g., M431)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
    elif args.iadc:
        params = decode_iadc(args.iadc)
        run_design(params)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()