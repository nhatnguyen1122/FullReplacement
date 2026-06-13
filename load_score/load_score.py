from pathlib import Path
import csv

script_dir = Path(__file__).resolve().parent

csv_path = "/Users/nhatnguyen/Documents/Training Lab cô Bình/TAPL/code/workspace/FullReplacementMCTS/output_nvidia/kmod_sp_mcts_baseline_10runs_gpt_oss_120b/summary.csv"
column_name = "primary_score"
output_path = script_dir / "output.txt"

with open(csv_path, newline="", encoding="utf-8") as csv_file:
    reader = csv.DictReader(csv_file)

    with open(output_path, "w", encoding="utf-8") as out_file:
        for row in reader:
            out_file.write(row[column_name] + "\n")

print(f"Saved to {output_path}")