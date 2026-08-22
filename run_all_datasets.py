#!/usr/bin/env python3
"""
Batch processor for Table2Image-VIF across all OpenML datasets
Enhanced with Weight Decay 
"""

import os
import sys
import subprocess
import argparse
import json
import time
from pathlib import Path
import pandas as pd
from datetime import datetime

def create_output_structure(base_output_dir, job_id):
    """
    Create organized folder structure for results
    """
    timestamp = datetime.now().strftime("%Y%m%d")
    run_name = f"{timestamp}_JOB{job_id}"
    
    run_dir = os.path.join(base_output_dir, run_name)
    
    # Create subdirectories
    subdirs = {
        'csv': os.path.join(run_dir, 'csv'),
        'latex': os.path.join(run_dir, 'latex'),
        'logs': os.path.join(run_dir, 'logs'),
        
    }
    
    for subdir in subdirs.values():
        os.makedirs(subdir, exist_ok=True)
    
    # Create README with run information
    readme_path = os.path.join(run_dir, 'README.txt')
    with open(readme_path, 'w') as f:
        f.write(f"Table2Image-VIF Batch Processing Results\n")
        f.write(f"="*50 + "\n\n")
        f.write(f"Job ID: {job_id}\n")
        f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\nConfiguration:\n")
        f.write(f"  - Weight Decay: 1e-4 (AdamW optimizer)\n")
        f.write(f"  - Model Saving: Disabled\n")
        f.write(f"\nFolder Structure:\n")
        f.write(f"  csv/              - Results in CSV format\n")
        f.write(f"  latex/            - LaTeX tables for paper\n")
        f.write(f"  logs/             - Processing logs (JSONL format)\n")
        f.write(f"\nImages saved to: /home/gkianfar/scratch/Amin/AI/output/imageout/\n")
    
    return run_dir, subdirs

def find_datasets(datasets_dir):
    """
    Find all dataset files in folder structure
    """
    dataset_files = []
    datasets_path = Path(datasets_dir)
    
    for folder in sorted(datasets_path.iterdir()):
        if not folder.is_dir():
            continue
        
        # Search for data files
        found_files = []
        for pattern in ['*.csv', '*.arff', '*.data']:
            found_files.extend(list(folder.glob(pattern)))
        
        if found_files:
            dataset_files.append(found_files[0])
            if len(found_files) > 1:
                print(f"[INFO] Folder '{folder.name}' has {len(found_files)} files, using: {found_files[0].name}")
        else:
            print(f"[WARNING] No data files found in folder: {folder.name}")
    
    return dataset_files



def run_single_dataset(dataset_path, subdirs, script_path, timeout):
    """
    Run Table2Image-VIF on a single dataset
    """
    dataset_name = dataset_path.parent.name

    print(f"\n{'='*70}")
    print(f"Processing: {dataset_name}")
    print(f"{'='*70}")
    print(f"Folder: {dataset_path.parent.name}")
    print(f"File: {dataset_path.name}")

    # --- per‑dataset overrides (CIFAR‑10 special case) ---
    effective_timeout = timeout
    num_images = '5'

    # Adjust these names if your CIFAR folder/file is different
    if dataset_name.lower() in ['cifar', 'cifar-10', 'cifar10']:
        print(f"🔧 Detected CIFAR‑10 – increasing timeout and reducing images per class")
        effective_timeout = max(timeout, 28800)  # at least 4 hours
        

    # Build command with per‑dataset num_images
    cmd = [
        'python', script_path,
        '--data', str(dataset_path),
        '--num_images', num_images,
    ]

    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            timeout=effective_timeout,
            capture_output=True,
            text=True
        )
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ SUCCESS in {elapsed:.1f}s")
            
            # Extract JSON results
            results_data = None
            try:
                stdout = result.stdout
                if "RESULTS_JSON_START" in stdout and "RESULTS_JSON_END" in stdout:
                    json_start = stdout.find("RESULTS_JSON_START") + len("RESULTS_JSON_START")
                    json_end = stdout.find("RESULTS_JSON_END")
                    json_str = stdout[json_start:json_end].strip()
                    results_data = json.loads(json_str)
                    print(f"📊 Tabular Accuracy: {results_data.get('best_accuracy', 0):.2f}%")
                    print(f"📊 AUC: {results_data.get('best_auc', 0):.4f}")
                    print(f"🖼️  Images: {results_data.get('images_dir', 'N/A')}")
            except Exception as e:
                print(f"⚠️  Could not parse results: {e}")
            
            
            
            return {
                'status': 'success',
                'dataset': dataset_name,
                'elapsed_time': elapsed,
                'results': results_data,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
        else:
            print(f"❌ FAILED (exit code {result.returncode})")
            stderr_preview = result.stderr[-1000:] if result.stderr else "No error output"
            print(f"Error preview:\n{stderr_preview}")
            return {
                'status': 'failed',
                'dataset': dataset_name,
                'elapsed_time': elapsed,
                'exit_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"⏱️  TIMEOUT after {elapsed:.1f}s ({effective_timeout}s limit)")

        return {
            'status': 'timeout',
            'dataset': dataset_name,
            'elapsed_time': elapsed
        }
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"💥 EXCEPTION: {str(e)}")
        return {
            'status': 'exception',
            'dataset': dataset_name,
            'elapsed_time': elapsed,
            'error': str(e)
        }

def parse_results_jsonl(subdirs):
    """
    Parse results.jsonl and create summary DataFrame
    """
    results_file = os.path.join(subdirs['logs'], 'results.jsonl')
    
    if not os.path.exists(results_file):
        print(f"[WARNING] No results file found at {results_file}")
        return None
    
    results = []
    with open(results_file, 'r') as f:
        for line in f:
            try:
                results.append(json.loads(line))
            except:
                pass
    
    if not results:
        print("[WARNING] No valid results found in results.jsonl")
        return None
    
    df = pd.DataFrame(results)
    df = df.sort_values('best_accuracy', ascending=False)
    
    return df

def create_summary_tables(df, subdirs, run_dir):
    """
    Create CSV and LaTeX summary tables with enhanced statistics
    """
    if df is None or len(df) == 0:
        print("[WARNING] No results to summarize")
        return
    
    # Select columns for summary
    summary_df = df[[
        'dataset', 'num_samples', 'num_features', 'num_classes',
        'best_accuracy', 'best_auc', 'best_epoch'
    ]].copy()
    
    # Calculate statistics
    avg_accuracy = summary_df['best_accuracy'].mean()
    avg_auc = summary_df['best_auc'].mean()
    std_accuracy = summary_df['best_accuracy'].std()
    std_auc = summary_df['best_auc'].std()
    
    # ========== 1. RESULTS SUMMARY CSV ==========
    csv_summary_path = os.path.join(subdirs['csv'], 'results_summary.csv')
    summary_df.to_csv(csv_summary_path, index=False)
    print(f"\n✅ Results summary: {csv_summary_path}")
    
    # ========== 2. DETAILED RESULTS CSV ==========
    csv_detailed_path = os.path.join(subdirs['csv'], 'results_detailed.csv')
    df.to_csv(csv_detailed_path, index=False)
    print(f"✅ Detailed results: {csv_detailed_path}")
    
    # ========== 3. 🆕 ENHANCED STATISTICS CSV ==========
    stats_data = {
        'Metric': [
            'Average Accuracy', 'Std Accuracy', 'Average AUC', 'Std AUC',
            'Best Accuracy', 'Worst Accuracy', 'Median Accuracy',
            'Datasets >90%', 'Datasets >95%', 'Datasets >99%',
            'Configuration', 'Weight Decay'
            
        ],
        'Value': [
            f"{avg_accuracy:.2f}",
            f"{std_accuracy:.2f}",
            f"{avg_auc:.4f}",
            f"{std_auc:.4f}",
            f"{summary_df['best_accuracy'].max():.2f}",
            f"{summary_df['best_accuracy'].min():.2f}",
            f"{summary_df['best_accuracy'].median():.2f}",
            len(summary_df[summary_df['best_accuracy'] > 90]),
            len(summary_df[summary_df['best_accuracy'] > 95]),
            len(summary_df[summary_df['best_accuracy'] > 99]),
            'AdamW + VIF Initialization',
            '1e-4'
            
        ]
    }
    stats_df = pd.DataFrame(stats_data)
    csv_stats_path = os.path.join(subdirs['csv'], 'statistics.csv')
    stats_df.to_csv(csv_stats_path, index=False)
    print(f"✅ Statistics: {csv_stats_path}")
    
    # ========== 4. LATEX TABLE ==========
    latex_path = os.path.join(subdirs['latex'], 'results_latex.txt')
    with open(latex_path, 'w') as f:
        f.write("% LaTeX Table for Paper - Table2Image-VIF Results\n")
        f.write("% With Weight Decay (1e-4)\n\n")
        
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Table2Image-VIF Performance on OpenML-CC18 Benchmark}\n")
        f.write("\\label{tab:results}\n")
        f.write("\\begin{tabular}{lrrrccc}\n")
        f.write("\\hline\n")
        f.write("\\textbf{Dataset} & \\textbf{N} & \\textbf{Features} & "
                "\\textbf{Classes} & \\textbf{Acc (\\%)} & \\textbf{AUC} & \\textbf{Epoch} \\\\\n")
        f.write("\\hline\n")
        
        # Top 20 datasets
        top_n = min(20, len(summary_df))
        for idx, row in summary_df.head(top_n).iterrows():
            dataset_name = row['dataset'].replace('_', '\\_')
            f.write(f"{dataset_name} & "
                    f"{row['num_samples']:,} & "
                    f"{row['num_features']} & "
                    f"{row['num_classes']} & "
                    f"{row['best_accuracy']:.2f} & "
                    f"{row['best_auc']:.4f} & "
                    f"{row['best_epoch']} \\\\\n")
        
        if len(summary_df) > top_n:
            f.write(f"\\multicolumn{{7}}{{c}}{{\\textit{{... {len(summary_df) - top_n} more datasets}}}} \\\\\n")
        
        f.write("\\hline\n")
        f.write(f"\\textbf{{Average}} & "
                f"- & - & - & "
                f"\\textbf{{{avg_accuracy:.2f}}} $\\pm$ {std_accuracy:.2f} & "
                f"\\textbf{{{avg_auc:.4f}}} $\\pm$ {std_auc:.4f} & "
                f"- \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
        
        # 🆕 Add note about robustness
        f.write("\n% Note: Results obtained with:\n")
        f.write("% - Weight decay (lambda=1e-4) for improved generalization\n")
        
    
    print(f"✅ LaTeX table: {latex_path}")
    

    
    
    
    # ========== 6. PRINT SUMMARY ==========
    print(f"\n{'='*70}")
    print(f"SUMMARY STATISTICS (With Weight Decay 1e-4)")
    print(f"{'='*70}")
    print(f"Total datasets: {len(summary_df)}")
    print(f"")
    print(f"📊 AVERAGE ACCURACY: {avg_accuracy:.2f}% ± {std_accuracy:.2f}%")
    print(f"📊 AVERAGE AUC:      {avg_auc:.4f} ± {std_auc:.4f}")
    print(f"")
    print(f"🏆 Best:  {summary_df.iloc[0]['dataset']:30s} {summary_df.iloc[0]['best_accuracy']:.2f}%")
    print(f"📉 Worst: {summary_df.iloc[-1]['dataset']:30s} {summary_df.iloc[-1]['best_accuracy']:.2f}%")
    print(f"")
    print(f"Datasets with >90% accuracy: {len(summary_df[summary_df['best_accuracy'] > 90])}")
    print(f"Datasets with >95% accuracy: {len(summary_df[summary_df['best_accuracy'] > 95])}")
    print(f"Datasets with >99% accuracy: {len(summary_df[summary_df['best_accuracy'] > 99])}")
    print(f"")
    
    print(f"{'='*70}\n")
    
    # Update README with final results
    readme_path = os.path.join(run_dir, 'README.txt')
    with open(readme_path, 'a') as f:
        f.write(f"\n\nFinal Results:\n")
        f.write(f"="*50 + "\n")
        f.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Datasets processed: {len(summary_df)}\n")
        f.write(f"Average Accuracy: {avg_accuracy:.2f}% ± {std_accuracy:.2f}%\n")
        f.write(f"Average AUC: {avg_auc:.4f} ± {std_auc:.4f}\n")
        

def main():
    parser = argparse.ArgumentParser(
        description='Batch process all OpenML datasets with Table2Image-VIF'
    )
    parser.add_argument('--datasets_dir', type=str, required=True,
                        help='Directory containing dataset folders')
    parser.add_argument('--output_base', type=str, required=True,
                        help='Base output directory (results/ folder)')
    parser.add_argument('--job_id', type=str, required=True,
                        help='SLURM job ID for folder naming')
    parser.add_argument('--script_path', type=str, required=True,
                        help='Path to run_vif.py')
    parser.add_argument('--timeout', type=int, default=7200,
                        help='Timeout per dataset in seconds (default: 2 hours)')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip datasets that already have results')
    
    args = parser.parse_args()
    
    # Create organized output structure
    print(f"{'='*70}")
    print(f"TABLE2IMAGE-VIF BATCH PROCESSOR")
    print(f"Configuration: Weight Decay (1e-4) ")
    print(f"{'='*70}")
    run_dir, subdirs = create_output_structure(args.output_base, args.job_id)
    print(f"Output directory: {run_dir}")
    print(f"  📊 csv/              → {subdirs['csv']}")
    print(f"  📄 latex/            → {subdirs['latex']}")
    print(f"  📝 logs/             → {subdirs['logs']}")
    print(f"{'='*70}\n")
    
    # Find all datasets
    print(f"{'='*70}")
    print(f"DISCOVERING DATASETS")
    print(f"{'='*70}")
    print(f"Searching in: {args.datasets_dir}")
    print(f"")
    
    dataset_files = find_datasets(args.datasets_dir)
    
    print(f"\nFound {len(dataset_files)} datasets:")
    for i, dataset_path in enumerate(dataset_files[:15], 1):
        print(f"  {i:2d}. {dataset_path.parent.name:40s} → {dataset_path.name}")
    if len(dataset_files) > 15:
        print(f"  ... and {len(dataset_files) - 15} more datasets")
    print(f"{'='*70}\n")
    
    if len(dataset_files) == 0:
        print("❌ No datasets found!")
        return 1
    
    # Process each dataset
    results_log = []
    success_count = 0
    failed_count = 0
    timeout_count = 0
    skipped_count = 0
    
    start_time = time.time()
    
    for i, dataset_path in enumerate(dataset_files, 1):
        elapsed_hours = (time.time() - start_time) / 3600
        remaining = len(dataset_files) - i
        
        print(f"\n{'='*70}")
        print(f"Progress: {i}/{len(dataset_files)} datasets")
        print(f"Elapsed: {elapsed_hours:.1f}h | Remaining: ~{remaining * 0.1:.1f}h")
        print(f"✅ {success_count} | ❌ {failed_count} | ⏱️ {timeout_count} | ⏭️ {skipped_count}")
        print(f"{'='*70}")
        
        # Check if already processed
        if args.skip_existing:
            dataset_name = dataset_path.parent.name
            progress_log_path = os.path.join(subdirs['logs'], 'progress_log.jsonl')
            if os.path.exists(progress_log_path):
                with open(progress_log_path, 'r') as f:
                    already_processed = any(dataset_name in line for line in f)
                
                if already_processed:
                    print(f"⏭️  SKIPPED: {dataset_name} (result found in log)")
                    skipped_count += 1
                    continue
        
        # Run dataset
        result = run_single_dataset(
            dataset_path=dataset_path,
            subdirs=subdirs,
            script_path=args.script_path,
            timeout=args.timeout
        )
        
        results_log.append(result)
        
        if result['status'] == 'success':
            success_count += 1
        elif result['status'] == 'timeout':
            timeout_count += 1
        else:
            failed_count += 1
        
        # Save progress log
        progress_log_path = os.path.join(subdirs['logs'], 'progress_log.jsonl')
        with open(progress_log_path, 'a') as f:
            f.write(json.dumps(result) + '\n')
        
        # Save results to results.jsonl if available
        if result['status'] == 'success' and result.get('results'):
            results_jsonl_path = os.path.join(subdirs['logs'], 'results.jsonl')
            with open(results_jsonl_path, 'a') as f:
                f.write(json.dumps(result['results']) + '\n')
    
    # Final summary
    total_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"  ✅ Success: {success_count}")
    print(f"  ❌ Failed: {failed_count}")
    print(f"  ⏱️  Timeout: {timeout_count}")
    print(f"  ⏭️  Skipped: {skipped_count}")
    print(f"{'='*70}\n")
    
    # Create summary tables
    if success_count > 0:
        print("Creating summary tables...")
        df = parse_results_jsonl(subdirs)
        create_summary_tables(df, subdirs, run_dir)
    else:
        print("⚠️  No successful results to summarize")
    
    print(f"\n📂 All results saved to: {run_dir}")
    
    return 0 if failed_count == 0 else 1

if __name__ == '__main__':
    sys.exit(main())
