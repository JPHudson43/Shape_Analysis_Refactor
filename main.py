#!/usr/bin/env python3
"""
Hippocampal Morphometry Pipeline
--------------------------------
A comprehensive pipeline for hippocampal segmentation and shape analysis.
"""

import os
import sys
import argparse
import time
import logging
from datetime import timedelta
from typing import Dict, Optional, List, Any, Union

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import config # Imports config_module_v9_template_fix
except ImportError:
    logger.error("CRITICAL: config.py not found. Please ensure it exists in the PYTHONPATH.")
    sys.exit(1)

try:
    from preprocessing import segment_hippocampus, binarize_masks, register_masks
    from shape_analysis import run_spharm_pipeline_adapted 
    from statistics import run_static_analysis, prepare_longitudinal_data, run_longitudinal_analysis 
    from utils import (dir_to_csv, check_files, excel_to_csv, 
                       print_box_header, print_section_header, 
                       merge_covariates_to_static_csv) 
except ImportError as e:
    logger.error(f"CRITICAL: Failed to import one or more pipeline modules: {e}")
    logger.error("Please ensure all pipeline modules are in the PYTHONPATH and have correct names.")
    sys.exit(1)


def run_full_pipeline(args: argparse.Namespace):
    start_time = time.time()
    print_box_header("HIPPOCAMPAL MORPHOMETRY PIPELINE")
    
    is_interactive_mode = not (args.non_interactive if args.non_interactive is not None else config.NON_INTERACTIVE)
    if args.non_interactive: 
        logger.info("Running in NON-INTERACTIVE mode due to command-line flag.")
    elif config.NON_INTERACTIVE:
        logger.info("Running in NON-INTERACTIVE mode due to config.py setting.")
    else:
        logger.info("Running in INTERACTIVE mode.")

    print_section_header("[1/8] Setting up environment")
    if not config.create_directories(non_interactive_flag= not is_interactive_mode): 
        logger.error("Failed to create necessary directories. Exiting.")
        return
    environment_check_passed = config.check_environment() 
    if not environment_check_passed and not args.skip_env_check:
        logger.error("Environment check failed. Fix issues or use --skip-env-check to proceed anyway.")
        return
    
    print_section_header("[2/8] Hippocampal segmentation (if not skipped)")
    if args.skip_segmentation or config.SKIP_SEGMENTATION:
        logger.info("Skipping segmentation step.")
    else:
        # Pass the specific input directory for segmentation from args
        segment_hippocampus(input_dir_param=args.input_dir_segmentation) 

    print_section_header("[3/8] Binarizing segmentation masks (if not skipped)")
    if args.skip_binarization or config.SKIP_BINARIZATION:
        logger.info("Skipping binarization step.")
    else:
        binarize_masks() 
    
    print_section_header("[4/8] Registering hippocampal masks (if not skipped)")
    if args.skip_registration or config.SKIP_REGISTRATION:
        logger.info("Skipping registration step.")
    else:
        register_masks() 
    
    print_section_header("[5/8] Running SPHARM-PDM processing")
    if args.skip_spharm or config.SKIP_SPHARM:
        logger.info("Skipping SPHARM-PDM step.")
    else:
        effective_with_gui = args.with_gui if args.with_gui is not None else config.WITH_GUI
        if args.with_gui is not None: config.WITH_GUI = args.with_gui
        
        run_spharm_pipeline_adapted() 
    
    print_section_header("[6/8] Preparing CSVs for statistical analysis")
    logger.info("Automatically generating base static CSVs with SubjectID and SubjectPath...")
    dir_to_csv(side="left") 
    dir_to_csv(side="right")

    logger.info("Attempting to merge covariates into static CSVs...")
    merge_covariates_to_static_csv(static_csv_path=config.STATIC_LEFT_CSV)
    merge_covariates_to_static_csv(static_csv_path=config.STATIC_RIGHT_CSV)
    
    if not (os.path.exists(config.STATIC_LEFT_CSV) and os.path.exists(config.STATIC_RIGHT_CSV)):
         logger.warning("One or both static CSV files are still missing after attempting automatic creation/merge.")
    else:
        logger.info(f"Static CSVs prepared: {config.STATIC_LEFT_CSV}, {config.STATIC_RIGHT_CSV}")

    print_section_header("[7/8] Running static statistical analysis")
    if args.skip_stats or config.SKIP_STATS:
        logger.info("Skipping static statistical analysis.")
    else:
        if not (os.path.exists(config.STATIC_LEFT_CSV) and os.path.exists(config.STATIC_RIGHT_CSV)):
            logger.error("Cannot run static analysis: one or both required CSV files are missing or failed to be created/merged.")
            if is_interactive_mode: 
                logger.warning("Static analysis CSVs are missing.")
        else: 
            if config.SIDE == "both" or config.SIDE == "left":
                logger.info("Running static analysis for left hemisphere...")
                run_static_analysis(side="left") 
            if config.SIDE == "both" or config.SIDE == "right":
                logger.info("Running static analysis for right hemisphere...")
                run_static_analysis(side="right") 
    
    print_section_header("[8/8] Running longitudinal analysis")
    if args.skip_longitudinal or config.SKIP_LONGITUDINAL:
        logger.info("Skipping longitudinal analysis.")
    else:
        longit_left_m2md_csv = config.LONGITUDINAL_LEFT_M2MD_CSV 
        longit_right_m2md_csv = config.LONGITUDINAL_RIGHT_M2MD_CSV
        if not (os.path.exists(longit_left_m2md_csv) and os.path.exists(longit_right_m2md_csv)):
            logger.warning(f"CSV files for longitudinal M2MD not found: {longit_left_m2md_csv}, {longit_right_m2md_csv}. Skipping M2MD preparation.")
        else:
            if config.SIDE == "both" or config.SIDE == "left":
                logger.info("Preparing longitudinal data (M2MD) for left hemisphere...")
                prepare_longitudinal_data(side="left") 
            if config.SIDE == "both" or config.SIDE == "right":
                logger.info("Preparing longitudinal data (M2MD) for right hemisphere...")
                prepare_longitudinal_data(side="right") 

            longit_stats_left_csv_path = config.LONGITUDINAL_STATS_LEFT_CSV
            longit_stats_right_csv_path = config.LONGITUDINAL_STATS_RIGHT_CSV
            if not (os.path.exists(longit_stats_left_csv_path) and os.path.exists(longit_stats_right_csv_path)):
                logger.warning(f"CSV files for longitudinal stats not found: {longit_stats_left_csv_path}, {longit_stats_right_csv_path}. Skipping longitudinal stats.")
            else:
                if config.SIDE == "both" or config.SIDE == "left":
                    logger.info("Running longitudinal statistical analysis for left hemisphere...")
                    run_longitudinal_analysis(side="left") 
                if config.SIDE == "both" or config.SIDE == "right":
                    logger.info("Running longitudinal statistical analysis for right hemisphere...")
                    run_longitudinal_analysis(side="right") 
    
    elapsed_time_seconds = time.time() - start_time
    elapsed_timedelta = timedelta(seconds=int(elapsed_time_seconds))
    print_box_header(f"Pipeline completed in {elapsed_timedelta}")

def run_segmentation_cmd(args: argparse.Namespace):
    logger.info("Running standalone segmentation...")
    segment_hippocampus(input_dir_param=args.input_dir_segmentation) 

def run_binarization_cmd(args: argparse.Namespace):
    logger.info("Running standalone binarization...")
    binarize_masks()

def run_registration_cmd(args: argparse.Namespace):
    logger.info("Running standalone registration...")
    register_masks()

def run_spharm_cmd(args: argparse.Namespace):
    logger.info(f"Running standalone SPHARM-PDM processing (adapted logic)...")
    effective_with_gui = args.with_gui if args.with_gui is not None else config.WITH_GUI
    if args.with_gui is not None: config.WITH_GUI = args.with_gui 
    
    run_spharm_pipeline_adapted()
    logger.info("Standalone SPHARM-PDM (adapted logic) finished.")


def run_static_stats_cmd(args: argparse.Namespace):
    logger.info(f"Running standalone static statistical analysis for side: {args.side}...")
    
    logger.info(f"Ensuring base static CSV for {args.side} is created/merged...")
    if args.side == "both":
        dir_to_csv(side="left")
        dir_to_csv(side="right")
        merge_covariates_to_static_csv(static_csv_path=config.STATIC_LEFT_CSV)
        merge_covariates_to_static_csv(static_csv_path=config.STATIC_RIGHT_CSV)
    else:
        dir_to_csv(side=args.side)
        static_csv = config.STATIC_LEFT_CSV if args.side == "left" else config.STATIC_RIGHT_CSV
        merge_covariates_to_static_csv(static_csv_path=static_csv)

    if args.side in ["left", "both"]:
        left_csv_exists_and_not_empty = False
        if os.path.exists(config.STATIC_LEFT_CSV):
            try:
                if pd.read_csv(config.STATIC_LEFT_CSV).shape[0] > 0: 
                    left_csv_exists_and_not_empty = True
            except pd.errors.EmptyDataError: 
                logger.warning(f"Static CSV for left hemisphere ({config.STATIC_LEFT_CSV}) is empty.")
        
        if left_csv_exists_and_not_empty:
            logger.info("Running static analysis for left hemisphere...")
            run_static_analysis(side="left")
        else:
            logger.error(f"Static CSV for left hemisphere not found or empty ({config.STATIC_LEFT_CSV}), cannot run stats.")

    if args.side in ["right", "both"]:
        right_csv_exists_and_not_empty = False
        if os.path.exists(config.STATIC_RIGHT_CSV):
            try:
                if pd.read_csv(config.STATIC_RIGHT_CSV).shape[0] > 0:
                    right_csv_exists_and_not_empty = True
            except pd.errors.EmptyDataError:
                logger.warning(f"Static CSV for right hemisphere ({config.STATIC_RIGHT_CSV}) is empty.")

        if right_csv_exists_and_not_empty:
            logger.info("Running static analysis for right hemisphere...")
            run_static_analysis(side="right")
        else:
            logger.error(f"Static CSV for right hemisphere not found or empty ({config.STATIC_RIGHT_CSV}), cannot run stats.")


def run_longitudinal_cmd(args: argparse.Namespace):
    logger.info(f"Running standalone longitudinal analysis for side: {args.side}...")
    if args.side in ["left", "both"]:
        logger.info("Preparing longitudinal data for left hemisphere...")
        prepare_longitudinal_data(side="left")
        if args.run_stats:
            logger.info("Running longitudinal statistics for left hemisphere...")
            run_longitudinal_analysis(side="left")
    if args.side in ["right", "both"]:
        logger.info("Preparing longitudinal data for right hemisphere...")
        prepare_longitudinal_data(side="right")
        if args.run_stats:
            logger.info("Running longitudinal statistics for right hemisphere...")
            run_longitudinal_analysis(side="right")

def run_utilities_cmd(args: argparse.Namespace):
    logger.info(f"Running utility: {args.util_choice}...")
    if args.util_choice == "dir2csv":
        results = dir_to_csv(side=args.side) 
        logger.info(f"dir2csv utility completed for side: {args.side}.")
        if args.side == "both":
            logger.info(f"Expected CSVs: {results.get('left', 'N/A')} and {results.get('right', 'N/A')}")
        else:
            logger.info(f"Expected CSV: {results}")
        logger.info("IMPORTANT: If you intend to use these for static stats, run 'utils merge_static_covariates' or manually add covariates.")

    elif args.util_choice == "merge_static_covariates": 
        logger.info(f"Attempting to merge covariates for side: {args.side}")
        if args.side == "both" or args.side == "left":
            merge_covariates_to_static_csv(static_csv_path=config.STATIC_LEFT_CSV)
        if args.side == "both" or args.side == "right":
            merge_covariates_to_static_csv(static_csv_path=config.STATIC_RIGHT_CSV)
        logger.info("Covariate merge attempt finished.")

    elif args.util_choice == "check_files":
        if not args.csv_file:
            logger.error("Please specify a CSV file with --csv-file for check_files utility.")
            return
        missing_files = check_files(csv_file_path=args.csv_file, file_column_header=args.column_name or "SubjectPath") 
        if missing_files:
            logger.warning(f"Missing Files ({len(missing_files)}):")
            for file_path in missing_files:
                print(f"  - {file_path}") 
        else:
            logger.info(f"All files listed in {args.csv_file} (column: {args.column_name or 'SubjectPath'}) exist or CSV had issues.")
            
    elif args.util_choice == "excel2csv":
        if not args.excel_file:
            logger.error("Please specify an Excel file with --excel-file for excel2csv utility.")
            return
        output_location = excel_to_csv(excel_file_path=args.excel_file) 
        logger.info(f"excel2csv utility completed. Output location: {output_location}")

def main():
    parser = argparse.ArgumentParser(description="Hippocampal Morphometry Pipeline.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--non-interactive", action="store_true", 
                        help="Run without interactive prompts. If set, overrides config.NON_INTERACTIVE for this run.")
    parser.add_argument("--with-gui", action="store_true",
                        help="Attempt to run SPHARM-PDM with its GUI. If set, overrides config.WITH_GUI for this run.")

    subparsers = parser.add_subparsers(dest="command", help="Command to run", required=True)
    
    pipeline_parser = subparsers.add_parser("pipeline", help="Run the complete pipeline.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Corrected help string and variable name in f-string
    pipeline_parser.add_argument("--input-dir-segmentation", type=str, default=None, 
                                 help=f"Directory for initial T1s for segmentation (if not skipped). Defaults to config.T1_IMAGES_DIR_ROOT_FOR_SPHARM_INPUT ('{config.T1_IMAGES_DIR_ROOT_FOR_SPHARM_INPUT}').")
    pipeline_parser.add_argument("--skip-env-check", action="store_true", help="Skip environment check and proceed even if issues are found.")
    pipeline_parser.add_argument("--skip-segmentation", action="store_true", help="Skip segmentation step.")
    pipeline_parser.add_argument("--skip-binarization", action="store_true", help="Skip binarization step.")
    pipeline_parser.add_argument("--skip-registration", action="store_true", help="Skip registration step.")
    pipeline_parser.add_argument("--skip-spharm", action="store_true", help="Skip SPHARM-PDM processing.")
    pipeline_parser.add_argument("--skip-stats", action="store_true", help="Skip static statistical analysis.")
    pipeline_parser.add_argument("--skip-longitudinal", action="store_true", help="Skip longitudinal analysis.")
    pipeline_parser.set_defaults(func=run_full_pipeline)
    
    seg_parser = subparsers.add_parser("segment", help="Run hippocampal segmentation only.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Corrected help string and variable name in f-string
    seg_parser.add_argument("--input-dir-segmentation", type=str, default=None, 
                            help=f"Directory containing T1 MRI images for segmentation. Defaults to config.T1_IMAGES_DIR_ROOT_FOR_SPHARM_INPUT ('{config.T1_IMAGES_DIR_ROOT_FOR_SPHARM_INPUT}').")
    seg_parser.set_defaults(func=run_segmentation_cmd)
    
    bin_parser = subparsers.add_parser("binarize", help="Binarize segmentation masks only.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    bin_parser.set_defaults(func=run_binarization_cmd)
    
    reg_parser = subparsers.add_parser("register", help="Register binarized masks only.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    reg_parser.set_defaults(func=run_registration_cmd)
    
    spharm_parser = subparsers.add_parser("spharm", help="Run SPHARM-PDM processing (adapted logic for all subjects/sides).", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    spharm_parser.set_defaults(func=run_spharm_cmd) 
    
    stats_parser = subparsers.add_parser("static-stats", help="Run static statistical analysis only.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    stats_parser.add_argument("--side", choices=["left", "right", "both"], default=config.SIDE, help="Hemisphere to process.")
    stats_parser.set_defaults(func=run_static_stats_cmd)
    
    longit_parser = subparsers.add_parser("longitudinal", help="Run longitudinal analysis (data prep & optionally stats).", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    longit_parser.add_argument("--side", choices=["left", "right", "both"], default=config.SIDE, help="Hemisphere to process.")
    longit_parser.add_argument("--run-stats", action="store_true", help="Also run longitudinal statistical analysis after data preparation.")
    longit_parser.set_defaults(func=run_longitudinal_cmd)
    
    utils_parser = subparsers.add_parser("utils", help="Run utility functions.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    utils_parser.add_argument("util_choice", choices=["dir2csv", "check_files", "excel2csv", "merge_static_covariates"], help="Utility to run.")
    utils_parser.add_argument("--side", choices=["left", "right", "both"], default="both", help="Hemisphere for dir2csv or merge_static_covariates.")
    utils_parser.add_argument("--csv-file", type=str, help="Path to CSV file (for check_files).")
    utils_parser.add_argument("--column-name", type=str, help="Name of the column containing file paths (for check_files). Defaults to SubjectPath.")
    utils_parser.add_argument("--excel-file", type=str, help="Path to Excel file (for excel2csv).")
    utils_parser.set_defaults(func=run_utilities_cmd)
    
    setup_parser = subparsers.add_parser("setup", help="Create project directory structure based on config.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    setup_parser.set_defaults(func=lambda args_lambda: config.create_directories(non_interactive_flag=args_lambda.non_interactive))

    args = parser.parse_args()
    
    if args.non_interactive: 
        config.NON_INTERACTIVE = True 
    
    if args.with_gui:
        config.WITH_GUI = True
    
    if hasattr(args, 'func'):
        try:
            args.func(args)
        except Exception as e:
            logger.error(f"An error occurred while executing command '{args.command}': {str(e)}")
            logger.exception("Detailed traceback:") 
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
