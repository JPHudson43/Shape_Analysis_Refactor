#!/usr/bin/env python3
"""
Hippocampal Morphometry Pipeline
--------------------------------
A comprehensive pipeline for hippocampal segmentation and shape analysis.

This script serves as the main entry point for the pipeline, allowing users
to run the entire process or individual components.
"""

import os
import sys
import argparse
import time
import logging
from datetime import timedelta
from typing import Dict, Optional, List, Any, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import configuration
try:
    import config
except ImportError:
    logger.error("CRITICAL: config.py not found. Please ensure it exists in the PYTHONPATH.")
    sys.exit(1)

# Import modules
try:
    from preprocessing import segment_hippocampus, binarize_masks, register_masks
    from shape_analysis import run_spharm_pdm # extract_aligned_models is used within run_spharm_pdm
    # Corrected import for the statistics module
    from statistics import run_static_analysis, prepare_longitudinal_data, run_longitudinal_analysis 
    from utils import dir_to_csv, check_files, excel_to_csv, print_box_header, print_section_header
except ImportError as e:
    logger.error(f"CRITICAL: Failed to import one or more pipeline modules: {e}")
    logger.error("Please ensure all pipeline modules (preprocessing.py, shape_analysis.py, statistics.py, utils.py) are in the PYTHONPATH.")
    sys.exit(1)


def run_full_pipeline(args: argparse.Namespace):
    """
    Run the complete hippocampal morphometry pipeline.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
    """
    start_time = time.time()
    print_box_header("HIPPOCAMPAL MORPHOMETRY PIPELINE")
    
    is_interactive_mode = not (args.non_interactive or config.NON_INTERACTIVE)

    # Setup environment
    print_section_header("[1/8] Setting up environment")
    if not config.create_directories(non_interactive_flag=not is_interactive_mode):
        logger.error("Failed to create necessary directories. Exiting.")
        return
        
    environment_check_passed = config.check_environment() # check_environment uses config.NON_INTERACTIVE
    
    if not environment_check_passed and not args.skip_env_check:
        logger.error("Environment check failed. Fix issues or use --skip-env-check to proceed anyway.")
        return
    
    # Segmentation
    print_section_header("[2/8] Hippocampal segmentation")
    if args.skip_segmentation or config.SKIP_SEGMENTATION:
        logger.info("Skipping segmentation step as per configuration or command-line flag.")
    else:
        # Corrected keyword argument for segment_hippocampus
        segment_hippocampus(input_dir_param=args.input_dir) 
    
    # Binarization
    print_section_header("[3/8] Binarizing segmentation masks")
    if args.skip_binarization or config.SKIP_BINARIZATION:
        logger.info("Skipping binarization step as per configuration or command-line flag.")
    else:
        binarize_masks() 
    
    # Registration
    print_section_header("[4/8] Registering hippocampal masks")
    if args.skip_registration or config.SKIP_REGISTRATION:
        logger.info("Skipping registration step as per configuration or command-line flag.")
    else:
        register_masks() 
    
    # SPHARM-PDM Processing
    print_section_header("[5/8] Running SPHARM-PDM processing")
    if args.skip_spharm or config.SKIP_SPHARM:
        logger.info("Skipping SPHARM-PDM step as per configuration or command-line flag.")
    else:
        use_registered_masks_for_spharm = not (args.skip_registration or config.SKIP_REGISTRATION)
        logger.info(f"SPHARM-PDM will use {'registered' if use_registered_masks_for_spharm else 'binarized'} masks as input.")

        if config.SIDE == "both" or config.SIDE == "left":
            logger.info("Processing left hemisphere for SPHARM-PDM...")
            run_spharm_pdm(side="left", use_registration=use_registered_masks_for_spharm, headless=not args.with_gui)
        
        if config.SIDE == "both" or config.SIDE == "right":
            logger.info("Processing right hemisphere for SPHARM-PDM...")
            run_spharm_pdm(side="right", use_registration=use_registered_masks_for_spharm, headless=not args.with_gui)
    
    print_section_header("[6/8] Preparing for statistical analysis")
    static_left_csv_path = config.STATIC_LEFT_CSV
    static_right_csv_path = config.STATIC_RIGHT_CSV
    
    if not (os.path.exists(static_left_csv_path) and os.path.exists(static_right_csv_path)):
        logger.info("One or both CSV files for static analysis may need to be created/populated.")
        logger.info(f"Expected Left CSV: {static_left_csv_path}")
        logger.info(f"Expected Right CSV: {static_right_csv_path}")
        logger.info("You can use 'python main.py utils dir2csv --side both' to create base CSV files from SPHARM model directories.")
        logger.info("After creation, please ensure these CSVs contain necessary subject IDs and covariates.")
        
        if is_interactive_mode:
            answer = input("Do you want to attempt to create base CSV files for static analysis now? (y/n): ").strip().lower()
            if answer == 'y':
                logger.info("Attempting to create base CSV files for static analysis...")
                dir_to_csv(side="left") 
                dir_to_csv(side="right") 
                logger.info(f"Base CSV files might have been created at {static_left_csv_path} and {static_right_csv_path}. Please verify and add covariates.")
            else:
                logger.info("Skipping automatic creation of base CSV files for static analysis.")
    else:
        logger.info(f"Static analysis CSV files found: {static_left_csv_path}, {static_right_csv_path}")

    print_section_header("[7/8] Running static statistical analysis")
    if args.skip_stats or config.SKIP_STATS:
        logger.info("Skipping static statistical analysis as per configuration or command-line flag.")
    else:
        if not (os.path.exists(static_left_csv_path) and os.path.exists(static_right_csv_path)):
            logger.warning("Cannot run static analysis: one or both CSV files are missing.")
            if is_interactive_mode:
                answer = input("Do you want to continue to longitudinal analysis (if applicable)? (y/n): ").strip().lower()
                if answer != 'y':
                    logger.info("Exiting pipeline as per user choice.")
                    return
        else:
            if config.SIDE == "both" or config.SIDE == "left":
                logger.info("Running static analysis for left hemisphere...")
                run_static_analysis(side="left") 
            
            if config.SIDE == "both" or config.SIDE == "right":
                logger.info("Running static analysis for right hemisphere...")
                run_static_analysis(side="right") 
    
    print_section_header("[8/8] Running longitudinal analysis")
    if args.skip_longitudinal or config.SKIP_LONGITUDINAL:
        logger.info("Skipping longitudinal analysis as per configuration or command-line flag.")
    else:
        longit_left_m2md_csv = config.LONGITUDINAL_LEFT_M2MD_CSV 
        longit_right_m2md_csv = config.LONGITUDINAL_RIGHT_M2MD_CSV
        
        if not (os.path.exists(longit_left_m2md_csv) and os.path.exists(longit_right_m2md_csv)):
            logger.warning("CSV files defining pairs for longitudinal M2MD analysis not found.")
            logger.warning(f"Please ensure {longit_left_m2md_csv} and {longit_right_m2md_csv} exist and are populated.")
            logger.warning("These files should typically have columns like 'Timepoint1_Model', 'Timepoint2_Model', 'Output_DeltaModel_Name'.")
        else:
            if config.SIDE == "both" or config.SIDE == "left":
                logger.info("Preparing longitudinal data (e.g., M2MD) for left hemisphere...")
                prepare_longitudinal_data(side="left") 
            
            if config.SIDE == "both" or config.SIDE == "right":
                logger.info("Preparing longitudinal data (e.g., M2MD) for right hemisphere...")
                prepare_longitudinal_data(side="right") 

            longit_stats_left_csv_path = config.LONGITUDINAL_STATS_LEFT_CSV
            longit_stats_right_csv_path = config.LONGITUDINAL_STATS_RIGHT_CSV

            if not (os.path.exists(longit_stats_left_csv_path) and os.path.exists(longit_stats_right_csv_path)):
                logger.warning("CSV files for longitudinal statistical analysis not found.")
                logger.warning(f"Please ensure {longit_stats_left_csv_path} and {longit_stats_right_csv_path} exist (e.g. output from M2MD + covariates).")
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
    """Run the segmentation component."""
    logger.info("Running standalone segmentation...")
    # Corrected keyword argument
    segment_hippocampus(input_dir_param=args.input_dir)

def run_binarization_cmd(args: argparse.Namespace):
    """Run the binarization component."""
    logger.info("Running standalone binarization...")
    binarize_masks()

def run_registration_cmd(args: argparse.Namespace):
    """Run the registration component."""
    logger.info("Running standalone registration...")
    register_masks()

def run_spharm_cmd(args: argparse.Namespace):
    """Run the SPHARM-PDM component."""
    logger.info(f"Running standalone SPHARM-PDM for side: {args.side}...")
    use_reg_flag = not args.use_binarized_input if hasattr(args, 'use_binarized_input') else True

    if args.side in ["left", "both"]:
        logger.info("Processing left hemisphere for SPHARM-PDM...")
        run_spharm_pdm(side="left", use_registration=use_reg_flag, headless=not args.with_gui)
    
    if args.side in ["right", "both"]:
        logger.info("Processing right hemisphere for SPHARM-PDM...")
        run_spharm_pdm(side="right", use_registration=use_reg_flag, headless=not args.with_gui)

def run_static_stats_cmd(args: argparse.Namespace):
    """Run the static statistical analysis component."""
    logger.info(f"Running standalone static statistical analysis for side: {args.side}...")
    if args.side in ["left", "both"]:
        logger.info("Running static analysis for left hemisphere...")
        run_static_analysis(side="left")
    
    if args.side in ["right", "both"]:
        logger.info("Running static analysis for right hemisphere...")
        run_static_analysis(side="right")

def run_longitudinal_cmd(args: argparse.Namespace):
    """Run the longitudinal analysis component."""
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
    """Run utility functions."""
    logger.info(f"Running utility: {args.util_choice}...")
    if args.util_choice == "dir2csv":
        dir_to_csv(side=args.side) 
        logger.info(f"dir2csv utility completed for side: {args.side}.")
        if args.side == "both":
            logger.info(f"Check for CSVs: {config.STATIC_LEFT_CSV} and {config.STATIC_RIGHT_CSV}")
        elif args.side == "left":
            logger.info(f"Check for CSV: {config.STATIC_LEFT_CSV}")
        else: # right
            logger.info(f"Check for CSV: {config.STATIC_RIGHT_CSV}")

    elif args.util_choice == "check_files":
        if not args.csv_file:
            logger.error("Please specify a CSV file with --csv-file for check_files utility.")
            return
        missing_files = check_files(csv_file_path=args.csv_file)
        if missing_files:
            logger.warning(f"Missing Files ({len(missing_files)}):")
            for file_path in missing_files:
                print(f"  - {file_path}")
        else:
            logger.info(f"All files listed in {args.csv_file} exist.")
            
    elif args.util_choice == "excel2csv":
        if not args.excel_file:
            logger.error("Please specify an Excel file with --excel-file for excel2csv utility.")
            return
        output_csv_paths = excel_to_csv(excel_file_path=args.excel_file) 
        if isinstance(output_csv_paths, str): 
             logger.info(f"Excel sheets converted to CSV files in directory: {output_csv_paths}")
        elif isinstance(output_csv_paths, list):
             logger.info(f"Excel sheets converted to CSV files: {', '.join(output_csv_paths)}")
        else:
             logger.info(f"excel2csv utility completed.")


def main():
    """Parse arguments and run the requested command."""
    parser = argparse.ArgumentParser(
        description="Hippocampal Morphometry Pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--non-interactive", action="store_true", default=config.NON_INTERACTIVE,
                        help="Run without interactive prompts, using defaults or failing if input is missing.")
    parser.add_argument("--with-gui", action="store_true", default=config.WITH_GUI,
                        help="Attempt to run SPHARM-PDM with its GUI (not headless). May not apply to all steps.")

    subparsers = parser.add_subparsers(dest="command", help="Command to run", required=True)
    
    pipeline_parser = subparsers.add_parser("pipeline", help="Run the complete pipeline.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    pipeline_parser.add_argument("--input-dir", type=str, default=None,
                                 help=f"Directory containing T1 MRI images. Defaults to config.T1_INPUT_DIR ('{config.T1_INPUT_DIR}').")
    pipeline_parser.add_argument("--skip-env-check", action="store_true", help="Skip environment check and proceed even if issues are found.")
    pipeline_parser.add_argument("--skip-segmentation", action="store_true", help="Skip segmentation step.")
    pipeline_parser.add_argument("--skip-binarization", action="store_true", help="Skip binarization step.")
    pipeline_parser.add_argument("--skip-registration", action="store_true", help="Skip registration step.")
    pipeline_parser.add_argument("--skip-spharm", action="store_true", help="Skip SPHARM-PDM processing.")
    pipeline_parser.add_argument("--skip-stats", action="store_true", help="Skip static statistical analysis.")
    pipeline_parser.add_argument("--skip-longitudinal", action="store_true", help="Skip longitudinal analysis.")
    pipeline_parser.set_defaults(func=run_full_pipeline)
    
    seg_parser = subparsers.add_parser("segment", help="Run hippocampal segmentation only.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    seg_parser.add_argument("--input-dir", type=str, default=None,
                            help=f"Directory containing T1 MRI images. Defaults to config.T1_INPUT_DIR ('{config.T1_INPUT_DIR}').")
    seg_parser.set_defaults(func=run_segmentation_cmd)
    
    bin_parser = subparsers.add_parser("binarize", help="Binarize segmentation masks only.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    bin_parser.set_defaults(func=run_binarization_cmd)
    
    reg_parser = subparsers.add_parser("register", help="Register binarized masks only.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    reg_parser.set_defaults(func=run_registration_cmd)
    
    spharm_parser = subparsers.add_parser("spharm", help="Run SPHARM-PDM processing only.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    spharm_parser.add_argument("--side", choices=["left", "right", "both"], default=config.SIDE, help="Hemisphere to process.")
    spharm_parser.add_argument("--use-binarized-input", action="store_true", help="Use binarized masks as input instead of registered masks for SPHARM.")
    spharm_parser.set_defaults(func=run_spharm_cmd) 
    
    stats_parser = subparsers.add_parser("static-stats", help="Run static statistical analysis only.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    stats_parser.add_argument("--side", choices=["left", "right", "both"], default=config.SIDE, help="Hemisphere to process.")
    stats_parser.set_defaults(func=run_static_stats_cmd)
    
    longit_parser = subparsers.add_parser("longitudinal", help="Run longitudinal analysis (data prep & optionally stats).", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    longit_parser.add_argument("--side", choices=["left", "right", "both"], default=config.SIDE, help="Hemisphere to process.")
    longit_parser.add_argument("--run-stats", action="store_true", help="Also run longitudinal statistical analysis after data preparation.")
    longit_parser.set_defaults(func=run_longitudinal_cmd)
    
    utils_parser = subparsers.add_parser("utils", help="Run utility functions.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    utils_parser.add_argument("util_choice", choices=["dir2csv", "check_files", "excel2csv"], help="Utility to run.")
    utils_parser.add_argument("--side", choices=["left", "right", "both"], default="both", help="Hemisphere for dir2csv utility.")
    utils_parser.add_argument("--csv-file", type=str, help="Path to CSV file (for check_files).")
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
