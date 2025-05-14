"""
Preprocessing Module
-------------------
Handles hippocampal segmentation, binarization, and registration.
"""

import os
import glob
import shutil
import subprocess
# import tkinter as tk # Tkinter is no longer used
# from tkinter import filedialog, messagebox # Tkinter is no longer used
import logging
import sys
from typing import Dict, List, Optional, Union, Tuple

import config
from utils import run_command, get_file_paths_by_pattern, normalize_path_for_platform, print_section_header

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def segment_hippocampus(input_dir_param: Optional[str] = None, 
                        output_dir_param: Optional[str] = None, 
                        hippodeep_path_param: Optional[str] = None) -> Dict[str, str]:
    """
    Segment the hippocampus from T1-weighted MRI images using hippodeep_pytorch.
    Input directory is taken from config.T1_INPUT_DIR by default but can be overridden.
    
    Parameters:
    -----------
    input_dir_param : str, optional
        Directory containing T1-weighted MRI images in .nii or .nii.gz format.
        If None, config.T1_INPUT_DIR will be used.
    output_dir_param : str, optional
        Directory where segmentation results will be saved.
        If None, config.SEG_OUTPUT_ROOT_DIR will be used.
    hippodeep_path_param : str, optional
        Path to the hippodeep_pytorch deepseg1.sh script.
        If None, config.HIPPODEEP_PATH will be used.
    
    Returns:
    --------
    dict
        Dictionary of output directories, or empty dict on failure.
    """
    hippodeep_path = hippodeep_path_param or config.HIPPODEEP_PATH
    output_dir = output_dir_param or config.SEG_OUTPUT_ROOT_DIR 
    input_dir = input_dir_param or config.T1_INPUT_DIR 

    print_section_header("HIPPOCAMPAL SEGMENTATION")
    
    if not input_dir:
        logger.error("Input directory for T1 images is not specified in arguments or config.T1_INPUT_DIR.")
        return {}
    if not os.path.isdir(input_dir):
        logger.error(f"Specified input directory does not exist: {input_dir}")
        return {}
        
    logger.info(f"Using input directory for T1 images: {input_dir}")

    t1_nii_files = get_file_paths_by_pattern(input_dir, "*.nii") + \
                   get_file_paths_by_pattern(input_dir, "*.nii.gz")
    
    if not t1_nii_files:
        logger.error(f"No .nii or .nii.gz files found in {input_dir}. Exiting...")
        return {}
    
    logger.info(f"Found the following T1 MRI files in {input_dir} to process:")
    normalized_t1_file_paths = []
    for nii_file_path in t1_nii_files:
        normalized_path = normalize_path_for_platform(nii_file_path)
        normalized_t1_file_paths.append(normalized_path)
        logger.info(f"  - {normalized_path}")
    
    logger.info("Proceeding with the hippodeep_pytorch segmentation. This may take a while...")
    
    output_subdirs = {
        "left_masks": config.SEG_LEFT_MASKS_DIR, 
        "right_masks": config.SEG_RIGHT_MASKS_DIR,
        "cerebrum_masks": config.SEG_CEREBRUM_MASKS_DIR,
        "brain_masks": config.SEG_BRAIN_MASKS_DIR,
        "volume_reports": config.SEG_VOLUME_REPORTS_DIR
    }
    
    for subdir_name, subdir_path in output_subdirs.items():
        try:
            os.makedirs(subdir_path, exist_ok=True)
            logger.info(f"Ensured directory exists: {subdir_path}")
        except OSError as e:
            logger.error(f"Could not create directory {subdir_path}: {e}")
            return {} 
            
    try:
        if not os.access(hippodeep_path, os.X_OK):
            logger.warning(f"Hippodeep script at {hippodeep_path} may not be executable. Attempting to run with 'bash'.")

        # Construct the command to pass individual files to deepseg1.sh
        # The deepseg1.sh script should be able to handle multiple file paths as arguments.
        hippodeep_command_list = ["bash", hippodeep_path] + normalized_t1_file_paths
        
        logger.info(f"Preparing to run hippodeep command with {len(normalized_t1_file_paths)} file(s).")
        logger.debug(f"Full command: {' '.join(hippodeep_command_list)}") # Log full command for debugging
        
        result = run_command(
            hippodeep_command_list, 
            description=f"Running hippodeep_pytorch segmentation on {len(normalized_t1_file_paths)} file(s)",
            check=False 
        )
        
        if result.returncode != 0:
            logger.error(f"Error running hippodeep segmentation. Return code: {result.returncode}")
            logger.error(f"Stdout: {result.stdout}")
            logger.error(f"Stderr: {result.stderr}")
            print("\nHippodeep segmentation failed. Check the logs for details.")
            return {}
            
        logger.info("Hippodeep_pytorch segmentation completed successfully")
        if result.stdout:
            logger.info(f"Hippodeep STDOUT:\n{result.stdout}")
        if result.stderr: 
            logger.warning(f"Hippodeep STDERR:\n{result.stderr}")
            
    except FileNotFoundError:
        logger.error(f"The hippodeep script was not found at {hippodeep_path} or 'bash' is not available.")
        return {}
    except Exception as e:
        logger.error(f"An unexpected error occurred while running the hippodeep script: {str(e)}")
        return {}
        
    # Hippodeep typically outputs files into the same directory as the input T1s,
    # or into a subdirectory named after the input file.
    # We will search for output masks in the original input_dir.
    move_hippodeep_files(source_dir=input_dir, target_output_subdirs=output_subdirs) 
    
    print_section_header("SEGMENTATION COMPLETE")
    logger.info("The segmentation process is complete and all files have been placed into their respective folders.")
    
    return output_subdirs

def move_hippodeep_files(source_dir: str, target_output_subdirs: Dict[str, str]) -> None:
    """
    Move hippodeep output files from a source directory to their respective target subdirectories.
    
    Parameters:
    -----------
    source_dir : str
        Directory where hippodeep generated the output files (e.g., the T1 input directory).
    target_output_subdirs : dict
        Dictionary with keys like 'left_masks' and values as absolute paths to target directories
        (e.g., config.SEG_LEFT_MASKS_DIR).
    """
    patterns = {
        "left_masks": "*mask_L.nii.gz", # Hippodeep output pattern for left mask
        "right_masks": "*mask_R.nii.gz",# Hippodeep output pattern for right mask
        "cerebrum_masks": "*cerebrum_mask.nii.gz", 
        "brain_masks": "*brain_mask.nii.gz",       
        "volume_reports": "*.csv" # Hippodeep output pattern for volume reports
        # Add other patterns if hippodeep generates more file types to be moved
    }
    
    logger.info(f"Attempting to move hippodeep output files from source: {source_dir}")

    # Also check for files in subdirectories named after the input files, as some tools do this.
    # Example: if input is source_dir/subjectA.nii, output might be source_dir/subjectA/subjectA_mask_L.nii.gz
    potential_source_dirs = [source_dir]
    for item in os.listdir(source_dir):
        item_path = os.path.join(source_dir, item)
        if os.path.isdir(item_path):
            potential_source_dirs.append(item_path)
    
    logger.info(f"Searching for output files in: {potential_source_dirs}")

    for dir_key, pattern in patterns.items():
        if dir_key in target_output_subdirs:
            target_dir_path = target_output_subdirs[dir_key] 
            
            files_found_for_pattern = []
            for s_dir in potential_source_dirs:
                files_found_for_pattern.extend(get_file_paths_by_pattern(s_dir, pattern))
            
            if not files_found_for_pattern:
                logger.warning(f"No files found matching pattern '{pattern}' in any potential source directories for '{dir_key}'.")
                continue

            logger.info(f"Found {len(files_found_for_pattern)} file(s) for pattern '{pattern}' to move to '{target_dir_path}'.")

            for file_path in files_found_for_pattern:
                try:
                    file_name = os.path.basename(file_path)
                    if not os.path.exists(target_dir_path):
                        logger.warning(f"Target directory {target_dir_path} does not exist. Attempting to create.")
                        os.makedirs(target_dir_path, exist_ok=True)

                    final_target_path = os.path.join(target_dir_path, file_name)
                    
                    if os.path.abspath(file_path) == os.path.abspath(final_target_path):
                        logger.info(f"File {file_name} is already in the target directory {target_dir_path}. Skipping move.")
                        continue

                    shutil.move(file_path, final_target_path)
                    logger.info(f"Moved {file_name} from {file_path} to {final_target_path}")
                except Exception as e:
                    logger.error(f"Error moving {file_path} to {target_dir_path}: {str(e)}")
        else:
            logger.warning(f"Directory key '{dir_key}' not found in target_output_subdirs. Skipping move for pattern '{pattern}'.")


def binarize_masks(left_masks_dir_param: Optional[str] = None,
                   right_masks_dir_param: Optional[str] = None,
                   left_output_dir_param: Optional[str] = None, 
                   right_output_dir_param: Optional[str] = None) -> bool:
    """
    Binarize hippocampal segmentation masks using FSL's fslmaths.
    """
    left_masks_dir = left_masks_dir_param or config.SEG_LEFT_MASKS_DIR
    right_masks_dir = right_masks_dir_param or config.SEG_RIGHT_MASKS_DIR
    left_output_dir = left_output_dir_param or config.BINARIZED_LEFT_MASKS_DIR
    right_output_dir = right_output_dir_param or config.BINARIZED_RIGHT_MASKS_DIR
    
    print_section_header("MASK BINARIZATION")
    logger.info("Attempting binarization via FSL's fslmaths.")
    
    try:
        result = run_command(["fslmaths", "-version"], check=False) 
        if result.returncode != 0 : 
            result_which = run_command(["which", "fslmaths"], check=False)
            if result_which.returncode != 0:
                logger.error("FSL's fslmaths not found in PATH. Cannot binarize masks.")
                return False
            logger.info("fslmaths found via 'which'. Proceeding.")
        else:
            logger.info("fslmaths found and responded to -version. Proceeding.")
    except FileNotFoundError: 
        logger.error("FSL's fslmaths command not found. Ensure FSL is installed and in PATH.")
        return False
    except Exception as e:
        logger.error(f"Error checking for fslmaths: {str(e)}")
        return False
        
    try:
        os.makedirs(left_output_dir, exist_ok=True)
        logger.info(f"Ensured output directory exists: {left_output_dir}")
        os.makedirs(right_output_dir, exist_ok=True)
        logger.info(f"Ensured output directory exists: {right_output_dir}")
    except OSError as e:
        logger.error(f"Could not create output directories for binarized masks: {e}")
        return False
        
    left_masks = get_file_paths_by_pattern(left_masks_dir, "*.nii.gz")
    right_masks = get_file_paths_by_pattern(right_masks_dir, "*.nii.gz")
    
    if not left_masks and not right_masks:
        logger.warning(f"No masks found to binarize in {left_masks_dir} or {right_masks_dir}")
        if getattr(config, 'FAIL_PIPELINE_ON_EMPTY_INPUT_MASKS', False): 
            logger.error("Configuration set to fail if no input masks are found for binarization.")
            return False
        return True 

    all_successful = True
    
    if left_masks:
        logger.info(f"Beginning binarization for {len(left_masks)} left hippocampal masks from {left_masks_dir}")
        for mask_path in left_masks:
            if not binarize_single_mask(mask_path, left_output_dir):
                all_successful = False 
    
    if right_masks:
        logger.info(f"Beginning binarization for {len(right_masks)} right hippocampal masks from {right_masks_dir}")
        for mask_path in right_masks:
            if not binarize_single_mask(mask_path, right_output_dir):
                all_successful = False 
                
    if all_successful:
        logger.info("All binarizations completed successfully.")
    else:
        logger.warning("Some binarizations failed. Check the logs for details.")
    
    print_section_header("BINARIZATION COMPLETE")
    return all_successful

def binarize_single_mask(mask_path: str, output_dir: str) -> bool:
    """
    Binarize a single mask using fslmaths.
    """
    try:
        mask_basename = os.path.basename(mask_path)
        if mask_basename.endswith(".nii.gz"):
            name_part = mask_basename[:-7] 
        elif mask_basename.endswith(".nii"):
            name_part = mask_basename[:-4]
        else:
            name_part = os.path.splitext(mask_basename)[0]
            
        output_filename = f'{name_part}_bin.nii.gz' 
        output_path = os.path.join(output_dir, output_filename)
        
        fslmaths_command_list = ['fslmaths', mask_path, '-bin', output_path]
        
        logger.info(f"Binarizing {mask_basename} -> {output_filename}")
        result = run_command(
            fslmaths_command_list,
            description=f"Binarizing {mask_basename}",
            check=False 
        )
        
        if result.returncode == 0:
            logger.info(f'Binarization successful for {mask_path}, output: {output_path}')
            return True
        else:
            logger.error(f'Error during binarization of {mask_path}. Return code: {result.returncode}')
            logger.error(f'FSLmaths STDOUT: {result.stdout}')
            logger.error(f'FSLmaths STDERR: {result.stderr}')
            return False
            
    except Exception as e:
        logger.error(f'An unexpected error occurred during binarization of {mask_path}: {str(e)}')
        return False

def register_masks(left_masks_dir_param: Optional[str] = None,
                   right_masks_dir_param: Optional[str] = None,
                   left_output_dir_param: Optional[str] = None,
                   right_output_dir_param: Optional[str] = None,
                   left_reference_param: Optional[str] = None,
                   right_reference_param: Optional[str] = None,
                   flirt_options_param: Optional[str] = None) -> bool:
    """
    Register hippocampal masks to reference templates using FSL FLIRT.
    """
    left_masks_dir = left_masks_dir_param or config.BINARIZED_LEFT_MASKS_DIR
    right_masks_dir = right_masks_dir_param or config.BINARIZED_RIGHT_MASKS_DIR
    left_output_dir = left_output_dir_param or config.REGISTERED_LEFT_MASKS_DIR
    right_output_dir = right_output_dir_param or config.REGISTERED_RIGHT_MASKS_DIR
    left_reference = left_reference_param or config.LEFT_REFERENCE_IMAGE
    right_reference = right_reference_param or config.RIGHT_REFERENCE_IMAGE
    flirt_options_str = flirt_options_param or config.FLIRT_OPTIONS 
    
    print_section_header("MASK REGISTRATION")
    logger.info("Attempting rigid (affine) alignment via FSL FLIRT.")
    
    try:
        result = run_command(["flirt", "-version"], check=False) 
        if result.returncode != 0: 
            result_which = run_command(["which", "flirt"], check=False)
            if result_which.returncode != 0:
                logger.error("FSL's flirt not found in PATH. Cannot register masks.")
                return False
            logger.info("flirt found via 'which'. Proceeding.")
        else:
            logger.info("flirt found and responded to -version. Proceeding.")
    except FileNotFoundError:
        logger.error("FSL's flirt command not found. Ensure FSL is installed and in PATH.")
        return False
    except Exception as e:
        logger.error(f"Error checking for flirt: {str(e)}")
        return False
        
    try:
        os.makedirs(left_output_dir, exist_ok=True)
        logger.info(f"Ensured output directory exists: {left_output_dir}")
        os.makedirs(right_output_dir, exist_ok=True)
        logger.info(f"Ensured output directory exists: {right_output_dir}")
    except OSError as e:
        logger.error(f"Could not create output directories for registered masks: {e}")
        return False
        
    if not os.path.exists(left_reference):
        logger.error(f"Left reference image not found: {left_reference}")
        return False
    if not os.path.exists(right_reference):
        logger.error(f"Right reference image not found: {right_reference}")
        return False
        
    left_masks = get_file_paths_by_pattern(left_masks_dir, "*.nii.gz")
    right_masks = get_file_paths_by_pattern(right_masks_dir, "*.nii.gz")
    
    if not left_masks and not right_masks:
        logger.warning(f"No masks found to register in {left_masks_dir} or {right_masks_dir}")
        if getattr(config, 'FAIL_PIPELINE_ON_EMPTY_INPUT_MASKS', False): 
            logger.error("Configuration set to fail if no input masks are found for registration.")
            return False
        return True 

    all_successful = True
    
    if left_masks:
        logger.info(f"Beginning alignment for {len(left_masks)} left hippocampal masks from {left_masks_dir}")
        for mask_path in left_masks:
            if not register_single_mask(mask_path, left_reference, left_output_dir, flirt_options_str):
                all_successful = False
    
    if right_masks:
        logger.info(f"Beginning alignment for {len(right_masks)} right hippocampal masks from {right_masks_dir}")
        for mask_path in right_masks:
            if not register_single_mask(mask_path, right_reference, right_output_dir, flirt_options_str):
                all_successful = False
                
    if all_successful:
        logger.info("All registrations completed successfully.")
    else:
        logger.warning("Some registrations failed. Check the logs for details.")
        
    print_section_header("REGISTRATION COMPLETE")
    return all_successful

def register_single_mask(mask_path: str, reference_image: str, 
                         output_dir: str, flirt_options_str: str) -> bool:
    """
    Register a single mask using FLIRT.
    """
    try:
        mask_basename = os.path.basename(mask_path)
        if mask_basename.endswith(".nii.gz"):
            name_part = mask_basename[:-7]
        elif mask_basename.endswith(".nii"):
            name_part = mask_basename[:-4]
        else:
            name_part = os.path.splitext(mask_basename)[0]
            
        output_filename = f'{name_part}_reg.nii.gz' 
        output_path = os.path.join(output_dir, output_filename)
        
        flirt_command_list = ['flirt', '-in', mask_path, '-ref', reference_image, '-out', output_path] + flirt_options_str.split()
        
        logger.info(f"Registering {mask_basename} -> {output_filename} using reference {os.path.basename(reference_image)}")
        result = run_command(
            flirt_command_list,
            description=f"Registering {mask_basename}",
            check=False 
        )
        
        if result.returncode == 0:
            logger.info(f'Registration successful for {mask_path}, output: {output_path}')
            return True
        else:
            logger.error(f'Error during registration of {mask_path}. Return code: {result.returncode}')
            logger.error(f'FLIRT STDOUT: {result.stdout}')
            logger.error(f'FLIRT STDERR: {result.stderr}')
            return False
            
    except Exception as e:
        logger.error(f'An unexpected error occurred during registration of {mask_path}: {str(e)}')
        return False
