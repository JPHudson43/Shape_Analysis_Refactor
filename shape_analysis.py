"""
Shape Analysis Module
--------------------
Handles SPHARM-PDM processing and shape model generation.
"""

import os
import glob
import shutil
import subprocess
import configparser
import tempfile
import logging
from typing import Dict, Optional, List, Tuple, Union

import config
from utils import run_command, get_file_paths_by_pattern, print_section_header

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_spharm_pdm(side: str = "left",
                   use_registration: bool = True,
                   headless: bool = True) -> str:
    """
    Run SPHARM-PDM processing on hippocampal masks.

    Parameters:
    -----------
    side : str
        Hemisphere to process ("left" or "right").
    use_registration : bool
        Whether to use registered masks (True) or binarized masks (False).
    headless : bool
        Whether to run in headless mode.

    Returns:
    --------
    str
        Path to the SPHARM tool output directory, or empty string on failure.
    """
    print_section_header(f"SPHARM-PDM PROCESSING: {side.upper()} HEMISPHERE")

    if not os.path.exists(config.SLICERSALT_PATH):
        logger.error(f"SlicerSALT not found at {config.SLICERSALT_PATH}")
        return ""

    if use_registration:
        spharm_input_component_type = "reg_mask_input" 
        logger.info(f"SPHARM-PDM will use registered masks for {side} hemisphere.")
    else:
        spharm_input_component_type = "bin_mask_input" 
        logger.info(f"SPHARM-PDM will use binarized masks for {side} hemisphere (registration likely skipped or not used).")
    
    try:
        input_dir_for_spharm = config.get_paths_by_side(side, spharm_input_component_type)
        spharm_tool_output_dir = config.get_paths_by_side(side, "spharm_tool_output")
        sphere_template_path = config.get_paths_by_side(side, "sphere_template")
        flip_template_path = config.get_paths_by_side(side, "flip_template")
        extracted_models_target_dir = config.get_paths_by_side(side, "spharm_extracted_models")

        logger.info(f"Using sphere template: {sphere_template_path}")
        logger.info(f"Using flip template: {flip_template_path}")

    except ValueError as e: 
        logger.error(f"Configuration error fetching paths for SPHARM ({side}, component type used was '{spharm_input_component_type}' or others): {e}")
        return ""
    except AttributeError as e: 
        logger.error(f"Configuration error: {e}. A required path variable might be missing in config.py.")
        return ""

    if not os.path.exists(input_dir_for_spharm):
        logger.error(f"Input directory for SPHARM not found: {input_dir_for_spharm}")
        return ""

    mask_files = get_file_paths_by_pattern(input_dir_for_spharm, "*.nii.gz")
    logger.info(f"Found {len(mask_files)} mask file(s) in {input_dir_for_spharm} for SPHARM processing.")
    if not mask_files:
        logger.error(f"No mask files (*.nii.gz) found in {input_dir_for_spharm} for SPHARM processing.")
        if getattr(config, 'FAIL_PIPELINE_ON_EMPTY_INPUT_MASKS', False):
            logger.error("Configuration set to fail if no input masks are found for SPHARM.")
            return ""
        logger.warning("Proceeding without SPHARM input files, SPHARM step will likely do nothing.")
        # SPHARM-PDM.py might still run and create an empty output structure or error out.
        # If it errors out due to no input, the run_command result will capture that.

    os.makedirs(spharm_tool_output_dir, exist_ok=True)
    logger.info(f"SPHARM-PDM tool output directory: {spharm_tool_output_dir}")

    ini_file_path = create_spharm_ini_file(
        input_dir=input_dir_for_spharm, 
        output_dir=spharm_tool_output_dir, 
        reg_template=sphere_template_path,
        flip_template=flip_template_path,
        params=config.SPHARM_PARAMS 
    )

    if not ini_file_path: 
        logger.error("Failed to create SPHARM-PDM parameter file.")
        return ""
    logger.info(f"Created SPHARM-PDM parameter file: {ini_file_path}")
    # For debugging, one might want to log the content of the INI file:
    # with open(ini_file_path, 'r') as f:
    # logger.debug(f"SPHARM INI file content:\n{f.read()}")


    try:
        spharm_command_list = [config.SLICERSALT_PATH]
        if headless:
            spharm_command_list.extend(["--no-main-window"])
        spharm_command_list.extend(["--python-script", config.SPHARM_PDM_PATH, ini_file_path])

        logger.info(f"Executing SPHARM command: {' '.join(spharm_command_list)}")

        result = run_command(
            spharm_command_list,
            description="Running SPHARM-PDM computation (this may take a while)",
            check=False
        )

        # Log stdout and stderr regardless of return code for better diagnostics
        if result.stdout:
            logger.info(f"SPHARM-PDM STDOUT:\n{result.stdout}")
        else:
            logger.info("SPHARM-PDM STDOUT: <No output>")
        
        if result.stderr:
            # SPHARM-PDM might use stderr for informational messages too
            logger.info(f"SPHARM-PDM STDERR:\n{result.stderr}") # Changed to INFO to always show it
        else:
            logger.info("SPHARM-PDM STDERR: <No output>")

        if result.returncode != 0:
            logger.error(f"SPHARM-PDM computation failed with return code {result.returncode}")
            # Detailed stdout/stderr are already logged above
            return ""

        logger.info("SPHARM-PDM computation completed successfully.")

    except FileNotFoundError as e: 
        logger.error(f"File not found during SPHARM-PDM execution: {e}. Check SlicerSALT_PATH and SPHARM_PDM_PATH.")
        return ""
    except Exception as e:
        logger.error(f"An unexpected error occurred during SPHARM-PDM computation: {str(e)}")
        logger.exception("Traceback for SPHARM-PDM unexpected error:") # Adds full traceback
        return ""
    finally:
        if ini_file_path and os.path.exists(ini_file_path): 
            try:
                os.remove(ini_file_path)
                logger.info(f"Removed temporary INI file: {ini_file_path}")
            except OSError as e:
                logger.warning(f"Could not remove temporary INI file {ini_file_path}: {e}")

    print_section_header(f"SPHARM-PDM COMPUTATION FOR {side.upper()} COMPLETE")

    extract_aligned_models(
        source_dir=spharm_tool_output_dir, 
        target_dir=extracted_models_target_dir, 
        alignment_type=config.ALIGNMENT_TYPE 
    )

    return spharm_tool_output_dir 

def create_spharm_ini_file(input_dir: str,
                           output_dir: str,
                           reg_template: str,
                           flip_template: str,
                           params: Dict[str, str]) -> str:
    """
    Create an INI file for SPHARM-PDM processing.
    """
    config_parser = configparser.ConfigParser()
    config_parser["DirectoryPath"] = {
        "inputDirectoryPath": input_dir,
        "outputDirectoryPath": output_dir
    }
    config_parser["SegPostProcess"] = {
        "rescale": params.get("rescale", "False"),
        "space": params.get("space", "0.5,0.5,0.5"),
        "label": params.get("label", "1"), 
        "gauss": params.get("gauss", "False"),
        "var": params.get("var", "10,10,10")
    }
    config_parser["GenParaMesh"] = {
        "iter": params.get("iter", "1000")
    }
    config_parser["ParaToSPHARMMesh"] = {
        "subdivLevel": params.get("subdivLevel", "10"),
        "spharmDegree": params.get("spharmDegree", "12"), 
        "medialMesh": params.get("medialMesh", "False"),
        "phiIteration": params.get("phiIteration", "100"),
        "thetaIteration": params.get("thetaIteration", "100"),
        "regParaTemplateFileOn": params.get("regParaTemplateFileOn", "True"),
        "regParaTemplate": reg_template,
        "flipTemplateOn": params.get("flipTemplateOn", "True"),
        "flipTemplate": flip_template,
        "flip": params.get("flip", "0") 
    }
    ini_file_path = ""
    try:
        fd, ini_file_path = tempfile.mkstemp(suffix='.ini', prefix='spharm_params_', text=True)
        with os.fdopen(fd, 'w') as ini_file: 
            config_parser.write(ini_file)
        logger.info(f"SPHARM INI content written to: {ini_file_path}")
        return ini_file_path
    except IOError as e:
        logger.error(f"Failed to create or write SPHARM INI file: {e}")
        if ini_file_path and os.path.exists(ini_file_path): 
            os.remove(ini_file_path)
        return "" 
    except Exception as e:
        logger.error(f"An unexpected error occurred creating INI file: {e}")
        if ini_file_path and os.path.exists(ini_file_path): 
            os.remove(ini_file_path)
        return ""

def extract_aligned_models(source_dir: str,
                           target_dir: str,
                           alignment_type: Optional[str] = None) -> int:
    """
    Extract aligned models from SPHARM-PDM output to a separate directory.
    """
    current_alignment_type = alignment_type or config.ALIGNMENT_TYPE
    if not current_alignment_type:
        logger.error("Alignment type for extraction is not specified in config or arguments.")
        return 0

    model_pattern_to_search = f"*_{current_alignment_type}.vtk"
    logger.info(f"Extracting aligned models matching pattern '{model_pattern_to_search}' from {source_dir} to {target_dir}")

    try:
        os.makedirs(target_dir, exist_ok=True)
        logger.info(f"Ensured target directory for extracted models exists: {target_dir}")
    except OSError as e:
        logger.error(f"Could not create target directory {target_dir}: {e}")
        return 0

    possible_step3_dirs = []
    direct_step3 = os.path.join(source_dir, "Step3_ParaToSPHARMMesh")
    if os.path.isdir(direct_step3):
        possible_step3_dirs.append(direct_step3)

    if os.path.isdir(source_dir): 
        for item in os.listdir(source_dir):
            item_path = os.path.join(source_dir, item)
            if os.path.isdir(item_path): 
                potential_subject_step3_path = os.path.join(item_path, "Step3_ParaToSPHARMMesh")
                if os.path.isdir(potential_subject_step3_path):
                    if potential_subject_step3_path not in possible_step3_dirs: 
                        possible_step3_dirs.append(potential_subject_step3_path)
    else:
        logger.warning(f"Source directory for SPHARM model extraction does not exist: {source_dir}")
        return 0
    
    if not possible_step3_dirs:
        logger.warning(f"No 'Step3_ParaToSPHARMMesh' directory found directly in {source_dir} or in its immediate subdirectories.")
        return 0
    
    logger.info(f"Found potential SPHARM output Step3 directories: {possible_step3_dirs}")

    all_models_found_to_copy = []
    for step3_dir_path in possible_step3_dirs:
        logger.debug(f"Searching for pattern '{model_pattern_to_search}' in {step3_dir_path}")
        models_in_this_step3_dir = get_file_paths_by_pattern(step3_dir_path, model_pattern_to_search)
        if models_in_this_step3_dir:
            logger.info(f"Found {len(models_in_this_step3_dir)} model(s) in {step3_dir_path}: {models_in_this_step3_dir}")
            all_models_found_to_copy.extend(models_in_this_step3_dir)
        else:
            logger.debug(f"No models matching pattern found in {step3_dir_path}")


    if not all_models_found_to_copy:
        logger.warning(f"No models found with pattern suffix '_{current_alignment_type}.vtk' in any identified Step3_ParaToSPHARMMesh directories.")
        return 0
    
    logger.info(f"Found a total of {len(all_models_found_to_copy)} SPHARM model(s) to potentially copy: {all_models_found_to_copy}")

    copied_models_count = 0
    for model_path in all_models_found_to_copy:
        model_name = os.path.basename(model_path)
        target_path = os.path.join(target_dir, model_name) 

        try:
            if os.path.exists(target_path):
                logger.info(f"Target file {target_path} already exists. Overwriting.")
            else:
                logger.info(f"Copying {model_name} from {model_path} to {target_path}")
            shutil.copy2(model_path, target_path)
            # logger.info(f"Successfully copied {model_name} to {target_path}") # Redundant if overwrite log is present
            copied_models_count += 1
        except Exception as e:
            logger.error(f"Error copying {model_path} to {target_path}: {str(e)}")

    if copied_models_count > 0:
        logger.info(f"Successfully extracted {copied_models_count} aligned models to {target_dir}")
    else:
        # This case should ideally not be reached if all_models_found_to_copy was not empty,
        # unless all copy operations failed.
        logger.warning(f"No models were successfully copied. Check logs for errors or pattern mismatches with ALIGNMENT_TYPE '{current_alignment_type}'.")
        
    return copied_models_count
