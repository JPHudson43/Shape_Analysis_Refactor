"""
Shape Analysis Module
--------------------
Handles SPHARM-PDM processing and shape model generation.
This version is adapted for processing mask directories directly
 (e.g., output from segmentation of one or more subjects).
"""

import os
import glob
import shutil
import subprocess
import configparser
import tempfile
import logging
import re 
import time # For manual timeout with Popen
from typing import Dict, Optional, List, Tuple, Union

import config
from utils import get_file_paths_by_pattern, print_section_header 

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_spharm_ini_file_dynamic(
    input_mask_dir: str, 
    output_processing_dir: str, 
    reg_template_path: str,
    flip_template_path: str,
    base_params: Dict[str, str],
    temp_ini_save_path: str) -> Optional[str]:
    """
    Creates a SPHARM INI file with dynamically set paths.
    The SPHARM-PDM tool is expected to process all compatible mask files
    found in the input_mask_dir.
    """
    config_parser = configparser.ConfigParser()
    config_parser["DirectoryPath"] = {
        "inputDirectoryPath": input_mask_dir, # Directory containing .nii.gz masks
        "outputDirectoryPath": output_processing_dir # Directory where SPHARM tool will create its subfolder structure
    }
    config_parser["SegPostProcess"] = {
        "rescale": base_params.get("rescale", "False"),
        "space": base_params.get("space", "0.5,0.5,0.5"),
        "label": base_params.get("label", "0"), 
        "gauss": base_params.get("gauss", "False"),
        "var": base_params.get("var", "10,10,10")
    }
    config_parser["GenParaMesh"] = {
        "iter": base_params.get("iter", "1000")
    }
    
    reg_on = base_params.get("regParaTemplateFileOn", "False").lower() == "true" and bool(reg_template_path)
    flip_on = base_params.get("flipTemplateOn", "False").lower() == "true" and bool(flip_template_path)

    config_parser["ParaToSPHARMMesh"] = {
        "subdivLevel": base_params.get("subdivLevel", "10"),
        "spharmDegree": base_params.get("spharmDegree", "15"), 
        "medialMesh": base_params.get("medialMesh", "True"),   
        "phiIteration": base_params.get("phiIteration", "100"),
        "thetaIteration": base_params.get("thetaIteration", "100"),
        "regParaTemplateFileOn": str(reg_on), 
        "regParaTemplate": reg_template_path if reg_on else "",
        "flipTemplateOn": str(flip_on),     
        "flipTemplate": flip_template_path if flip_on else "",
        "flip": base_params.get("flip", "0") 
    }
    
    try:
        os.makedirs(os.path.dirname(temp_ini_save_path), exist_ok=True)
        with open(temp_ini_save_path, 'w') as ini_file: 
            config_parser.write(ini_file)
        logger.info(f"SPHARM INI content written to: {temp_ini_save_path}")
        return temp_ini_save_path
    except IOError as e:
        logger.error(f"Failed to create or write SPHARM INI file {temp_ini_save_path}: {e}")
        return None
    except Exception as e: 
        logger.error(f"Unexpected error creating INI file {temp_ini_save_path}: {e}")
        return None

def run_spharm_for_case_with_popen(
    slicer_path: str,
    spharm_script_path: str,
    temp_ini_file: str,
    timeout_seconds: Optional[int],
    exit_phrase: str = "completed without errors", 
    headless: bool = True):
    """
    Runs a single SPHARM-PDM instance using subprocess.Popen,
    reading stdout for an exit phrase, and implementing a timeout.
    """
    spharm_command_list = [slicer_path]
    if headless:
        spharm_command_list.extend(["--no-main-window"])
    spharm_command_list.extend(["--python-script", spharm_script_path, temp_ini_file])

    logger.info(f"Executing SPHARM (Popen): {' '.join(spharm_command_list)}")
    logger.info(f"Watching for exit phrase: '{exit_phrase}'")
    if timeout_seconds:
        logger.info(f"Timeout set to: {timeout_seconds} seconds")

    process = None
    try:
        process = subprocess.Popen(
            spharm_command_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, 
            text=True,
            bufsize=1,  
            universal_newlines=True
        )

        start_time = time.time()
        log_buffer = [] 

        while True:
            if process.stdout:
                output_line = process.stdout.readline()
                if output_line:
                    line_stripped = output_line.strip()
                    if line_stripped: 
                        logger.info(f"[SPHARM_LOG] {line_stripped}") 
                        log_buffer.append(line_stripped)
                        if len(log_buffer) > 50: 
                            log_buffer.pop(0)
                    if exit_phrase in output_line:
                        logger.info(f"SPHARM exit phrase '{exit_phrase}' detected. Process likely complete.")
                        process.terminate() 
                        try:
                            process.wait(timeout=10) 
                        except subprocess.TimeoutExpired:
                            logger.warning("SPHARM process did not terminate gracefully after exit phrase, killing.")
                            process.kill()
                        logger.info(f"SPHARM completed successfully for {os.path.basename(temp_ini_file)} based on exit phrase.")
                        return True
                elif process.poll() is not None: 
                    logger.info(f"SPHARM process exited with code {process.returncode} before exit phrase was detected.")
                    if process.returncode == 0:
                         logger.warning(f"SPHARM process for {os.path.basename(temp_ini_file)} exited with 0, but exit phrase not seen. Assuming success with caution.")
                         return True
                    else:
                         logger.error(f"SPHARM failed for {os.path.basename(temp_ini_file)} with exit code {process.returncode}.")
                         return False
            
            if timeout_seconds and (time.time() - start_time) > timeout_seconds:
                logger.error(f"SPHARM timed out after {timeout_seconds} seconds for {os.path.basename(temp_ini_file)}.")
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning("SPHARM process did not terminate gracefully after timeout, killing.")
                    process.kill()
                return False

            if process.poll() is not None: 
                break 
            
            time.sleep(0.1) 

        if process.returncode == 0:
            logger.warning(f"SPHARM process for {os.path.basename(temp_ini_file)} exited with 0, but exit phrase not seen. Assuming success with caution.")
            return True
        else:
            logger.error(f"SPHARM failed for {os.path.basename(temp_ini_file)} with exit code {process.returncode or 'Unknown'}.")
            return False

    except FileNotFoundError:
        logger.error(f"SlicerSALT or SPHARM-PDM script not found. Command: {' '.join(spharm_command_list)}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred running SPHARM for {os.path.basename(temp_ini_file)}: {e}")
        logger.exception("Detailed traceback for SPHARM Popen error:")
        if process and process.poll() is None: 
            process.kill()
        return False
    finally:
        if process and process.poll() is None: 
            logger.warning(f"Ensuring SPHARM process for {os.path.basename(temp_ini_file)} is terminated due to exception in monitoring loop.")
            process.kill()

def run_spharm_pipeline_adapted():
    """
    Runs the SPHARM-PDM processing.
    It processes masks directly from segmentation output directories.
    """
    print_section_header("ADAPTED SPHARM-PDM PIPELINE INITIATED (WITH STRING CHECK)")

    if not os.path.exists(config.SLICERSALT_PATH):
        logger.error(f"SlicerSALT not found at {config.SLICERSALT_PATH}")
        return
    if not os.path.exists(config.SPHARM_PDM_PYTHON_SCRIPT_PATH):
        logger.error(f"SPHARM-PDM.py script not found at {config.SPHARM_PDM_PYTHON_SCRIPT_PATH}")
        return

    overall_success_tally = {"success": 0, "failed": 0, "skipped": 0}

    for side_label in ["left", "right"]:
        logger.info(f"Processing SPHARM for {side_label.upper()} hemisphere.")

        try:
            # Input for SPHARM is the output directory of segmentation masks for this side
            input_mask_dir_for_spharm = config.get_paths_by_side(side_label, "seg_mask_input")
            # Output directory where the SPHARM tool will write its own results (may create subdirs)
            output_processing_dir_for_spharm = config.get_paths_by_side(side_label, "spharm_tool_output")
            
            reg_template_path = config.get_paths_by_side(side_label, "sphere_template")
            flip_template_path = config.get_paths_by_side(side_label, "flip_template")
        except ValueError as e:
            logger.error(f"Configuration error getting paths for SPHARM (side: {side_label}): {e}")
            overall_success_tally["failed"] += 1
            continue

        if not os.path.isdir(input_mask_dir_for_spharm) or not any(f.endswith(('.nii', '.nii.gz')) for f in os.listdir(input_mask_dir_for_spharm)):
            logger.warning(f"Input mask directory for SPHARM is empty, does not exist, or contains no NIFTI files: {input_mask_dir_for_spharm}. Skipping {side_label} SPHARM.")
            overall_success_tally["skipped"] += 1
            continue
        
        os.makedirs(output_processing_dir_for_spharm, exist_ok=True)
        logger.info(f"Ensured SPHARM tool output directory exists: {output_processing_dir_for_spharm}")

        if config.SPHARM_PARAMS.get("regParaTemplateFileOn", "False").lower() == "true" and not os.path.exists(reg_template_path):
            logger.error(f"Registration template not found: {reg_template_path}, but regParaTemplateFileOn is True. Skipping SPHARM for {side_label}")
            overall_success_tally["failed"] += 1
            continue
        if config.SPHARM_PARAMS.get("flipTemplateOn", "False").lower() == "true" and not os.path.exists(flip_template_path):
            logger.error(f"Flip template not found: {flip_template_path}, but flipTemplateOn is True. Skipping SPHARM for {side_label}")
            overall_success_tally["failed"] += 1
            continue
        
        temp_ini_filename = f"spharm_params_{side_label}.ini"
        temp_ini_path = os.path.join(output_processing_dir_for_spharm, temp_ini_filename)

        logger.info(f"  Input SPHARM dir for {side_label} (contains masks): {input_mask_dir_for_spharm}")
        logger.info(f"  Output SPHARM tool dir for {side_label} (tool writes here): {output_processing_dir_for_spharm}")
        logger.info(f"  Reg Template: {reg_template_path if config.SPHARM_PARAMS.get('regParaTemplateFileOn','').lower() == 'true' else 'Not Used'}")
        logger.info(f"  Flip Template: {flip_template_path if config.SPHARM_PARAMS.get('flipTemplateOn','').lower() == 'true' else 'Not Used'}")

        current_ini_path = create_spharm_ini_file_dynamic(
            input_mask_dir=input_mask_dir_for_spharm, 
            output_processing_dir=output_processing_dir_for_spharm, 
            reg_template_path=reg_template_path,
            flip_template_path=flip_template_path,
            base_params=config.SPHARM_PARAMS,
            temp_ini_save_path=temp_ini_path
        )

        if not current_ini_path:
            logger.error(f"Failed to create INI for SPHARM processing of {input_mask_dir_for_spharm}. Skipping {side_label}.")
            overall_success_tally["failed"] += 1
            continue
        
        spharm_exit_phrase = getattr(config, "SPHARM_COMPLETION_PHRASE", "ParaToSPHARMMesh completed without errors")

        case_success = run_spharm_for_case_with_popen(
            slicer_path=config.SLICERSALT_PATH,
            spharm_script_path=config.SPHARM_PDM_PYTHON_SCRIPT_PATH,
            temp_ini_file=current_ini_path,
            timeout_seconds=config.SPHARM_TIMEOUT_SECONDS,
            exit_phrase=spharm_exit_phrase, 
            headless=not config.WITH_GUI 
        )
        if case_success:
            overall_success_tally["success"] += 1
        else:
            overall_success_tally["failed"] += 1
            logger.error(f"SPHARM-PDM failed for: Side: {side_label}, Input Masks: {input_mask_dir_for_spharm}")
        
        if case_success and os.path.exists(current_ini_path):
            try:
                os.remove(current_ini_path)
                logger.debug(f"Removed temp INI: {current_ini_path}")
            except OSError:
                logger.warning(f"Could not remove temp INI: {current_ini_path}")
    
    logger.info(f"SPHARM-PDM processing summary: {overall_success_tally['success']} succeeded, {overall_success_tally['failed']} failed, {overall_success_tally['skipped']} skipped.")
    
    expected_successes = 0
    if os.path.isdir(config.SEG_LEFT_MASKS_DIR) and any(f.endswith(('.nii', '.nii.gz')) for f in os.listdir(config.SEG_LEFT_MASKS_DIR)):
        expected_successes +=1
    if os.path.isdir(config.SEG_RIGHT_MASKS_DIR) and any(f.endswith(('.nii', '.nii.gz')) for f in os.listdir(config.SEG_RIGHT_MASKS_DIR)):
        expected_successes +=1

    if overall_success_tally["failed"] > 0 or (overall_success_tally["success"] < expected_successes and expected_successes > 0) :
        logger.warning("One or more SPHARM-PDM processing cases failed or not all expected cases were processed successfully.")
    else:
        logger.info("All SPHARM-PDM processing cases completed successfully or were appropriately skipped.")

    logger.info("Attempting to extract aligned models...")
    extract_aligned_models_simplified(
        spharm_tool_output_left_dir=config.SPHARM_TOOL_LEFT_OUTPUT_DIR,
        spharm_tool_output_right_dir=config.SPHARM_TOOL_RIGHT_OUTPUT_DIR,
        target_extracted_left_dir=config.SPHARM_EXTRACTED_MODELS_LEFT_DIR,
        target_extracted_right_dir=config.SPHARM_EXTRACTED_MODELS_RIGHT_DIR,
        alignment_type_suffix=config.ALIGNMENT_TYPE
    )
    print_section_header("ADAPTED SPHARM-PDM PIPELINE FINISHED")

def extract_aligned_models_simplified(spharm_tool_output_left_dir: str,
                                      spharm_tool_output_right_dir: str,
                                      target_extracted_left_dir: str,
                                      target_extracted_right_dir: str,
                                      alignment_type_suffix: str):
    """
    Extracts aligned models from the SPHARM tool's output directories.
    SPHARM-PDM.py typically creates subdirectories like:
    <outputDirectoryPath>/<InputFileName_NoExt>/Step3_ParaToSPHARMMesh/*.vtk
    This function will search for the desired VTK files within these structures.
    """
    logger.info(f"Starting simplified extraction of aligned models ('*{alignment_type_suffix}.vtk').")
    total_models_copied = 0

    for side_label, source_spharm_main_output_dir, target_extract_dir in [
        ("left", spharm_tool_output_left_dir, target_extracted_left_dir),
        ("right", spharm_tool_output_right_dir, target_extracted_right_dir)
    ]:
        logger.info(f"Extracting for {side_label} from SPHARM tool output: {source_spharm_main_output_dir} to target: {target_extract_dir}")
        os.makedirs(target_extract_dir, exist_ok=True)

        if not os.path.isdir(source_spharm_main_output_dir):
            logger.warning(f"SPHARM tool main output directory for {side_label} does not exist: {source_spharm_main_output_dir}. Cannot extract models.")
            continue
        
        model_pattern_to_find = f"*_{alignment_type_suffix}.vtk"
        found_models_for_side = []

        # SPHARM-PDM.py creates subdirectories based on input mask filenames.
        # e.g., if input mask was "SubjectA_mask_L.nii.gz", it creates "SubjectA_mask_L" subdir
        # within the outputDirectoryPath specified in the INI.
        for item_in_spharm_output in os.listdir(source_spharm_main_output_dir):
            potential_mask_output_folder = os.path.join(source_spharm_main_output_dir, item_in_spharm_output)
            if os.path.isdir(potential_mask_output_folder):
                # Look for the "Step3_ParaToSPHARMMesh" subdirectory
                step3_dir = os.path.join(potential_mask_output_folder, "Step3_ParaToSPHARMMesh")
                if os.path.isdir(step3_dir):
                    logger.debug(f"Searching for '{model_pattern_to_find}' in {step3_dir}")
                    for model_file in glob.glob(os.path.join(step3_dir, model_pattern_to_find)):
                        found_models_for_side.append(model_file)
                else: # Also check directly in potential_mask_output_folder for simpler SPHARM outputs
                    logger.debug(f"Searching for '{model_pattern_to_find}' in {potential_mask_output_folder} (Step3 not found)")
                    for model_file in glob.glob(os.path.join(potential_mask_output_folder, model_pattern_to_find)):
                        found_models_for_side.append(model_file)
            # Check also directly in source_spharm_main_output_dir if SPHARM output isn't nested per mask
            elif item_in_spharm_output.endswith(f"_{alignment_type_suffix}.vtk"): # Unlikely for SPHARM-PDM.py but as a fallback
                 if item_in_spharm_output.endswith(model_pattern_to_find.replace("*","")): # More specific check
                    found_models_for_side.append(os.path.join(source_spharm_main_output_dir, item_in_spharm_output))


        if found_models_for_side:
            logger.info(f"Found {len(found_models_for_side)} model(s) for side {side_label} in SPHARM output subdirectories.")
            for model_path in found_models_for_side:
                model_basename = os.path.basename(model_path)
                # For a single T1, the model_basename from SPHARM should be sufficiently unique.
                # It typically includes the original mask name.
                target_file_path = os.path.join(target_extract_dir, model_basename)
                
                try:
                    if os.path.exists(target_file_path):
                        logger.info(f"Target file {target_file_path} already exists. Overwriting.")
                    else:
                        logger.info(f"Copying {model_basename} from {model_path} to {target_file_path}")
                    shutil.copy2(model_path, target_file_path)
                    total_models_copied += 1
                except Exception as e:
                    logger.error(f"Error copying {model_path} to {target_file_path}: {e}")
        else:
            logger.warning(f"No models matching '{model_pattern_to_find}' found for side {side_label} in expected SPHARM output locations within {source_spharm_main_output_dir}")

    logger.info(f"Simplified extraction complete. Copied {total_models_copied} models.")