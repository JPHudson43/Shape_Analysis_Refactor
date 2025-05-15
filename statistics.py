"""
Statistics Module
---------------
Handles statistical analysis for static and longitudinal data.
"""

import os
import pandas as pd
import subprocess
import tempfile
import logging
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Optional, List, Union, Tuple

import config # Ensures MFSDA_TIMEOUT_SECONDS can be accessed
from utils import run_command, print_section_header 

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_static_analysis(side: str = "left") -> bool:
    """
    Run static statistical analysis on SPHARM models.
    
    Parameters:
    -----------
    side : str
        Hemisphere to process ("left" or "right").
        
    Returns:
    --------
    bool
        True if analysis was successful, False otherwise
    """
    print_section_header(f"STATIC STATISTICAL ANALYSIS: {side.upper()} HEMISPHERE")
    
    try:
        # This is the original CSV with SubjectPath, SubjectID, Covariates...
        original_csv_path = config.get_paths_by_side(side, "static_csv")
        output_dir = config.get_paths_by_side(side, "stats_static_output") 
        shape_template = config.get_paths_by_side(side, "sphere_template") 
        sphere_template_for_mfsda = config.get_paths_by_side(side, "sphere_template")
    except ValueError as e:
        logger.error(f"Configuration error getting paths for static analysis (side: {side}): {e}")
        return False
    except AttributeError as e: 
        logger.error(f"Configuration error: {e}. A required path variable might be missing in config.py.")
        return False

    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Ensured output directory for static stats exists: {output_dir}")
    except OSError as e:
        logger.error(f"Could not create output directory {output_dir}: {e}")
        return False
        
    if not os.path.exists(original_csv_path):
        logger.error(f"Static analysis CSV file not found: {original_csv_path}. This file should contain SubjectPath and covariates.")
        return False
    
    logger.info(f"Running static statistical analysis using base CSV: {original_csv_path}")
    
    try:
        df_original = pd.read_csv(original_csv_path)
        if df_original.empty:
            logger.error(f"Static analysis CSV file {original_csv_path} is empty. Cannot proceed.")
            return False
        if "SubjectPath" not in df_original.columns or df_original.columns[0] != "SubjectPath": 
            logger.error(f"Static analysis CSV file {original_csv_path} must contain 'SubjectPath' as the first column.")
            return False
        
        # Determine columns for MFSDA (numerical covariates after SubjectPath and SubjectID)
        if "SubjectID" not in df_original.columns or df_original.columns[1] != "SubjectID":
            logger.warning(f"Static analysis CSV file {original_csv_path} does not have 'SubjectID' as the second column. Assuming numerical covariates start from the second column.")
            cols_for_mfsda_data = list(df_original.columns[1:])
        else:
            subject_id_col_index = df_original.columns.get_loc('SubjectID')
            cols_for_mfsda_data = list(df_original.columns[subject_id_col_index + 1:])

        df_for_mfsda = df_original[['SubjectPath'] + cols_for_mfsda_data]
        
    except Exception as e:
        logger.error(f"Error reading or preparing data from static analysis CSV {original_csv_path}: {str(e)}")
        return False

    mfsda_run_path = getattr(config, "MFSDA_RUN_PATH", None)
    mfsda_createshapes_path = getattr(config, "MFSDA_CREATE_SHAPES_PATH", None)

    if not mfsda_run_path or not mfsda_createshapes_path:
        logger.error("MFSDA_RUN_PATH or MFSDA_CREATE_SHAPES_PATH not defined in config.py. Cannot run MFSDA.")
        return False
    if not os.path.exists(mfsda_run_path):
        logger.error(f"MFSDA_run.py script not found at: {mfsda_run_path}")
        return False
    if not os.path.exists(mfsda_createshapes_path):
        logger.error(f"MFSDA_createShapes.py script not found at: {mfsda_createshapes_path}")
        return False

    temp_csv_for_mfsda_fd = -1 
    temp_csv_for_mfsda_path = ""
    success = False
    try:
        temp_csv_for_mfsda_fd, temp_csv_for_mfsda_path = tempfile.mkstemp(suffix='.csv', prefix='mfsda_static_input_')
        logger.info(f"Creating temporary MFSDA-compatible CSV for static analysis: {temp_csv_for_mfsda_path}")
        df_for_mfsda.to_csv(temp_csv_for_mfsda_path, index=False)

        success = run_mfsda_subprocess( 
            csv_path_for_mfsda=temp_csv_for_mfsda_path, 
            output_dir=output_dir,
            shape_template=shape_template, 
            sphere_template_for_mfsda=sphere_template_for_mfsda, 
            mfsda_run_path=mfsda_run_path,
            mfsda_createshapes_path=mfsda_createshapes_path,
            slicersalt_path=config.SLICERSALT_PATH
        )
    except Exception as e_outer:
        logger.error(f"Outer error during static MFSDA setup or execution: {e_outer}")
        success = False
    finally:
        if temp_csv_for_mfsda_fd != -1:
            try:
                os.close(temp_csv_for_mfsda_fd)
            except OSError as e:
                logger.warning(f"Error closing temp file descriptor for {temp_csv_for_mfsda_path}: {e}")
        if temp_csv_for_mfsda_path and os.path.exists(temp_csv_for_mfsda_path):
            try:
                os.remove(temp_csv_for_mfsda_path)
                logger.info(f"Removed temporary MFSDA CSV: {temp_csv_for_mfsda_path}")
            except OSError as e:
                 logger.warning(f"Error removing temp file {temp_csv_for_mfsda_path}: {e}")

    if success:
        logger.info(f"Static statistical analysis completed for {side} hemisphere. Results saved to {output_dir}")
    else:
        logger.error(f"Static statistical analysis failed for {side} hemisphere. Check logs for details.")
    
    return success

def run_longitudinal_analysis(side: str = "left") -> bool:
    """
    Run longitudinal statistical analysis on delta models.
    """
    print_section_header(f"LONGITUDINAL STATISTICAL ANALYSIS: {side.upper()} HEMISPHERE")
    
    try:
        original_csv_path = config.get_paths_by_side(side, "longit_stats_csv")
        output_dir = config.get_paths_by_side(side, "stats_longit_output")
        shape_template = config.get_paths_by_side(side, "sphere_template") 
        sphere_template_for_mfsda = config.get_paths_by_side(side, "sphere_template")
    except ValueError as e:
        logger.error(f"Configuration error getting paths for longitudinal analysis (side: {side}): {e}")
        return False
    except AttributeError as e:
        logger.error(f"Configuration error: {e}. A required path variable might be missing in config.py.")
        return False

    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Ensured output directory for longitudinal stats exists: {output_dir}")
    except OSError as e:
        logger.error(f"Could not create output directory {output_dir}: {e}")
        return False
        
    if not os.path.exists(original_csv_path):
        logger.error(f"Longitudinal statistics CSV file not found: {original_csv_path}.")
        return False
    
    logger.info(f"Running longitudinal statistical analysis using base CSV: {original_csv_path}")
    
    try:
        df_original = pd.read_csv(original_csv_path)
        if df_original.empty:
            logger.error(f"Longitudinal statistics CSV file {original_csv_path} is empty.")
            return False
        
        path_column_name = "DeltaModelPath" # Expected first column for longitudinal MFSDA
        if path_column_name not in df_original.columns or df_original.columns[0] != path_column_name: 
            logger.error(f"Longitudinal statistics CSV {original_csv_path} must have '{path_column_name}' as its first column for MFSDA.")
            return False
        
        if "SubjectID" not in df_original.columns or df_original.columns[1] != "SubjectID":
            logger.warning(f"Longitudinal CSV {original_csv_path} does not have 'SubjectID' as the second column. Assuming numerical covariates start after '{path_column_name}'.")
            cols_for_mfsda_data = list(df_original.columns[1:])
        else:
            subject_id_col_index = df_original.columns.get_loc('SubjectID')
            cols_for_mfsda_data = list(df_original.columns[subject_id_col_index + 1:])
        
        df_for_mfsda = df_original[[path_column_name] + cols_for_mfsda_data]

    except Exception as e:
        logger.error(f"Error reading or preparing data from longitudinal statistics CSV {original_csv_path}: {str(e)}")
        return False

    mfsda_run_path = getattr(config, "MFSDA_RUN_PATH", None)
    mfsda_createshapes_path = getattr(config, "MFSDA_CREATE_SHAPES_PATH", None)
    if not mfsda_run_path or not mfsda_createshapes_path: 
        logger.error("MFSDA_RUN_PATH or MFSDA_CREATE_SHAPES_PATH not defined in config.py.")
        return False

    temp_csv_for_mfsda_fd = -1
    temp_csv_for_mfsda_path = ""
    success = False
    try:
        temp_csv_for_mfsda_fd, temp_csv_for_mfsda_path = tempfile.mkstemp(suffix='.csv', prefix='mfsda_longit_input_')
        logger.info(f"Creating temporary MFSDA-compatible longitudinal CSV: {temp_csv_for_mfsda_path}")
        df_for_mfsda.to_csv(temp_csv_for_mfsda_path, index=False)

        success = run_mfsda_subprocess(
            csv_path_for_mfsda=temp_csv_for_mfsda_path, 
            output_dir=output_dir,
            shape_template=shape_template, 
            sphere_template_for_mfsda=sphere_template_for_mfsda,
            mfsda_run_path=mfsda_run_path,
            mfsda_createshapes_path=mfsda_createshapes_path,
            slicersalt_path=config.SLICERSALT_PATH
        )
    except Exception as e_outer:
        logger.error(f"Outer error during longitudinal MFSDA setup or execution: {e_outer}")
        success = False
    finally:
        if temp_csv_for_mfsda_fd != -1:
            try:
                os.close(temp_csv_for_mfsda_fd)
            except OSError as e:
                logger.warning(f"Error closing temp file descriptor for {temp_csv_for_mfsda_path}: {e}")
        if temp_csv_for_mfsda_path and os.path.exists(temp_csv_for_mfsda_path):
            try:
                os.remove(temp_csv_for_mfsda_path)
                logger.info(f"Removed temporary MFSDA longitudinal CSV: {temp_csv_for_mfsda_path}")
            except OSError as e:
                logger.warning(f"Error removing temp file {temp_csv_for_mfsda_path}: {e}")


    if success:
        logger.info(f"Longitudinal statistical analysis completed for {side} hemisphere. Results saved to {output_dir}")
    else:
        logger.error(f"Longitudinal statistical analysis failed for {side} hemisphere. Check logs for details.")
    
    return success

def run_mfsda_subprocess(csv_path_for_mfsda: str, 
                         output_dir: str, 
                         shape_template: str, 
                         sphere_template_for_mfsda: str, 
                         mfsda_run_path: str, 
                         mfsda_createshapes_path: str, 
                         slicersalt_path: str) -> bool: 
    
    logger.info(f"Starting MFSDA subprocess with MFSDA-specific CSV: {csv_path_for_mfsda}")
    logger.info(f"MFSDA output directory: {output_dir}")
    logger.info(f"MFSDA shape template (for createShapes): {shape_template}")
    logger.info(f"MFSDA sphere template (for MFSDA_run coordData): {sphere_template_for_mfsda}")

    mfsda_timeout = getattr(config, 'MFSDA_TIMEOUT_SECONDS', 7200) # Default to 2 hours if not in config
    logger.info(f"MFSDA subprocess timeout set to: {mfsda_timeout} seconds.")

    for filepath_to_check in [csv_path_for_mfsda, shape_template, sphere_template_for_mfsda, slicersalt_path, mfsda_run_path, mfsda_createshapes_path]:
        if not os.path.exists(filepath_to_check):
            logger.error(f"MFSDA required file not found: {filepath_to_check}")
            return False
    
    try:
        temp_df_header = pd.read_csv(csv_path_for_mfsda, nrows=0).columns.tolist()
        if not temp_df_header or len(temp_df_header) < 1 :
             logger.error(f"Temporary MFSDA CSV {csv_path_for_mfsda} seems to have no header or is invalid.")
             return False
        
        mfsda_create_shapes_covariates_names = temp_df_header[1:]
        covariates_str_for_createshapes = ' '.join(mfsda_create_shapes_covariates_names)
        
        logger.info(f"MFSDA using subject path column from first column of: {csv_path_for_mfsda}")
        logger.info(f"MFSDA numerical covariates (for createShapes) derived from columns: {covariates_str_for_createshapes if covariates_str_for_createshapes else 'None'}")
        
        mfsda_run_command = [
            slicersalt_path, "--no-main-window", "--python-script", mfsda_run_path,
            "--shapeData", csv_path_for_mfsda, 
            "--coordData", sphere_template_for_mfsda, 
            "--outputDir", output_dir,
        ]
        
        logger.info("Running MFSDA_run.py...")
        logger.debug(f"MFSDA_run command: {' '.join(mfsda_run_command)}")
        result_run = run_command(
            mfsda_run_command,
            description="Running MFSDA statistical computations",
            check=False,
            timeout=mfsda_timeout # Apply timeout
        )
        if result_run.stdout: logger.info(f"MFSDA_run STDOUT:\n{result_run.stdout}")
        if result_run.stderr: logger.warning(f"MFSDA_run STDERR:\n{result_run.stderr}") 
        
        if result_run.returncode != 0:
            logger.error(f"MFSDA_run.py failed with return code {result_run.returncode}. Check for timeout messages above if applicable.")
            return False
            
        pvalues_file = os.path.join(output_dir, 'pvalues.json')
        efit_file = os.path.join(output_dir, 'efit.json')
        
        if not os.path.exists(pvalues_file) or not os.path.exists(efit_file):
            logger.error(f"MFSDA_run.py did not create expected output files (pvalues.json, efit.json) in {output_dir}")
            return False
            
        try:
            with open(pvalues_file, 'r') as f:
                pvalues_data = json.load(f)
                min_pval_val = "N/A" 
                if 'Gpvals' in pvalues_data and isinstance(pvalues_data['Gpvals'], list) and pvalues_data['Gpvals']:
                    min_pval_val = min(pvalues_data['Gpvals'])
                elif isinstance(pvalues_data, list) and pvalues_data: 
                    min_pval_val = min(pvalues_data)
                logger.info(f"MFSDA pvalues.json content summary: Min Gpval (if applicable) ~ {min_pval_val}")
        except Exception as e:
            logger.warning(f"Could not read or parse p-values file {pvalues_file}: {str(e)}")
        
        output_vtk_filename = f"MFSDA_visualization_{Path(csv_path_for_mfsda).stem.replace('mfsda_input_', '').replace('mfsda_static_input_', '').replace('mfsda_longit_input_', '')}.vtk"
        output_vtk_path = os.path.join(output_dir, output_vtk_filename)
        
        mfsda_create_shapes_command = [
            slicersalt_path, "--no-main-window", "--python-script", mfsda_createshapes_path,
            "--shape", shape_template, 
            "--pvalues", pvalues_file,
            "--efit", efit_file,
        ]
        if covariates_str_for_createshapes:
             mfsda_create_shapes_command.extend(["--covariates", covariates_str_for_createshapes])
        mfsda_create_shapes_command.extend(["--output", output_vtk_path])

        logger.info("Running MFSDA_createShapes.py...")
        logger.debug(f"MFSDA_createShapes command: {' '.join(mfsda_create_shapes_command)}")
        result_create = run_command(
            mfsda_create_shapes_command,
            description="Creating MFSDA shape visualization",
            check=False,
            timeout=mfsda_timeout # Apply timeout
        )
        if result_create.stdout: logger.info(f"MFSDA_createShapes STDOUT:\n{result_create.stdout}")
        if result_create.stderr: logger.warning(f"MFSDA_createShapes STDERR:\n{result_create.stderr}")

        if result_create.returncode != 0:
            logger.error(f"MFSDA_createShapes.py failed with return code {result_create.returncode}. Check for timeout messages above if applicable.")
            return False
        
        if not os.path.exists(output_vtk_path):
            logger.error(f"MFSDA_createShapes.py did not create the expected output VTK file: {output_vtk_path}")
            return False

        logger.info(f"MFSDA analysis completed successfully. Visualization: {output_vtk_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error during MFSDA subprocess execution: {str(e)}")
        logger.exception("Detailed traceback for MFSDA error:")
        return False

# --- Functions below this line are for M2MD and Delta Model generation ---
# --- They are assumed to be mostly correct from previous versions ---
# --- but reviewed for basic consistency with the MFSDA changes. ---

def prepare_longitudinal_data(side: str = "left") -> bool:
    print_section_header(f"LONGITUDINAL DATA PREPARATION (M2MD): {side.upper()} HEMISPHERE")
    
    try:
        m2md_input_csv_path = config.get_paths_by_side(side, "longit_m2md_csv")
        m2md_output_dir = config.get_paths_by_side(side, "m2md_output") 
        delta_models_final_dir = config.get_paths_by_side(side, "delta_models_output") 
    except ValueError as e:
        logger.error(f"Configuration error getting paths for longitudinal data prep (side: {side}): {e}")
        return False
    except AttributeError as e:
        logger.error(f"Configuration error: {e}. A required path variable might be missing in config.py.")
        return False

    try:
        os.makedirs(m2md_output_dir, exist_ok=True)
        logger.info(f"Ensured M2MD output directory exists: {m2md_output_dir}")
        os.makedirs(delta_models_final_dir, exist_ok=True)
        logger.info(f"Ensured delta models final directory exists: {delta_models_final_dir}")
    except OSError as e:
        logger.error(f"Could not create output directories for longitudinal data: {e}")
        return False
        
    if not os.path.exists(m2md_input_csv_path):
        logger.error(f"M2MD input CSV file not found: {m2md_input_csv_path}. This file should define Timepoint1, Timepoint2, and Output names.")
        return False
    
    try:
        df = pd.read_csv(m2md_input_csv_path)
        required_cols = ['Timepoint 1', 'Timepoint 2', 'Output'] 
        if not all(col in df.columns for col in required_cols):
            logger.error(f"M2MD input CSV {m2md_input_csv_path} is missing one or more required columns: {required_cols}")
            return False
        if df.empty:
            logger.warning(f"M2MD input CSV {m2md_input_csv_path} is empty. No pairs to process.")
            return True 
    except Exception as e:
        logger.error(f"Error reading or validating M2MD input CSV {m2md_input_csv_path}: {e}")
        return False
        
    m2md_success = run_m2md_slicer_script(
        input_csv_for_pairs=m2md_input_csv_path, 
        m2md_tool_output_dir=m2md_output_dir, 
        slicersalt_path=config.SLICERSALT_PATH
    )
    
    if not m2md_success:
        logger.error(f"M2MD Slicer script execution failed for {side} hemisphere.")
        return False
    
    m2md_results_summary_csv = os.path.join(m2md_output_dir, "M2MD_results.csv") 
    if not os.path.exists(m2md_results_summary_csv):
        logger.error(f"M2MD Slicer script did not create the expected results summary CSV: {m2md_results_summary_csv}")
        return False
        
    delta_generation_success = generate_delta_models_from_m2md_outputs(
        m2md_results_summary_csv=m2md_results_summary_csv, 
        final_delta_models_dir=delta_models_final_dir
    )
    
    if not delta_generation_success:
        logger.error(f"Generation of final delta models failed for {side} hemisphere.")
        return False
            
    logger.info(f"Longitudinal data preparation (M2MD and delta model generation) completed for {side} hemisphere.")
    logger.info(f"M2MD outputs are in: {m2md_output_dir}")
    logger.info(f"Final delta models for statistical analysis are in: {delta_models_final_dir}")
    return True

def run_m2md_slicer_script(input_csv_for_pairs: str, m2md_tool_output_dir: str, slicersalt_path: str) -> bool:
    logger.info(f"Running Model-to-Model Distance Slicer script for CSV: {input_csv_for_pairs}")
    abs_input_csv_for_pairs = os.path.abspath(input_csv_for_pairs)
    abs_m2md_tool_output_dir = os.path.abspath(m2md_tool_output_dir)
    
    script_fd, script_path = tempfile.mkstemp(suffix='.py', prefix='m2md_slicer_script_')
    
    try:
        # Using f-strings with raw string literals for paths, ensure correctness for the embedded script
        m2md_script_content = f"""
# M2MD Script for SlicerSALT
import os
import numpy as np
import pandas as pd
import slicer 
import logging 

# Basic script logger setup within the Slicer environment
# Note: Slicer's Python environment might handle logging differently.
# Print statements are often more reliable for seeing output from Slicer --python-script.
print(f"M2MD Slicer Script started.")
print(f"Input CSV for pairs: {{abs_input_csv_for_pairs!r}}") # !r for repr to handle escapes
print(f"Output directory for M2MD results: {{abs_m2md_tool_output_dir!r}}")

try:
    os.makedirs(r"{abs_m2md_tool_output_dir}", exist_ok=True)
    print(f"Ensured M2MD output directory exists: {{abs_m2md_tool_output_dir!r}}")
except Exception as e:
    print(f"Failed to create M2MD output directory {{abs_m2md_tool_output_dir!r}}: {{e}}")
    slicer.app.exit(1)

try:
    data = pd.read_csv(r"{abs_input_csv_for_pairs}")
    if data.empty:
        print(f"Input CSV {{abs_input_csv_for_pairs!r}} is empty. No pairs to process.")
        slicer.app.exit(0) 
        
    required_columns = ['Timepoint 1', 'Timepoint 2', 'Output']
    if not all(col in data.columns for col in required_columns):
        print(f"Error: CSV {{abs_input_csv_for_pairs!r}} is missing one or more required columns: {{required_columns}}")
        slicer.app.exit(1)
            
    tp1_subjects_paths = data['Timepoint 1'].tolist()
    tp2_subjects_paths = data['Timepoint 2'].tolist()
    output_base_names = data['Output'].tolist() 
    
    m2md_results_data = [] 

    print(f"Found {{len(tp1_subjects_paths)}} pairs to process from {{abs_input_csv_for_pairs!r}}.")

    for i in range(len(tp1_subjects_paths)):
        tp1_subj_path = tp1_subjects_paths[i]
        tp2_subj_path = tp2_subjects_paths[i]
        output_subj_base_name = output_base_names[i]
        
        print(f"Processing pair {{i+1}}/{{len(tp1_subjects_paths)}}: {{output_subj_base_name}}")
        print(f"  Timepoint 1: {{tp1_subj_path}}")
        print(f"  Timepoint 2: {{tp2_subj_path}}")

        if not os.path.exists(tp1_subj_path):
            print(f"  Skipping pair {{output_subj_base_name}}: Timepoint 1 file not found: {{tp1_subj_path}}")
            continue
        if not os.path.exists(tp2_subj_path):
            print(f"  Skipping pair {{output_subj_base_name}}: Timepoint 2 file not found: {{tp2_subj_path}}")
            continue

        try:
            slicer.mrmlScene.Clear(0) 
            
            model1_node = slicer.util.loadModel(tp1_subj_path)
            if not model1_node:
                print(f"  Failed to load Timepoint 1 model: {{tp1_subj_path}}")
                continue
            model2_node = slicer.util.loadModel(tp2_subj_path)
            if not model2_node:
                print(f"  Failed to load Timepoint 2 model: {{tp2_subj_path}}")
                continue
            
            output_m2md_vtk_path = os.path.join(r"{abs_m2md_tool_output_dir}", output_subj_base_name + "_m2md.vtk")
            output_distance_scalars_csv_path = os.path.join(r"{abs_m2md_tool_output_dir}", output_subj_base_name + "_DistanceScalars.csv")
            
            slicer_output_node_name = output_subj_base_name + "_M2MDResult"
            output_model_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", slicer_output_node_name)
            if not output_model_node:
                print(f"  Failed to create output node for M2MD result: {{slicer_output_node_name}}")
                continue

            # Parameter names for ModelToModelDistance can vary slightly by Slicer version.
            # Common names are 'sourceModel', 'targetModel', 'outputModel'.
            # Check your Slicer version's CLI documentation if these don't work.
            params = {{
                'sourceModel': model1_node.GetID(), 
                'targetModel': model2_node.GetID(), 
                'outputModel': output_model_node.GetID(), 
                # 'distanceType': 'point_to_cell', # Example optional parameter
                # 'signedDistance': False          # Example optional parameter
            }}
            
            print(f"  Running ModelToModelDistance CLI module...")
            cli_module = slicer.modules.modeltomodeldistance
            cli_node = slicer.cli.runSync(cli_module, None, params) 

            if cli_node.GetStatusString() == "Completed":
                print(f"  ModelToModelDistance completed for {{output_subj_base_name}}.")
                if not slicer.util.saveNode(output_model_node, output_m2md_vtk_path):
                    print(f"  Failed to save output M2MD model: {{output_m2md_vtk_path}}")
                else:
                    print(f"  Output M2MD model saved: {{output_m2md_vtk_path}}")
                
                distance_array_vtk = output_model_node.GetPolyData().GetPointData().GetScalars("Distance") # Default scalar name
                if distance_array_vtk:
                    num_points = distance_array_vtk.GetNumberOfTuples()
                    distance_values_np = np.array([distance_array_vtk.GetTuple1(i) for i in range(num_points)])
                    np.savetxt(output_distance_scalars_csv_path, distance_values_np, delimiter=",", fmt='%f', comments='')
                    print(f"  Distance array saved: {{output_distance_scalars_csv_path}}")
                    
                    m2md_results_data.append({{
                        'Input_Timepoint1_Path': tp1_subj_path,
                        'Input_Timepoint2_Path': tp2_subj_path,
                        'Input_Output_BaseName': output_subj_base_name,
                        'Output_M2MD_VTK_Path': output_m2md_vtk_path,
                        'Output_DistanceScalars_CSV_Path': output_distance_scalars_csv_path 
                    }})
                else:
                    print(f"  Could not extract 'Distance' array from M2MD output for {{output_subj_base_name}}.")
                    m2md_results_data.append({{
                        'Input_Timepoint1_Path': tp1_subj_path,
                        'Input_Timepoint2_Path': tp2_subj_path,
                        'Input_Output_BaseName': output_subj_base_name,
                        'Output_M2MD_VTK_Path': output_m2md_vtk_path, 
                        'Output_DistanceScalars_CSV_Path': None 
                    }})
            else:
                print(f"  ModelToModelDistance failed for {{output_subj_base_name}}. Status: {{cli_node.GetStatusString()}}")
        except Exception as pair_e:
            print(f"  Error processing pair {{output_subj_base_name}}: {{str(pair_e)}}")
            import traceback
            print(traceback.format_exc())
            continue 
    
    if m2md_results_data:
        results_df = pd.DataFrame(m2md_results_data)
        summary_csv_path = os.path.join(r"{abs_m2md_tool_output_dir}", "M2MD_results.csv")
        results_df.to_csv(summary_csv_path, index=False)
        print(f"M2MD processing summary saved to: {{summary_csv_path}}")
    else:
        print("No M2MD pairs were successfully processed to create a summary CSV.")

    print(f"M2MD Slicer Script finished.")
    slicer.app.exit(0) 

except Exception as main_e:
    print(f"Critical error in M2MD Slicer Script: {{str(main_e)}}")
    import traceback
    print(traceback.format_exc())
    slicer.app.exit(1) 
"""
        with os.fdopen(script_fd, 'w') as f:
            f.write(m2md_script_content)
            
        slicer_command = [
            slicersalt_path,
            "--no-main-window", 
            "--python-script", script_path
        ]
        
        logger.info(f"Executing M2MD Slicer script via command: {' '.join(slicer_command)}")
        m2md_timeout = getattr(config, 'SPHARM_TIMEOUT_SECONDS', 7200) # Reuse SPHARM timeout or define a new one
        result = run_command(
            slicer_command,
            description=f"Running Model-to-Model Distance via SlicerSALT (CSV: {os.path.basename(input_csv_for_pairs)})",
            check=False, 
            timeout=m2md_timeout 
        )
        
        if result.stdout: logger.info(f"M2MD Slicer script STDOUT (captured by Python):\n{result.stdout}")
        if result.stderr: logger.warning(f"M2MD Slicer script STDERR (captured by Python):\n{result.stderr}")

        if result.returncode != 0:
            logger.error(f"M2MD Slicer script failed with SlicerSALT return code {result.returncode}")
            return False
            
        m2md_results_summary_csv = os.path.join(abs_m2md_tool_output_dir, "M2MD_results.csv")
        if not os.path.exists(m2md_results_summary_csv):
            logger.error(f"M2MD Slicer script completed (return code 0) but did not create the expected summary results file: {m2md_results_summary_csv}.")
            return False
            
        logger.info(f"M2MD Slicer script completed. Summary results expected at: {m2md_results_summary_csv}")
        return True
            
    except Exception as e:
        logger.error(f"Error setting up or running M2MD Slicer script: {str(e)}")
        logger.exception("Detailed traceback for M2MD Slicer script setup/run error:")
        return False
    finally:
        if os.path.exists(script_path): # script_fd is closed by with os.fdopen
            try:
                os.remove(script_path)
                logger.debug(f"Removed temporary M2MD Slicer script: {script_path}")
            except OSError as e:
                logger.warning(f"Could not remove temporary M2MD Slicer script: {script_path}: {e}")

def generate_delta_models_from_m2md_outputs(m2md_results_summary_csv: str, final_delta_models_dir: str) -> bool:
    logger.info(f"Generating final delta models from M2MD summary: {m2md_results_summary_csv}")
    logger.info(f"Output directory for final delta models: {final_delta_models_dir}")

    if not os.path.exists(m2md_results_summary_csv):
        logger.error(f"M2MD results summary CSV not found: {m2md_results_summary_csv}")
        return False
    
    try:
        os.makedirs(final_delta_models_dir, exist_ok=True)
    except OSError as e:
        logger.error(f"Could not create final delta models directory {final_delta_models_dir}: {e}")
        return False

    try:
        m2md_df = pd.read_csv(m2md_results_summary_csv)
        if m2md_df.empty:
            logger.warning(f"M2MD results summary CSV {m2md_results_summary_csv} is empty. No delta models to generate.")
            return True 

        required_cols = ['Input_Timepoint1_Path', 'Output_DistanceScalars_CSV_Path', 'Input_Output_BaseName']
        if not all(col in m2md_df.columns for col in required_cols):
            logger.error(f"M2MD summary CSV {m2md_results_summary_csv} is missing one or more required columns: {required_cols}")
            return False

        success_count = 0
        failure_count = 0

        try:
            import vtk
        except ImportError:
            logger.error("VTK Python package is not installed. Cannot generate delta models. Please install with 'pip install vtk'.")
            return False

        for index, row in m2md_df.iterrows():
            base_vtk_path = row['Input_Timepoint1_Path']
            distance_scalars_csv_path = row.get('Output_DistanceScalars_CSV_Path') 
            output_base_name = row['Input_Output_BaseName']
            
            final_delta_vtk_filename = f"{output_base_name}_delta.vtk" 
            final_delta_vtk_path = os.path.join(final_delta_models_dir, final_delta_vtk_filename)

            if pd.isna(distance_scalars_csv_path) or not distance_scalars_csv_path:
                logger.warning(f"Skipping delta model for '{output_base_name}': DistanceScalars CSV path is missing or invalid in summary.")
                failure_count +=1
                continue
            if not os.path.exists(base_vtk_path):
                logger.error(f"Skipping delta model for '{output_base_name}': Base VTK (Timepoint 1) not found: {base_vtk_path}")
                failure_count +=1
                continue
            if not os.path.exists(distance_scalars_csv_path):
                logger.error(f"Skipping delta model for '{output_base_name}': DistanceScalars CSV not found: {distance_scalars_csv_path}")
                failure_count +=1
                continue

            logger.info(f"Generating delta model: {final_delta_vtk_path}")
            logger.debug(f"  Using base VTK: {base_vtk_path}")
            logger.debug(f"  Using distance scalars: {distance_scalars_csv_path}")

            try:
                distance_values = pd.read_csv(distance_scalars_csv_path, header=None).values.astype(float).flatten()
                
                reader = vtk.vtkPolyDataReader()
                reader.SetFileName(base_vtk_path)
                reader.Update()
                polydata = reader.GetOutput()

                if not polydata or polydata.GetNumberOfPoints() == 0:
                    logger.error(f"  Failed to read base VTK or it's empty: {base_vtk_path} for {output_base_name}.")
                    failure_count +=1
                    continue

                if polydata.GetNumberOfPoints() != len(distance_values):
                    logger.error(f"  Mismatch in point count ({polydata.GetNumberOfPoints()}) and distance values ({len(distance_values)}) for {output_base_name}. Cannot create delta model.")
                    failure_count +=1
                    continue

                vtk_distance_array = vtk.vtkDoubleArray() 
                vtk_distance_array.SetName("DeltaValues") 
                vtk_distance_array.SetNumberOfComponents(1)
                for val in distance_values: # More robust way to add tuples
                    vtk_distance_array.InsertNextTuple1(val)
                
                polydata.GetPointData().AddArray(vtk_distance_array) 
                polydata.GetPointData().SetActiveScalars("DeltaValues")
                
                writer = vtk.vtkPolyDataWriter()
                writer.SetFileName(final_delta_vtk_path)
                writer.SetInputData(polydata)
                writer.SetFileVersion(42) 
                writer.Write()
                logger.info(f"  Successfully generated delta model: {final_delta_vtk_path}")
                success_count += 1

            except Exception as e_vtk:
                logger.error(f"  Error generating VTK delta model for {output_base_name}: {e_vtk}")
                logger.exception("Detailed traceback for VTK delta model generation error:")
                failure_count += 1
        
        logger.info(f"Delta model generation finished. Success: {success_count}, Failures: {failure_count}")
        return failure_count == 0 

    except Exception as e_main:
        logger.error(f"Error processing M2MD results summary CSV for delta model generation: {e_main}")
        logger.exception("Detailed traceback for delta model generation error:")
        return False
