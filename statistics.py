"""
Statistics Module
---------------
Handles statistical analysis for static and longitudinal data using SlicerSALT modules.
Monitors SlicerSALT output for completion phrases to manage process termination.
"""

import os
import pandas as pd
import subprocess
import tempfile
import logging
import json
from pathlib import Path
import time # For Popen monitoring loop
from typing import Dict, Optional, List, Union

# Attempt to import project-specific configuration.
try:
    import config
except ImportError:
    logging.basicConfig(level=logging.ERROR)
    logger = logging.getLogger(__name__)
    logger.error("CRITICAL: config.py not found. This script relies on it for paths and settings.")
    class DummyConfig: # Fallback for parsing if config is missing
        SLICERSALT_PATH = "/path/to/SlicerSALT"
        MFSDA_RUN_PATH = "/path/to/MFSDA_run.py"
        MFSDA_CREATE_SHAPES_PATH = "/path/to/MFSDA_createShapes.py"
        MFSDA_TIMEOUT_SECONDS = 7200
        SPHARM_TIMEOUT_SECONDS = 7200
        MFSDA_RUN_COMPLETION_PHRASE = "The total elapsed time is"
        MFSDA_CREATESHAPES_COMPLETION_PHRASE = None # Or a specific phrase if known
        def get_paths_by_side(self, side, component): return f"/dummy/path/{side}/{component}"
    config = DummyConfig()

# Assuming utils.py is in the same directory or PYTHONPATH
try:
    from utils import run_command, print_section_header # run_command might still be used for other things
except ImportError:
    logger = logging.getLogger(__name__)
    logger.error("CRITICAL: utils.py not found.")
    def print_section_header(title): logger.info(title)
    # Define a dummy run_command if not available, though Popen will be primary here
    def run_command(command, description="", check=True, timeout=None, cwd=None):
        logger.error(f"Dummy run_command called for: {command}. utils.py is missing.")
        return subprocess.CompletedProcess(command, 1, "Dummy STDOUT", "Dummy STDERR: utils.py missing")


if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
logger = logging.getLogger(__name__)


def _run_slicersalt_module_monitor_output(
    command_list: List[str],
    completion_phrase: Optional[str],
    timeout_seconds: int,
    description: str = ""
) -> bool:
    """
    Runs a SlicerSALT module using subprocess.Popen, monitors its stdout for a
    completion phrase, and terminates the process.

    Args:
        command_list: The command and arguments to execute.
        completion_phrase: The string to look for in stdout to indicate success.
                           If None, relies on process exit or timeout.
        timeout_seconds: Maximum time to wait for the process.
        description: A description of the task for logging.

    Returns:
        True if the completion phrase is found (or process exits with 0 if no phrase),
        False on timeout or error.
    """
    if description:
        logger.info(description)
    logger.info(f"Executing (with Popen monitoring): {' '.join(command_list)}")
    if completion_phrase:
        logger.info(f"Watching for completion phrase: '{completion_phrase}'")
    logger.info(f"Timeout set to: {timeout_seconds} seconds.")

    process = None
    try:
        process = subprocess.Popen(
            command_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # Redirect stderr to stdout
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )

        start_time = time.time()
        log_buffer = [] # To store recent lines for context on error

        while True:
            # Check for timeout
            if (time.time() - start_time) > timeout_seconds:
                logger.error(f"Timeout: Process exceeded {timeout_seconds} seconds for: {description}")
                process.terminate()
                try:
                    process.wait(timeout=5) # Wait a bit for terminate
                except subprocess.TimeoutExpired:
                    logger.warning("Process did not terminate gracefully after timeout, killing.")
                    process.kill()
                return False

            # Read output line
            if process.stdout:
                output_line = process.stdout.readline()
                if output_line:
                    line_stripped = output_line.strip()
                    if line_stripped:
                        logger.info(f"[SlicerSALT_LOG] {line_stripped}")
                        log_buffer.append(line_stripped)
                        if len(log_buffer) > 50: # Keep last 50 lines
                            log_buffer.pop(0)
                    
                    if completion_phrase and completion_phrase in output_line:
                        logger.info(f"Completion phrase '{completion_phrase}' detected for: {description}")
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            logger.warning(f"Process for '{description}' did not terminate gracefully after phrase, killing.")
                            process.kill()
                        return True
                elif process.poll() is not None: # Process has exited
                    break # Exit while loop
            else: # Should not happen if stdout is piped
                time.sleep(0.1)


            # Check if process has exited (after attempting to read line)
            if process.poll() is not None:
                break
            
            time.sleep(0.05) # Small sleep to prevent busy-waiting

        # Process has exited, check return code
        return_code = process.returncode
        logger.info(f"SlicerSALT process for '{description}' exited with code: {return_code}")
        
        if completion_phrase: 
            logger.warning(f"Process for '{description}' exited before completion phrase was found. Return code: {return_code}")
            return False 
        else: 
            return return_code == 0

    except FileNotFoundError:
        logger.error(f"SlicerSALT executable or script not found. Command: {' '.join(command_list)}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred running SlicerSALT module '{description}': {e}")
        logger.exception("Detailed traceback:")
        if process and process.poll() is None:
            logger.warning(f"Terminating SlicerSALT process for '{description}' due to exception.")
            process.kill()
        return False
    finally:
        if process and process.poll() is None:
            logger.warning(f"Ensuring SlicerSALT process for '{description}' is terminated (reached finally block while running).")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


def _prepare_mfsda_dataframe(original_csv_path: str, expected_path_column: str) -> Optional[pd.DataFrame]:
    """
    Reads the CSV intended for MFSDA.
    It expects the first column to be the path to shape data (as per expected_path_column),
    and all subsequent columns to be numerical covariates.
    It will attempt to convert these subsequent columns to numeric and drop any that fail.
    """
    try:
        df_original = pd.read_csv(original_csv_path)
        if df_original.empty:
            logger.error(f"Input CSV file {original_csv_path} is empty.")
            return None
        
        if expected_path_column not in df_original.columns or df_original.columns[0] != expected_path_column:
            logger.error(f"Input CSV {original_csv_path} must contain '{expected_path_column}' as its first column.")
            return None

        # All columns after the first (path) column are considered potential numerical covariates.
        potential_covariate_cols = list(df_original.columns[1:])
        logger.info(f"Using '{expected_path_column}' as path column from {original_csv_path}. "
                    f"Attempting to use subsequent columns as numerical covariates: {potential_covariate_cols}")

        # Create a copy to modify, starting with the path column
        df_for_mfsda = df_original[[expected_path_column]].copy()
        
        actual_numerical_covariate_cols = []
        for col in potential_covariate_cols:
            try:
                # Attempt to convert the column to numeric
                numeric_col = pd.to_numeric(df_original[col])
                # If successful, add it to our DataFrame for MFSDA
                df_for_mfsda[col] = numeric_col
                actual_numerical_covariate_cols.append(col)
            except ValueError:
                logger.warning(f"Column '{col}' in {original_csv_path} could not be converted to numeric and will be excluded from MFSDA covariates.")
        
        if not actual_numerical_covariate_cols: # Check if any numerical covariates were successfully processed
            logger.error(f"After processing, no valid numerical covariates found in {original_csv_path} for MFSDA. "
                         f"Ensure columns after '{expected_path_column}' are numeric or can be converted to numeric.")
            return None
            
        logger.info(f"Prepared DataFrame for MFSDA with path column '{expected_path_column}' and numerical covariates: {actual_numerical_covariate_cols}")
        return df_for_mfsda

    except Exception as e:
        logger.error(f"Error reading or preparing data from CSV {original_csv_path} for MFSDA: {e}")
        logger.exception("Detailed traceback:")
        return None


def run_mfsda_subprocess(csv_path_for_mfsda: str,
                         output_dir: str,
                         shape_template: str,
                         sphere_template_for_mfsda: str,
                         mfsda_run_path: str,
                         mfsda_createshapes_path: str,
                         slicersalt_path: str) -> bool:
    """
    Executes the MFSDA SlicerSALT modules (run and createShapes) using Popen for monitoring.
    """
    logger.info(f"Starting MFSDA processing with MFSDA-specific CSV: {csv_path_for_mfsda}")
    logger.info(f"MFSDA output directory: {output_dir}")

    mfsda_timeout = getattr(config, 'MFSDA_TIMEOUT_SECONDS', 7200)
    mfsda_run_phrase = getattr(config, 'MFSDA_RUN_COMPLETION_PHRASE', "The total elapsed time is")
    mfsda_createshapes_phrase = getattr(config, 'MFSDA_CREATESHAPES_COMPLETION_PHRASE', None) 


    for required_file in [csv_path_for_mfsda, shape_template, sphere_template_for_mfsda,
                          slicersalt_path, mfsda_run_path, mfsda_createshapes_path]:
        if not os.path.exists(required_file):
            logger.error(f"MFSDA required file/script not found: {required_file}")
            return False
    try:
        temp_df_header = pd.read_csv(csv_path_for_mfsda, nrows=0).columns.tolist()
        if not temp_df_header or len(temp_df_header) < 1: # Should have at least path column
            logger.error(f"Prepared MFSDA CSV {csv_path_for_mfsda} has no header or is invalid.")
            return False
        
        # Covariate names for MFSDA_createShapes are all columns *after* the first (path) column
        mfsda_create_shapes_covariates_names = temp_df_header[1:]
        covariates_str_for_createshapes = ' '.join(mfsda_create_shapes_covariates_names)
        
        logger.info(f"MFSDA using path column from: {csv_path_for_mfsda}")
        logger.info(f"Covariate names for MFSDA_createShapes: '{covariates_str_for_createshapes if covariates_str_for_createshapes else 'None (or MFSDA_createShapes will use defaults)'}'")

        # --- Step 1: Run MFSDA_run.py ---
        mfsda_run_command = [
            slicersalt_path, "--no-main-window", "--python-script", mfsda_run_path,
            "--shapeData", csv_path_for_mfsda,
            "--coordData", sphere_template_for_mfsda,
            "--outputDir", output_dir,
        ]
        
        run_success = _run_slicersalt_module_monitor_output(
            command_list=mfsda_run_command,
            completion_phrase=mfsda_run_phrase,
            timeout_seconds=mfsda_timeout,
            description="MFSDA statistical computations (MFSDA_run.py)"
        )

        if not run_success:
            logger.error("MFSDA_run.py SlicerSALT module execution failed or timed out.")
            return False
        logger.info("MFSDA_run.py SlicerSALT module processing deemed complete.")

        pvalues_file = os.path.join(output_dir, 'pvalues.json')
        efit_file = os.path.join(output_dir, 'efit.json')
        if not os.path.exists(pvalues_file) or not os.path.exists(efit_file):
            logger.error(f"MFSDA_run.py did not create expected output files (pvalues.json, efit.json) in {output_dir} despite apparent completion.")
            return False

        # --- Step 2: Run MFSDA_createShapes.py ---
        input_csv_stem = Path(csv_path_for_mfsda).stem
        # Clean up common prefixes from temp file names for a cleaner output VTK name
        cleaned_stem = input_csv_stem
        for prefix_to_remove in ['mfsda_input_', 'mfsda_static_input_', 'mfsda_longit_input_', 'mfsda_static_left_', 'mfsda_static_right_', 'mfsda_longitudinal_left_', 'mfsda_longitudinal_right_']:
            if cleaned_stem.startswith(prefix_to_remove):
                cleaned_stem = cleaned_stem[len(prefix_to_remove):]
        output_vtk_filename = f"MFSDA_visualization_{cleaned_stem}.vtk"
        output_vtk_path = os.path.join(output_dir, output_vtk_filename)

        mfsda_create_shapes_command = [
            slicersalt_path, "--no-main-window", "--python-script", mfsda_createshapes_path,
            "--shape", shape_template,
            "--pvalues", pvalues_file,
            "--efit", efit_file,
            "--output", output_vtk_path
        ]
        # Only add the --covariates argument if there are actual covariate names to pass
        if covariates_str_for_createshapes:
            mfsda_create_shapes_command.extend(["--covariates", covariates_str_for_createshapes])
        else:
            logger.info("No covariate names to pass to MFSDA_createShapes.py; it may use defaults or not require them if only one group/no covariates.")


        create_shapes_success = _run_slicersalt_module_monitor_output(
            command_list=mfsda_create_shapes_command,
            completion_phrase=mfsda_createshapes_phrase, 
            timeout_seconds=mfsda_timeout,
            description="Creating MFSDA shape visualization (MFSDA_createShapes.py)"
        )

        if not create_shapes_success:
            logger.error("MFSDA_createShapes.py SlicerSALT module execution failed or timed out.")
            return False
        
        if not os.path.exists(output_vtk_path):
            logger.error(f"MFSDA_createShapes.py did not create the expected output VTK file: {output_vtk_path} despite apparent completion.")
            return False
        
        logger.info(f"MFSDA_createShapes.py SlicerSALT module processing deemed complete. Visualization: {output_vtk_path}")
        return True

    except Exception as e:
        logger.error(f"An unexpected error occurred during MFSDA subprocess orchestration: {e}")
        logger.exception("Detailed traceback:")
        return False


def run_analysis_with_mfsda(analysis_type: str, side: str, expected_path_column: str) -> bool:
    """Helper function to run MFSDA for static or longitudinal analysis."""
    print_section_header(f"{analysis_type.upper()} STATISTICAL ANALYSIS (MFSDA): {side.upper()} HEMISPHERE")

    try:
        if analysis_type == "static":
            original_csv_path = config.get_paths_by_side(side, "static_csv")
            output_dir = config.get_paths_by_side(side, "stats_static_output")
        elif analysis_type == "longitudinal":
            original_csv_path = config.get_paths_by_side(side, "longit_stats_csv")
            output_dir = config.get_paths_by_side(side, "stats_longit_output")
        else:
            logger.error(f"Unknown analysis type for MFSDA: {analysis_type}")
            return False
        shape_template = config.get_paths_by_side(side, "sphere_template")
        sphere_template_for_mfsda = config.get_paths_by_side(side, "sphere_template")
    except (ValueError, AttributeError) as e:
        logger.error(f"Configuration error getting paths for {analysis_type} MFSDA (side: {side}): {e}")
        return False

    if not os.path.exists(original_csv_path):
        logger.error(f"{analysis_type.capitalize()} analysis base CSV not found: {original_csv_path}")
        return False
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        logger.error(f"Could not create output directory {output_dir} for {analysis_type} MFSDA: {e}")
        return False

    logger.info(f"Preparing data for {analysis_type} MFSDA from base CSV: {original_csv_path}")
    df_for_mfsda = _prepare_mfsda_dataframe(original_csv_path, expected_path_column)
    if df_for_mfsda is None: # _prepare_mfsda_dataframe now returns None on critical failure
        logger.error(f"Failed to prepare MFSDA-compatible DataFrame from {original_csv_path}. Halting MFSDA for this run.")
        return False

    slicersalt_path = getattr(config, "SLICERSALT_PATH", None)
    mfsda_run_path = getattr(config, "MFSDA_RUN_PATH", None)
    mfsda_createshapes_path = getattr(config, "MFSDA_CREATE_SHAPES_PATH", None)

    for p, name in [(slicersalt_path, "SLICERSALT_PATH"), (mfsda_run_path, "MFSDA_RUN_PATH"), 
                    (mfsda_createshapes_path, "MFSDA_CREATE_SHAPES_PATH")]:
        if not p or not os.path.exists(p):
            logger.error(f"{name} ('{p}') not defined in config.py or does not exist. Cannot run MFSDA.")
            return False

    temp_csv_fd = -1
    temp_csv_path = ""
    analysis_success = False
    try:
        # Use a more descriptive prefix for the temp file
        temp_file_prefix = f'mfsda_{analysis_type}_{side}_processed_input_'
        temp_csv_fd, temp_csv_path = tempfile.mkstemp(suffix='.csv', prefix=temp_file_prefix)
        logger.info(f"Creating temporary MFSDA-compatible CSV: {temp_csv_path}")
        df_for_mfsda.to_csv(temp_csv_path, index=False)
        
        analysis_success = run_mfsda_subprocess(
            csv_path_for_mfsda=temp_csv_path,
            output_dir=output_dir,
            shape_template=shape_template,
            sphere_template_for_mfsda=sphere_template_for_mfsda,
            mfsda_run_path=mfsda_run_path,
            mfsda_createshapes_path=mfsda_createshapes_path,
            slicersalt_path=slicersalt_path
        )
    except Exception as e_outer:
        logger.error(f"Outer error during {analysis_type} MFSDA setup or execution for {side} side: {e_outer}")
        logger.exception("Detailed traceback:")
        analysis_success = False
    finally:
        if temp_csv_fd != -1:
            try: os.close(temp_csv_fd)
            except OSError as e: logger.warning(f"Error closing temp file descriptor for {temp_csv_path}: {e}")
        if temp_csv_path and os.path.exists(temp_csv_path):
            try: 
                os.remove(temp_csv_path)
                logger.info(f"Removed temporary MFSDA CSV: {temp_csv_path}")
            except OSError as e: logger.warning(f"Error removing temp file {temp_csv_path}: {e}")

    if analysis_success:
        logger.info(f"{analysis_type.capitalize()} statistical analysis (MFSDA) completed for {side} hemisphere.")
    else:
        logger.error(f"{analysis_type.capitalize()} statistical analysis (MFSDA) failed for {side} hemisphere.")
    return analysis_success


def run_static_analysis(side: str = "left") -> bool:
    return run_analysis_with_mfsda("static", side, expected_path_column="SubjectPath")

def run_longitudinal_analysis(side: str = "left") -> bool:
    return run_analysis_with_mfsda("longitudinal", side, expected_path_column="DeltaModelPath")


# --- M2MD and Delta Model Generation Functions ---
# (Assumed to be generally functional from previous versions)

def prepare_longitudinal_data(side: str = "left") -> bool:
    print_section_header(f"LONGITUDINAL DATA PREPARATION (M2MD): {side.upper()} HEMISPHERE")
    try:
        m2md_input_csv_path = config.get_paths_by_side(side, "longit_m2md_csv")
        m2md_tool_output_dir = config.get_paths_by_side(side, "m2md_output") 
        delta_models_final_dir = config.get_paths_by_side(side, "delta_models_output") 
        slicersalt_path = getattr(config, "SLICERSALT_PATH", None)
    except (ValueError, AttributeError) as e:
        logger.error(f"Configuration error getting paths for M2MD (side: {side}): {e}")
        return False
    if not slicersalt_path or not os.path.exists(slicersalt_path):
        logger.error(f"SlicerSALT path ('{slicersalt_path}') not defined or does not exist. Cannot run M2MD.")
        return False
    for m_dir in [m2md_tool_output_dir, delta_models_final_dir]:
        try:
            os.makedirs(m_dir, exist_ok=True)
            logger.info(f"Ensured directory exists: {m_dir}")
        except OSError as e:
            logger.error(f"Could not create directory {m_dir} for M2MD: {e}")
            return False
    if not os.path.exists(m2md_input_csv_path):
        logger.error(f"M2MD input CSV not found: {m2md_input_csv_path}.")
        return False
    try:
        df_m2md_pairs = pd.read_csv(m2md_input_csv_path)
        required_cols = ['Timepoint 1', 'Timepoint 2', 'Output'] 
        if not all(col in df_m2md_pairs.columns for col in required_cols):
            logger.error(f"M2MD input CSV {m2md_input_csv_path} is missing required columns: {required_cols}")
            return False
        if df_m2md_pairs.empty:
            logger.warning(f"M2MD input CSV {m2md_input_csv_path} is empty. No pairs to process.")
            return True 
    except Exception as e:
        logger.error(f"Error reading or validating M2MD input CSV {m2md_input_csv_path}: {e}")
        return False
    m2md_script_success = run_m2md_slicer_script(
        input_csv_for_pairs=m2md_input_csv_path, 
        m2md_tool_output_dir=m2md_tool_output_dir, 
        slicersalt_path=slicersalt_path
    )
    if not m2md_script_success:
        logger.error(f"M2MD Slicer script execution failed for {side} hemisphere.")
        return False
    m2md_results_summary_csv = os.path.join(m2md_tool_output_dir, "M2MD_results.csv") 
    if not os.path.exists(m2md_results_summary_csv):
        logger.error(f"M2MD Slicer script did not create the expected summary CSV: {m2md_results_summary_csv}")
        return False
    delta_gen_success = generate_delta_models_from_m2md_outputs(
        m2md_results_summary_csv=m2md_results_summary_csv, 
        final_delta_models_dir=delta_models_final_dir
    )
    if not delta_gen_success:
        logger.error(f"Generation of final delta models failed for {side} hemisphere.")
        return False
    logger.info(f"Longitudinal data preparation (M2MD & delta models) completed for {side} hemisphere.")
    return True

def run_m2md_slicer_script(input_csv_for_pairs: str, m2md_tool_output_dir: str, slicersalt_path: str) -> bool:
    logger.info(f"Preparing to run ModelToModelDistance Slicer script using CSV: {input_csv_for_pairs}")
    abs_input_csv = os.path.abspath(input_csv_for_pairs)
    abs_output_dir = os.path.abspath(m2md_tool_output_dir)
    script_fd, script_path = tempfile.mkstemp(suffix='.py', prefix='m2md_slicer_script_')
    try:
        # Minified the M2MD script content for brevity in this example
        m2md_script_content = f"""
# Dynamically generated Slicer script for ModelToModelDistance
import os, numpy as np, pandas as pd, slicer, logging
print(f"[M2MD Slicer Script] Started. Input CSV: {{abs_input_csv!r}}, Output Dir: {{abs_output_dir!r}}")
try: os.makedirs(r"{abs_output_dir}", exist_ok=True)
except Exception as e: print(f"[M2MD Slicer Script] ERROR creating output dir: {{e}}"); slicer.app.exit(1)
try:
    df_pairs = pd.read_csv(r"{abs_input_csv}")
    if df_pairs.empty: print(f"[M2MD Slicer Script] Input CSV empty. Exiting."); slicer.app.exit(0)
    if not all(c in df_pairs.columns for c in ['Timepoint 1','Timepoint 2','Output']):
        print(f"[M2MD Slicer Script] ERROR: CSV missing required columns."); slicer.app.exit(1)
    results = []
    print(f"[M2MD Slicer Script] Processing {{len(df_pairs)}} pairs.")
    for idx, r in df_pairs.iterrows():
        tp1, tp2, out_base = r['Timepoint 1'], r['Timepoint 2'], r['Output']
        print(f"[M2MD Slicer Script] Pair {{idx+1}}: {{out_base}} (TP1:{{tp1}}, TP2:{{tp2}})")
        if not os.path.exists(tp1) or not os.path.exists(tp2):
            print(f"  SKIPPING: File(s) not found."); results.append({{'Input_Output_BaseName':out_base,'Status':'Skipped - File Missing','Output_M2MD_VTK_Path':None,'Output_DistanceScalars_CSV_Path':None,'Input_Timepoint1_Path':tp1,'Input_Timepoint2_Path':tp2}}); continue
        ok, vtk_out, csv_out = False, os.path.join(r"{abs_output_dir}",out_base+"_m2md.vtk"), os.path.join(r"{abs_output_dir}",out_base+"_DistanceScalars.csv")
        try:
            slicer.mrmlScene.Clear(0); n1,n2 = slicer.util.loadModel(tp1),slicer.util.loadModel(tp2)
            if not n1 or not n2: raise RuntimeError("Model load fail")
            onode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode",out_base+"_M2MDResult")
            if not onode: raise RuntimeError("MRML node creation fail")
            params = {{'sourceModel':n1.GetID(),'targetModel':n2.GetID(),'outputModel':onode.GetID()}}
            print(f"  Running CLI..."); cli = slicer.cli.runSync(slicer.modules.modeltomodeldistance,None,params)
            if cli.GetStatusString()=="Completed":
                print(f"  CLI OK."); slicer.util.saveNode(onode,vtk_out)
                pd_ = onode.GetPolyData()
                if pd_ and pd_.GetPointData() and pd_.GetPointData().GetScalars("Distance"):
                    arr = pd_.GetPointData().GetScalars("Distance"); n = arr.GetNumberOfTuples()
                    np.savetxt(csv_out,np.array([arr.GetTuple1(i) for i in range(n)]),delimiter=",",fmt='%f',comments='')
                    print(f"  Saved scalars: {{csv_out}}"); ok=True
                else: print(f"  WARN: No 'Distance' scalars.")
            else: print(f"  ERR: CLI fail: {{cli.GetStatusString()}}")
        except Exception as e_p: print(f"  ERR pair proc: {{e_p}}"); import traceback; traceback.print_exc()
        results.append({{'Input_Output_BaseName':out_base,'Status':'Completed' if ok else 'Failed','Output_M2MD_VTK_Path':vtk_out if os.path.exists(vtk_out) else None,'Output_DistanceScalars_CSV_Path':csv_out if os.path.exists(csv_out) else None,'Input_Timepoint1_Path':tp1,'Input_Timepoint2_Path':tp2}})
    if results: pd.DataFrame(results).to_csv(os.path.join(r"{abs_output_dir}","M2MD_results.csv"),index=False); print(f"[M2MD Slicer Script] Summary saved.")
    else: print("[M2MD Slicer Script] No results to log.")
    print(f"[M2MD Slicer Script] Finished."); slicer.app.exit(0)
except Exception as e_m: print(f"[M2MD Slicer Script] CRITICAL ERR: {{e_m}}"); import traceback; traceback.print_exc(); slicer.app.exit(1)
"""
        with os.fdopen(script_fd, 'w') as f: f.write(m2md_script_content)
        slicer_cmd_list = [slicersalt_path, "--no-main-window", "--python-script", script_path]
        logger.info(f"Executing M2MD Slicer script: {' '.join(slicer_cmd_list)}")
        m2md_timeout = getattr(config, 'SPHARM_TIMEOUT_SECONDS', 7200)
        result = run_command(slicer_cmd_list, description="Running ModelToModelDistance via SlicerSALT", check=False, timeout=m2md_timeout)
        if result.stdout: logger.debug(f"M2MD Slicer script STDOUT:\n{result.stdout}") 
        if result.stderr: logger.debug(f"M2MD Slicer script STDERR:\n{result.stderr}") 
        if result.returncode == -1: logger.error(f"M2MD Slicer script call timed out."); return False
        if result.returncode != 0: logger.error(f"M2MD Slicer script failed (rc={result.returncode})."); return False
        summary_csv = os.path.join(abs_output_dir, "M2MD_results.csv")
        if not os.path.exists(summary_csv): logger.error(f"M2MD summary CSV missing: {summary_csv}"); return False
        logger.info(f"M2MD Slicer script completed. Summary: {summary_csv}"); return True
    except Exception as e: logger.error(f"M2MD setup/run error: {e}"); logger.exception("Traceback:"); return False
    finally:
        if os.path.exists(script_path):
            try: os.remove(script_path)
            except OSError as e: logger.warning(f"Could not remove temp M2MD script {script_path}: {e}")

def generate_delta_models_from_m2md_outputs(m2md_results_summary_csv: str, final_delta_models_dir: str) -> bool:
    logger.info(f"Generating delta models from M2MD summary: {m2md_results_summary_csv} -> {final_delta_models_dir}")
    if not os.path.exists(m2md_results_summary_csv): logger.error(f"M2MD summary CSV not found: {m2md_results_summary_csv}"); return False
    try: os.makedirs(final_delta_models_dir, exist_ok=True)
    except OSError as e: logger.error(f"Could not create delta models dir {final_delta_models_dir}: {e}"); return False
    try:
        df_results = pd.read_csv(m2md_results_summary_csv)
        if df_results.empty: logger.info("M2MD summary empty. No delta models."); return True
        req_cols = ['Input_Timepoint1_Path', 'Output_DistanceScalars_CSV_Path', 'Input_Output_BaseName', 'Status']
        if not all(c in df_results.columns for c in req_cols): logger.error(f"M2MD summary CSV missing cols from: {req_cols}"); return False
        s_count, a_count = 0, 0
        import vtk 
        for idx, r in df_results.iterrows():
            a_count +=1
            if r.get('Status') != 'Completed': logger.warning(f"Skipping delta for '{r['Input_Output_BaseName']}': M2MD status '{r.get('Status','Unknown')}'"); continue
            base_vtk, scalars_csv, out_base = r['Input_Timepoint1_Path'], r.get('Output_DistanceScalars_CSV_Path'), r['Input_Output_BaseName']
            delta_vtk_file = os.path.join(final_delta_models_dir, f"{out_base}_delta.vtk")
            if pd.isna(scalars_csv) or not scalars_csv: logger.warning(f"Skipping delta for '{out_base}': Scalars CSV path missing."); continue
            if not os.path.exists(base_vtk): logger.error(f"Skipping delta for '{out_base}': Base VTK missing: {base_vtk}"); continue
            if not os.path.exists(scalars_csv): logger.error(f"Skipping delta for '{out_base}': Scalars CSV missing: {scalars_csv}"); continue
            logger.info(f"Generating delta model for '{out_base}'")
            try:
                dists = pd.read_csv(scalars_csv, header=None).values.astype(float).flatten()
                reader = vtk.vtkPolyDataReader(); reader.SetFileName(base_vtk); reader.Update(); pdata = reader.GetOutput()
                if not pdata or pdata.GetNumberOfPoints()==0: logger.error(f"  Bad base VTK: {base_vtk}"); continue
                if pdata.GetNumberOfPoints()!=len(dists): logger.error(f"  Point mismatch for '{out_base}': VTK {pdata.GetNumberOfPoints()}, CSV {len(dists)}"); continue
                vtk_s = vtk.vtkDoubleArray(); vtk_s.SetName("DeltaValues"); vtk_s.SetNumberOfComponents(1)
                for v in dists: vtk_s.InsertNextTuple1(v)
                pdata.GetPointData().AddArray(vtk_s); pdata.GetPointData().SetActiveScalars("DeltaValues")
                writer = vtk.vtkPolyDataWriter(); writer.SetFileName(delta_vtk_file); writer.SetInputData(pdata); writer.SetFileVersion(42); writer.Write()
                logger.info(f"  OK: {delta_vtk_file}"); s_count +=1
            except Exception as e_v: logger.error(f"  ERR delta gen for '{out_base}': {e_v}"); logger.exception("  Traceback:")
        logger.info(f"Delta model gen done. Processed: {a_count}, Successful: {s_count}.")
        return s_count == (df_results[df_results['Status']=='Completed'].shape[0])
    except Exception as e_sum: logger.error(f"ERR proc M2MD summary {m2md_results_summary_csv}: {e_sum}"); logger.exception("Traceback:"); return False
