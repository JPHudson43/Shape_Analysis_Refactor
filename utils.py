"""
Utilities Module
--------------
Helper functions for file and data management.
"""
import subprocess
import os
import csv
import logging
import pandas as pd
import glob
import re # For regex to extract SubjectID
from typing import List, Optional, Union, Dict

import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_subject_id_from_path(file_path: str, regex_pattern: Optional[str]) -> str:
    """
    Extracts a subject ID from a file path using a regex pattern.
    The pattern should have one capturing group for the ID.
    """
    if not regex_pattern:
        logger.debug(f"No regex pattern provided for SubjectID extraction from {file_path}.")
        return "" # Or return filename without extension as a fallback

    file_name = os.path.basename(file_path)
    match = re.search(regex_pattern, file_name)
    if match and match.groups():
        return match.group(1) # Return the first captured group
    else:
        logger.warning(f"Could not extract SubjectID from filename '{file_name}' using pattern: {regex_pattern}")
        # Fallback: return filename without common SPHARM suffixes or .vtk
        base = file_name.replace(".vtk", "")
        suffixes_to_remove = ["_surfSPHARM_procalign", "_surfSPHARM", "_ellalign", "_para", "_surf", "_pp", "_reg", "_bin"] # Add more as needed
        for suffix in suffixes_to_remove:
            base = base.replace(suffix, "")
        return base # Return a cleaned up filename as a fallback ID

def dir_to_csv(side: str = "left", 
               output_csv_param: Optional[str] = None, 
               directory_path_param: Optional[str] = None) -> Union[str, Dict[str, str]]:
    """
    Create a CSV file with SubjectID and SubjectPath for all VTK files in a directory.
    SubjectID is extracted using config.SUBJECT_ID_REGEX_PATTERN.
    """
    if side == "both":
        logger.info("Processing dir_to_csv for both hemispheres.")
        left_csv = dir_to_csv(side="left", output_csv_param=None, directory_path_param=None)
        right_csv = dir_to_csv(side="right", output_csv_param=None, directory_path_param=None)
        return {"left": left_csv, "right": right_csv}
    
    try:
        directory_path = directory_path_param or config.get_paths_by_side(side, "spharm_extracted_models")
        if output_csv_param is None:
            output_csv = config.STATIC_LEFT_CSV if side == "left" else config.STATIC_RIGHT_CSV
        else:
            output_csv = output_csv_param
    except (ValueError, AttributeError) as e:
        logger.error(f"Error getting paths for dir_to_csv (side: {side}): {e}")
        return "" 

    logger.info(f"Preparing to create base CSV for {side} hemisphere.")
    logger.info(f"Source directory for VTK files: {directory_path}")
    logger.info(f"Output CSV file: {output_csv}")

    output_dir_for_csv = os.path.dirname(output_csv)
    if not os.path.exists(output_dir_for_csv):
        try:
            os.makedirs(output_dir_for_csv, exist_ok=True)
            logger.info(f"Created output directory for CSV: {output_dir_for_csv}")
        except OSError as e:
            logger.error(f"Failed to create directory {output_dir_for_csv}: {e}")
            return "" 

    vtk_file_paths = []
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        vtk_file_paths = get_file_paths_by_pattern(directory_path, "*.vtk")
    else:
        logger.warning(f"Directory not found or is not a directory: {directory_path}. CSV will be empty or only header.")
    
    subject_id_pattern = getattr(config, 'SUBJECT_ID_REGEX_PATTERN', None)
    if not subject_id_pattern:
        logger.warning("SUBJECT_ID_REGEX_PATTERN not defined in config. SubjectID column will be based on filename.")

    try:
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['SubjectID', 'SubjectPath']) # Header
            for path in vtk_file_paths:
                subject_id = extract_subject_id_from_path(path, subject_id_pattern)
                writer.writerow([subject_id, path])
        logger.info(f"Base CSV file created with {len(vtk_file_paths)} entries: {output_csv}")
    except IOError as e:
        logger.error(f"Failed to write CSV file {output_csv}: {e}")
        return "" 
        
    return output_csv

def merge_covariates_to_static_csv(static_csv_path: str, 
                                   master_covariates_csv_path: Optional[str] = None, 
                                   master_id_column: Optional[str] = None,
                                   static_id_column: str = "SubjectID"):
    """
    Merges covariates from a master CSV into a static analysis CSV.
    The static CSV (base) should have SubjectID and SubjectPath.
    The master CSV should have a matching SubjectID column and other covariate columns.
    The static CSV will be overwritten with the merged data.
    """
    master_path = master_covariates_csv_path or getattr(config, 'MASTER_COVARIATES_CSV', None)
    master_col_id = master_id_column or getattr(config, 'MASTER_COVARIATES_ID_COLUMN', None)

    if not master_path:
        logger.warning(f"MASTER_COVARIATES_CSV path not defined in config. Skipping covariate merge for {static_csv_path}.")
        return
    if not master_col_id:
        logger.warning(f"MASTER_COVARIATES_ID_COLUMN not defined in config. Skipping covariate merge for {static_csv_path}.")
        return
    if not os.path.exists(master_path):
        logger.warning(f"Master covariates file not found: {master_path}. Skipping covariate merge for {static_csv_path}.")
        return
    if not os.path.exists(static_csv_path):
        logger.warning(f"Base static CSV file not found: {static_csv_path}. Cannot merge covariates.")
        return

    logger.info(f"Attempting to merge covariates from '{master_path}' (ID col: '{master_col_id}') into '{static_csv_path}' (ID col: '{static_id_column}').")

    try:
        base_df = pd.read_csv(static_csv_path)
        master_df = pd.read_csv(master_path)

        if static_id_column not in base_df.columns:
            logger.error(f"Static ID column '{static_id_column}' not found in {static_csv_path}. Cannot merge.")
            return
        if master_col_id not in master_df.columns:
            logger.error(f"Master ID column '{master_col_id}' not found in {master_path}. Cannot merge.")
            return
        
        # Ensure ID columns are of the same type for merging, typically string.
        base_df[static_id_column] = base_df[static_id_column].astype(str)
        master_df[master_col_id] = master_df[master_col_id].astype(str)

        # Perform the merge
        # Keep all rows from base_df (left merge), add covariates from master_df
        # Suffixes are added if there are overlapping column names other than the ID
        merged_df = pd.merge(base_df, master_df, 
                             left_on=static_id_column, 
                             right_on=master_col_id, 
                             how='left',
                             suffixes=('', '_master'))
        
        # If the master_col_id was different from static_id_column and is now redundant, drop it.
        if static_id_column != master_col_id and master_col_id in merged_df.columns:
            merged_df = merged_df.drop(columns=[master_col_id])

        # Check for subjects in base_df that didn't find a match in master_df
        # These will have NaN values for the merged columns from master_df.
        num_unmatched = merged_df[master_df.columns.drop(master_col_id, errors='ignore')].isnull().all(axis=1).sum()
        if num_unmatched > 0:
            logger.warning(f"{num_unmatched} subjects in {static_csv_path} did not have matching covariates in {master_path}.")
            # Log some unmatched IDs for debugging
            unmatched_ids = base_df[~base_df[static_id_column].isin(master_df[master_col_id])][static_id_column].tolist()
            logger.debug(f"First few unmatched SubjectIDs from static CSV: {unmatched_ids[:5]}")


        merged_df.to_csv(static_csv_path, index=False)
        logger.info(f"Successfully merged covariates into {static_csv_path}. Final columns: {list(merged_df.columns)}")

    except Exception as e:
        logger.error(f"Error during covariate merge for {static_csv_path}: {e}")
        logger.exception("Detailed traceback for covariate merge error:")


def check_files(csv_file_path: str, file_column_header: str = "SubjectPath") -> List[str]:
    missing_files = []
    if not os.path.exists(csv_file_path):
        logger.error(f"CSV file not found: {csv_file_path}")
        return [f"CSV file not found: {csv_file_path}"] 
    try:
        df = pd.read_csv(csv_file_path)
        if file_column_header not in df.columns:
            logger.error(f"Column '{file_column_header}' not found in CSV file: {csv_file_path}")
            return [f"Column '{file_column_header}' not found in {csv_file_path}"]
        for idx, file_path in enumerate(df[file_column_header]):
            if pd.isna(file_path) or not str(file_path).strip(): 
                logger.warning(f"Empty or invalid file path at row {idx + 2} in column '{file_column_header}'.")
                continue
            file_path_str = str(file_path).strip()
            if not os.path.exists(file_path_str):
                missing_files.append(file_path_str)
                logger.warning(f"Missing file at row {idx + 2}: {file_path_str}")
    except pd.errors.EmptyDataError:
        logger.error(f"Error: CSV file {csv_file_path} is empty or improperly formatted.")
        return [f"Empty or invalid CSV: {csv_file_path}"]
    except Exception as e:
        logger.error(f"Error reading or processing CSV file {csv_file_path}: {e}")
        return [f"Error processing CSV: {csv_file_path}"]
    if missing_files:
        logger.warning(f"Found {len(missing_files)} missing files listed in {csv_file_path}.")
    else:
        logger.info(f"All files listed in column '{file_column_header}' of {csv_file_path} exist.")
    return missing_files

def excel_to_csv(excel_file_path: str, output_dir_param: Optional[str] = None) -> str:
    output_dir = output_dir_param or config.CSV_REPORTS_DIR 
    if not os.path.exists(excel_file_path):
        logger.error(f"Excel file not found: {excel_file_path}")
        return output_dir 
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Ensured output directory for CSVs exists: {output_dir}")
    except OSError as e:
        logger.error(f"Could not create output directory {output_dir}: {e}")
        return output_dir
    try:
        xls = pd.ExcelFile(excel_file_path)
        logger.info(f"Processing Excel file: {excel_file_path}. Sheets: {xls.sheet_names}")
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            safe_sheet_name = "".join(c if c.isalnum() else "_" for c in sheet_name)
            csv_file_name = f"{safe_sheet_name}.csv"
            csv_file_path = os.path.join(output_dir, csv_file_name)
            df.to_csv(csv_file_path, index=False)
            logger.info(f"Saved sheet '{sheet_name}' as '{csv_file_path}'")
        logger.info(f"All sheets from {excel_file_path} have been converted to CSV in {output_dir}")
    except Exception as e:
        logger.error(f"Error converting Excel to CSV: {str(e)}")
    return output_dir

def run_command(command: Union[List[str], str], 
                description: str = "", 
                check: bool = True, 
                timeout: Optional[int] = None, 
                shell: bool = False) -> subprocess.CompletedProcess:
    cmd_str_for_log = ' '.join(command) if isinstance(command, list) else command
    if description:
        logger.info(description)
    logger.info(f"Executing command: {cmd_str_for_log}") 
    if shell and isinstance(command, list):
        logger.warning("Executing command as list with shell=True. This is unusual. Ensure this is intended.")
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=check, timeout=timeout, shell=shell          
        )
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Command '{cmd_str_for_log}' failed with return code {e.returncode}.")
        logger.error(f"STDOUT:\n{e.stdout}")
        logger.error(f"STDERR:\n{e.stderr}")
        if not check: 
            return subprocess.CompletedProcess(e.cmd, e.returncode, e.stdout, e.stderr)
        raise 
    except FileNotFoundError:
        cmd_name = command[0] if isinstance(command, list) else command.split()[0]
        logger.error(f"Command not found: {cmd_name}. Ensure it's in PATH or provide full path.")
        if not check:
            return subprocess.CompletedProcess(command, -2, "", "File not found") 
        raise
    except subprocess.TimeoutExpired as e:
        logger.error(f"Command '{cmd_str_for_log}' timed out after {e.timeout} seconds.")
        stdout_decoded = e.stdout.decode(errors='ignore') if isinstance(e.stdout, bytes) else e.stdout
        stderr_decoded = e.stderr.decode(errors='ignore') if isinstance(e.stderr, bytes) else e.stderr
        logger.error(f"Timeout STDOUT:\n{stdout_decoded}")
        logger.error(f"Timeout STDERR:\n{stderr_decoded}")
        if not check:
             return subprocess.CompletedProcess(e.cmd or command, -1, stdout_decoded, stderr_decoded) 
        raise
    except Exception as e_general: 
        logger.error(f"An unexpected error occurred while running command '{cmd_str_for_log}': {e_general}")
        logger.exception("Details of unexpected error in run_command:") 
        if not check:
            return subprocess.CompletedProcess(command, -99, "", str(e_general))
        raise

def get_file_paths_by_pattern(directory: str, pattern: str) -> List[str]:
    if not os.path.isdir(directory): 
        logger.warning(f"Directory not found or is not a directory: {directory} (for pattern '{pattern}')")
        return []
    search_pattern = os.path.join(directory, pattern)
    matched_files = glob.glob(search_pattern)
    if not matched_files:
        logger.debug(f"No files matching '{pattern}' found in {directory}")
    return sorted(matched_files) 

def normalize_path_for_platform(path: str) -> str:
    if path is None: 
        return ""
    normalized_path = str(path).replace('\\', '/') 
    return normalized_path

def print_section_header(title: str, width: int = 100):
    header_line = f"\n{'-' * width}\n{title.center(width)}\n{'-' * width}\n"
    log_message = header_line.strip().replace('\n', ' | ') 
    logger.info(log_message)
    print(header_line) 

def print_box_header(title: str):
    box_width = len(title) + 10
    border_top_bottom = f"╔{'═' * box_width}╗"
    title_line_formatted = f"║{title.center(box_width)}║"
    border_bottom = f"╚{'═' * box_width}╝"
    full_header = f"\n{border_top_bottom}\n{title_line_formatted}\n{border_bottom}\n"
    log_message = full_header.strip().replace('\n', ' | ')
    logger.info(log_message)
    print(full_header)