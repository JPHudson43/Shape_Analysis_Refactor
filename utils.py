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
from typing import List, Optional, Union, Dict

import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def dir_to_csv(side: str = "left", 
              output_csv: Optional[str] = None, 
              directory_path: Optional[str] = None) -> str:
    """
    Create a CSV file containing all VTK files in a directory.
    
    Parameters:
    -----------
    side : str
        Hemisphere to process ("left", "right", or "both").
    output_csv : str, optional
        Path to the output CSV file. If None, defaults to config.STATIC_LEFT_CSV or config.STATIC_RIGHT_CSV.
    directory_path : str, optional
        Path to the directory containing VTK files. If None, defaults to SPHARM_MODELS_*_DIR from config.
    
    Returns:
    --------
    str
        Path to the created CSV file.
    """
    if side == "both":
        # Process both hemispheres
        left_csv = dir_to_csv(side="left", output_csv=None, directory_path=None)
        right_csv = dir_to_csv(side="right", output_csv=None, directory_path=None)
        return {"left": left_csv, "right": right_csv}
    
    # Get default values based on side
    if not directory_path:
        directory_path = config.get_paths_by_side(side, "spharm_models")
    
    if not output_csv:
        output_csv = config.get_paths_by_side(side, "static_csv")
    
    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(output_csv)
    os.makedirs(output_dir, exist_ok=True)
    
    # List all VTK files in the directory
    file_paths = []
    if os.path.exists(directory_path):
        for file in os.listdir(directory_path):
            if file.endswith('.vtk'):
                file_paths.append(os.path.join(directory_path, file))
        
        # Sort file paths for consistency
        file_paths.sort()
    else:
        logger.warning(f"Directory not found: {directory_path}")
    
    # Write file paths to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(['Subject'])
        
        # Write file paths
        for path in file_paths:
            writer.writerow([path])
    
    logger.info(f"CSV file created with {len(file_paths)} VTK files from {directory_path}")
    logger.info(f"Output CSV: {output_csv}")
    
    return output_csv

def check_files(csv_file_path: str) -> List[str]:
    """
    Check if all files specified in a CSV file exist.
    
    Parameters:
    -----------
    csv_file_path : str
        Path to the CSV file containing file paths.
    
    Returns:
    --------
    list
        List of missing files.
    """
    missing_files = []
    
    if not os.path.exists(csv_file_path):
        logger.error(f"CSV file not found: {csv_file_path}")
        return ["CSV file not found"]
    
    with open(csv_file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        
        # Skip header if present
        try:
            header = next(reader)
        except StopIteration:
            logger.error(f"Error: CSV file {csv_file_path} is empty.")
            return ["Empty CSV file"]
        
        # Check each file path in the CSV
        for row_idx, row in enumerate(reader, start=2):  # Start from 2 to account for header
            for col_idx, file_path in enumerate(row, start=1):
                # Check if the file path is not empty and the file does not exist
                if file_path and not os.path.exists(file_path):
                    missing_files.append(file_path)
                    logger.warning(f"Missing file at row {row_idx}, column {col_idx}: {file_path}")
    
    if missing_files:
        logger.warning(f"Found {len(missing_files)} missing files")
    else:
        logger.info("All files in the CSV exist")
        
    return missing_files

def excel_to_csv(excel_file_path: str, output_dir: Optional[str] = None) -> str:
    """
    Convert an Excel file to CSV files, one per sheet, each in its own subdirectory.
    
    Parameters:
    -----------
    excel_file_path : str
        Path to the Excel file.
    output_dir : str, optional
        Directory to store the CSV files.
        If None, the CSV_DIR from config will be used.
    
    Returns:
    --------
    str
        Path to the output directory.
    """
    # Set output directory
    output_dir = output_dir or config.CSV_DIR
    
    # Check if Excel file exists
    if not os.path.exists(excel_file_path):
        logger.error(f"Excel file not found: {excel_file_path}")
        return output_dir
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Read the Excel file
        xls = pd.ExcelFile(excel_file_path)
        
        # Process each sheet
        for sheet_name in xls.sheet_names:
            # Read the sheet into a DataFrame
            df = pd.read_excel(xls, sheet_name=sheet_name)
            
            # Create a subdirectory named after the sheet
            sub_dir_path = os.path.join(output_dir, sheet_name)
            os.makedirs(sub_dir_path, exist_ok=True)
            
            # Create a CSV file name based on the sheet name
            csv_file_name = f"{sheet_name}.csv"
            csv_file_path = os.path.join(sub_dir_path, csv_file_name)
            
            # Save the DataFrame to CSV
            df.to_csv(csv_file_path, index=False)
            
            logger.info(f"Saved sheet '{sheet_name}' as '{csv_file_name}' in '{sub_dir_path}'")
        
        logger.info(f"All sheets from {excel_file_path} have been converted to CSV")
    
    except Exception as e:
        logger.error(f"Error converting Excel to CSV: {str(e)}")
    
    return output_dir

def run_command(command: Union[str, List[str]], 
               description: Optional[str] = None, 
               check: bool = True) -> subprocess.CompletedProcess:
    """
    Run a shell command with proper logging and error handling.
    
    Parameters:
    -----------
    command : str or list
        The command to run, either as a string or list of arguments
    description : str, optional
        Description of the command for logging
    check : bool
        Whether to raise an exception on command failure
    
    Returns:
    --------
    subprocess.CompletedProcess
        Result of the command execution
    """
    cmd_str = command if isinstance(command, str) else " ".join(command)
    
    if description:
        logger.info(f"{description}")
    logger.debug(f"Running command: {cmd_str}")
    
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=isinstance(command, str),
            check=check
        )
        if result.returncode == 0:
            logger.debug("Command completed successfully")
        else:
            logger.warning(f"Command returned non-zero exit code: {result.returncode}")
            logger.debug(f"Command stderr: {result.stderr}")
        
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        logger.error(f"Error output: {e.stderr}")
        if check:
            raise
        return e
    except Exception as e:
        logger.error(f"Error executing command: {str(e)}")
        if check:
            raise
        # Create a mock CompletedProcess with error info
        return subprocess.CompletedProcess(
            args=cmd_str,
            returncode=-1,
            stdout="",
            stderr=str(e)
        )

def get_file_paths_by_pattern(directory: str, pattern: str) -> List[str]:
    """
    Get a list of file paths in a directory matching a glob pattern.
    
    Parameters:
    -----------
    directory : str
        Directory to search in
    pattern : str
        File pattern to match (e.g., "*.nii.gz")
    
    Returns:
    --------
    list
        List of matching file paths
    """
    import glob
    
    search_pattern = os.path.join(directory, pattern)
    matched_files = glob.glob(search_pattern)
    
    if not matched_files:
        logger.warning(f"No files matching '{pattern}' found in {directory}")
    
    return sorted(matched_files)

def normalize_path_for_platform(path: str) -> str:
    """
    Normalize a file path for the current platform.
    
    Parameters:
    -----------
    path : str
        Path to normalize
    
    Returns:
    --------
    str
        Normalized path
    """
    # Convert backslashes to forward slashes
    normalized_path = path.replace('\\', '/')
    
    # Handle Windows drive letters for UNIX-like tools
    if os.name == 'nt':  # Windows system
        if len(path) > 2 and path[1] == ':':
            drive_letter = path[0].lower()
            normalized_path = normalized_path.replace(f'{drive_letter}:/', f'/{drive_letter}/')
    
    return normalized_path

def print_section_header(title: str, width: int = 100):
    """Print a formatted section header."""
    print(f"\n{'-' * width}")
    print(f"{title.center(width)}")
    print(f"{'-' * width}\n")

def print_box_header(title: str):
    """Print a fancy box header."""
    box_width = len(title) + 10
    print(f"\n╔{'═' * box_width}╗")
    print(f"║{title.center(box_width)}║")
    print(f"╚{'═' * box_width}╝\n")
