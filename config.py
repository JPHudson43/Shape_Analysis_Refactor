"""
Master configuration file for Hippocampal Morphometry Pipeline.
Edit this file to configure paths and parameters for your analysis.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
import logging # Added for logging within config functions

# Configure basic logging for messages from this module
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===============================================================================
# GENERAL CONFIGURATION
# ===============================================================================

# Base directory to store all data
BASE_DIR = "/home/jeremy/Desktop/hippo_morph"

# Number of CPU cores to use for parallel processing
MAX_WORKERS = 4 # Default, can be overridden

# SPHARM-PDM command timeout in seconds (e.g., 7200 for 2 hours)
# Set to None for no timeout.
SPHARM_TIMEOUT_SECONDS = 7200

# Behavior for preprocessing steps (binarization, registration) if no input files are found
# If True, the step will be marked as failed.
# If False, the step will log a warning and be marked as successful (as there was nothing to process).
FAIL_PIPELINE_ON_EMPTY_INPUT_MASKS = False

# ===============================================================================
# RUNTIME CONFIGURATION (can be overridden by command-line arguments)
# ===============================================================================

# Pipeline execution options
SKIP_SEGMENTATION = False
SKIP_BINARIZATION = False
SKIP_REGISTRATION = False
SKIP_SPHARM = False
SKIP_STATS = False
SKIP_LONGITUDINAL = False
WITH_GUI = False        # If GUI elements should be attempted
NON_INTERACTIVE = False # If True, suppress all prompts for user confirmation

# Processing options
SIDE = "both"  # "left", "right", or "both"
# ALIGNMENT_TYPE for SPHARM model extraction. Example: "surfSPHARM_procalign" or just "procalign"
# This string should match the part of the SPHARM output filename that identifies the desired alignment.
# e.g., if files are subject_L_surfSPHARM_procalign.vtk, ALIGNMENT_TYPE = "surfSPHARM_procalign"
ALIGNMENT_TYPE = "procrustesalign"

# ===============================================================================
# EXTERNAL TOOLS
# ===============================================================================

# Path to SlicerSALT executable
SLICERSALT_PATH = os.path.join(BASE_DIR, "SlicerSALT-5.0.0-linux-amd64/SlicerSALT")

# Path to Hippodeep script (assuming deepseg1.sh is the entry point)
HIPPODEEP_PATH = os.path.join(BASE_DIR, "hippodeep_pytorch/deepseg1.sh")

def get_slicersalt_module_path(module_name: str, slicer_version_specific_path_part: str, module_type: str = "CommandLineTool") -> str:
    """
    Get the path to a SlicerSALT module.
    slicer_version_specific_path_part: e.g., "SlicerSALT-5.3" or "SlicerSALT-5.1"
    module_type can be "CommandLineTool", "qt-scripted-modules", etc.
    """
    slicer_root = os.path.dirname(SLICERSALT_PATH) # e.g., /path/to/SlicerSALT-X.Y.Z-platform
    
    if module_type == "CommandLineTool":
        return os.path.join(slicer_root, "share", slicer_version_specific_path_part, "CommandLineTool", f"{module_name}.py")
    elif module_type == "qt-scripted-modules":
        return os.path.join(slicer_root, "lib", slicer_version_specific_path_part, "qt-scripted-modules", f"{module_name}.py")
    else:
        raise ValueError(f"Unknown SlicerSALT module type: {module_type}")

# Specific module paths - verify these carefully for your SlicerSALT version
# The version part (e.g., "SlicerSALT-5.3") might differ between modules or SlicerSALT releases.
SPHARM_PDM_PATH = get_slicersalt_module_path("SPHARM-PDM", "SlicerSALT-5.3", "CommandLineTool")
# MFSDA_RUN_PATH = get_slicersalt_module_path("MFSDA_run", "SlicerSALT-5.1", "qt-scripted-modules")
# MFSDA_CREATE_SHAPES_PATH = get_slicersalt_module_path("MFSDA_createShapes", "SlicerSALT-5.1", "qt-scripted-modules")

# ===============================================================================
# DIRECTORY STRUCTURE
# ===============================================================================

def build_path(*parts: str) -> str:
    """Build an absolute path from BASE_DIR and additional parts."""
    return os.path.join(BASE_DIR, *parts)

# Input T1-weighted MRI images - User specified absolute path
T1_INPUT_DIR = "/home/jeremy/Desktop/hippo_morph/Input_T1s/"

# Segmentation outputs (where Hippodeep initially places files or where they are moved)
SEG_OUTPUT_ROOT_DIR = build_path("data", "segmentation_output")
SEG_LEFT_MASKS_DIR = os.path.join(SEG_OUTPUT_ROOT_DIR, "masks_L")
SEG_RIGHT_MASKS_DIR = os.path.join(SEG_OUTPUT_ROOT_DIR, "masks_R")
SEG_CEREBRUM_MASKS_DIR = os.path.join(SEG_OUTPUT_ROOT_DIR, "masks_cerebrum")
SEG_BRAIN_MASKS_DIR = os.path.join(SEG_OUTPUT_ROOT_DIR, "masks_brain")
SEG_VOLUME_REPORTS_DIR = os.path.join(SEG_OUTPUT_ROOT_DIR, "volume_reports")

# Binarized masks
BINARIZED_MASKS_ROOT_DIR = build_path("data", "binarized_masks")
BINARIZED_LEFT_MASKS_DIR = os.path.join(BINARIZED_MASKS_ROOT_DIR, "masks_L")
BINARIZED_RIGHT_MASKS_DIR = os.path.join(BINARIZED_MASKS_ROOT_DIR, "masks_R")

# Registered/aligned masks (output of FSL FLIRT)
REGISTERED_MASKS_ROOT_DIR = build_path("data", "registered_masks")
REGISTERED_LEFT_MASKS_DIR = os.path.join(REGISTERED_MASKS_ROOT_DIR, "masks_L")
REGISTERED_RIGHT_MASKS_DIR = os.path.join(REGISTERED_MASKS_ROOT_DIR, "masks_R")

# SPHARM computation outputs
SPHARM_PIPELINE_OUTPUT_ROOT_DIR = build_path("output", "spharm_processing")
SPHARM_TOOL_LEFT_OUTPUT_DIR = os.path.join(SPHARM_PIPELINE_OUTPUT_ROOT_DIR, "tool_output_L") # Raw SPHARM-PDM output
SPHARM_TOOL_RIGHT_OUTPUT_DIR = os.path.join(SPHARM_PIPELINE_OUTPUT_ROOT_DIR, "tool_output_R")# Raw SPHARM-PDM output
SPHARM_EXTRACTED_MODELS_ROOT_DIR = build_path("output", "spharm_extracted_models") # Copied final models
SPHARM_EXTRACTED_MODELS_LEFT_DIR = os.path.join(SPHARM_EXTRACTED_MODELS_ROOT_DIR, "left")
SPHARM_EXTRACTED_MODELS_RIGHT_DIR = os.path.join(SPHARM_EXTRACTED_MODELS_ROOT_DIR, "right")

# Statistical analysis directories
STATS_ROOT_DIR = build_path("output", "statistics")
STATS_STATIC_DIR = os.path.join(STATS_ROOT_DIR, "static")
STATS_STATIC_LEFT_DIR = os.path.join(STATS_STATIC_DIR, "left")
STATS_STATIC_RIGHT_DIR = os.path.join(STATS_STATIC_DIR, "right")
STATS_LONGITUDINAL_DIR = os.path.join(STATS_ROOT_DIR, "longitudinal")
STATS_LONGITUDINAL_LEFT_DIR = os.path.join(STATS_LONGITUDINAL_DIR, "left")
STATS_LONGITUDINAL_RIGHT_DIR = os.path.join(STATS_LONGITUDINAL_DIR, "right")

# CSV directory for storing tabular data
CSV_REPORTS_DIR = build_path("output", "csv_reports")

# Templates directory (contains reference files, etc.)
TEMPLATES_DIR = build_path("templates")

# ===============================================================================
# TEMPLATES AND REFERENCES (ensure these files exist in TEMPLATES_DIR)
# ===============================================================================

LEFT_SPHERE_TEMPLATE = os.path.join(TEMPLATES_DIR, "MCALT_T1_mask_L_bin_pp_surf_SPHARM.vtk")
RIGHT_SPHERE_TEMPLATE = os.path.join(TEMPLATES_DIR, "MCALT_T1_mask_R_bin_pp_surf_SPHARM.vtk")
LEFT_FLIP_TEMPLATE = os.path.join(TEMPLATES_DIR, "MCALT_T1_mask_L_bin_pp_surf_SPHARM.coef")
RIGHT_FLIP_TEMPLATE = os.path.join(TEMPLATES_DIR, "MCALT_T1_mask_R_bin_pp_surf_SPHARM.coef")
LEFT_REFERENCE_IMAGE = os.path.join(TEMPLATES_DIR, "MCALT_T1_mask_L_bin.nii.gz")
RIGHT_REFERENCE_IMAGE = os.path.join(TEMPLATES_DIR, "MCALT_T1_mask_R_bin.nii.gz")

# ===============================================================================
# PROCESSING PARAMETERS
# ===============================================================================

FLIRT_OPTIONS = "-searchcost labeldiff -cost labeldiff -dof 6 -interp nearestneighbour"
SPHARM_PARAMS = {
    "rescale": "False", "space": "0.5,0.5,0.5", "label": "1",
    "gauss": "False", "var": "10,10,10", "iter": "1000",
    "subdivLevel": "10", "spharmDegree": "12", "medialMesh": "False",
    "phiIteration": "100", "thetaIteration": "100",
    "regParaTemplateFileOn": "True", "flipTemplateOn": "True",
    "flip": "0" # Default, can be overridden per hemisphere if templates are one-sided
}

# ===============================================================================
# CSV FILES FOR STATISTICS (output locations)
# ===============================================================================

STATIC_LEFT_CSV = os.path.join(CSV_REPORTS_DIR, "static_left_hippocampal_stats.csv")
STATIC_RIGHT_CSV = os.path.join(CSV_REPORTS_DIR, "static_right_hippocampal_stats.csv")
LONGITUDINAL_LEFT_M2MD_CSV = os.path.join(CSV_REPORTS_DIR, "longitudinal_left_m2md.csv")
LONGITUDINAL_RIGHT_M2MD_CSV = os.path.join(CSV_REPORTS_DIR, "longitudinal_right_m2md.csv")
LONGITUDINAL_STATS_LEFT_CSV = os.path.join(CSV_REPORTS_DIR, "longitudinal_stats_left.csv")
LONGITUDINAL_STATS_RIGHT_CSV = os.path.join(CSV_REPORTS_DIR, "longitudinal_stats_right.csv")

# ===============================================================================
# DIRECTORY AND ENVIRONMENT MANAGEMENT
# ===============================================================================

ALL_DIRECTORIES_TO_CREATE = {
    # Input directories like T1_INPUT_DIR and TEMPLATES_DIR are expected to exist.
    "segmentation_output": {
        "main": SEG_OUTPUT_ROOT_DIR, "left_masks": SEG_LEFT_MASKS_DIR, "right_masks": SEG_RIGHT_MASKS_DIR,
        "cerebrum_masks": SEG_CEREBRUM_MASKS_DIR, "brain_masks": SEG_BRAIN_MASKS_DIR,
        "volume_reports": SEG_VOLUME_REPORTS_DIR
    },
    "binarized_masks": {
        "main": BINARIZED_MASKS_ROOT_DIR, "left": BINARIZED_LEFT_MASKS_DIR, "right": BINARIZED_RIGHT_MASKS_DIR
    },
    "registered_masks": {
        "main": REGISTERED_MASKS_ROOT_DIR, "left": REGISTERED_LEFT_MASKS_DIR, "right": REGISTERED_RIGHT_MASKS_DIR
    },
    "spharm_processing_outputs": {
        "root": SPHARM_PIPELINE_OUTPUT_ROOT_DIR,
        "tool_left_out": SPHARM_TOOL_LEFT_OUTPUT_DIR, "tool_right_out": SPHARM_TOOL_RIGHT_OUTPUT_DIR,
        "extracted_models_root": SPHARM_EXTRACTED_MODELS_ROOT_DIR,
        "extracted_left": SPHARM_EXTRACTED_MODELS_LEFT_DIR, "extracted_right": SPHARM_EXTRACTED_MODELS_RIGHT_DIR
    },
    "statistics_outputs": {
        "main": STATS_ROOT_DIR,
        "static_root": STATS_STATIC_DIR, "static_left": STATS_STATIC_LEFT_DIR, "static_right": STATS_STATIC_RIGHT_DIR,
        "longitudinal_root": STATS_LONGITUDINAL_DIR,
        "longitudinal_left": STATS_LONGITUDINAL_LEFT_DIR, "longitudinal_right": STATS_LONGITUDINAL_RIGHT_DIR
    },
    "csv_reports": CSV_REPORTS_DIR,
}

def create_directories(non_interactive_flag: bool = NON_INTERACTIVE):
    """Create all required pipeline output directories if they don't exist."""
    logger.info("Initializing project directories...")
    created_count = 0
    skipped_count = 0

    # Check fundamental directories first
    if not os.path.exists(BASE_DIR):
        logger.error(f"CRITICAL: Base directory {BASE_DIR} does not exist. Cannot proceed.")
        return False # Indicate failure
    if not os.path.exists(T1_INPUT_DIR):
        logger.warning(f"Input T1 directory {T1_INPUT_DIR} does not exist. Please ensure it's correctly specified and populated.")
        # Depending on workflow, this might be an error or just a warning if T1s are generated by a prior step.
    if not os.path.exists(TEMPLATES_DIR):
        logger.warning(f"Templates directory {TEMPLATES_DIR} does not exist. Please ensure it's correctly specified and populated.")


    def _create_dirs_recursive(item):
        nonlocal created_count, skipped_count
        if isinstance(item, dict):
            for k, v in item.items():
                _create_dirs_recursive(v)
        elif isinstance(item, str): # It's a path string
            if not os.path.exists(item):
                try:
                    os.makedirs(item, exist_ok=True)
                    logger.info(f"Created directory: {item}")
                    created_count += 1
                except OSError as e:
                    logger.error(f"Failed to create directory {item}: {e}")
            else:
                # logger.debug(f"Directory already exists: {item}") # Can be too verbose
                skipped_count += 1
    
    _create_dirs_recursive(ALL_DIRECTORIES_TO_CREATE)
    
    summary_message = f"Directory initialization complete. Created: {created_count}, Already existed/Skipped: {skipped_count}."
    logger.info(summary_message)
    if created_count > 0 and not non_interactive_flag:
        print(summary_message) # Also print to console if interactive and changes were made
    return True


def check_environment():
    """Check if required external tools and template files are available."""
    logger.info("Performing environment check...")
    issues = []
    
    # Check SlicerSALT path
    if not os.path.exists(SLICERSALT_PATH) or not os.access(SLICERSALT_PATH, os.X_OK):
        issues.append(f"SlicerSALT executable not found or not executable at {SLICERSALT_PATH}")
    
    # Check Hippodeep script path
    if not os.path.exists(HIPPODEEP_PATH) or not os.access(HIPPODEEP_PATH, os.X_OK):
        issues.append(f"Hippodeep script not found or not executable at {HIPPODEEP_PATH}")
    
    # Check for FSL FLIRT and FSLMATHS
    try:
        flirt_check = subprocess.run(["flirt", "-version"], capture_output=True, text=True, check=False)
        if flirt_check.returncode != 0:
            issues.append("FSL FLIRT (flirt -version) command failed or not found in PATH.")
    except FileNotFoundError:
        issues.append("FSL FLIRT (flirt) command not found. Ensure FSL is installed and in PATH.")
        
    try:
        fslmaths_check = subprocess.run(["fslmaths", "-version"], capture_output=True, text=True, check=False)
        # fslmaths -version might not return 0, check if it produces output or error indicating presence
        if fslmaths_check.returncode != 0 and not fslmaths_check.stdout and not fslmaths_check.stderr : # crude check
             issues.append("FSL MATHS (fslmaths -version) command failed or not found in PATH.")
    except FileNotFoundError:
        issues.append("FSL MATHS (fslmaths) command not found. Ensure FSL is installed and in PATH.")

    # Check for template files
    template_files_to_check = {
        "Left Sphere Template": LEFT_SPHERE_TEMPLATE,
        "Right Sphere Template": RIGHT_SPHERE_TEMPLATE,
        "Left Flip Template": LEFT_FLIP_TEMPLATE,
        "Right Flip Template": RIGHT_FLIP_TEMPLATE,
        "Left Reference Image": LEFT_REFERENCE_IMAGE,
        "Right Reference Image": RIGHT_REFERENCE_IMAGE
    }
    
    missing_templates_details = []
    for name, path in template_files_to_check.items():
        if not os.path.exists(path):
            missing_templates_details.append(f"  - {name}: {path}")
            
    if missing_templates_details:
        issues.append("The following template files are missing:")
        issues.extend(missing_templates_details)
        
    if issues:
        logger.warning("--- ENVIRONMENT CHECK FOUND ISSUES ---")
        for issue in issues:
            logger.warning(f"  - {issue}")
        logger.warning("--- Please resolve these issues for full pipeline functionality. ---")
        if not NON_INTERACTIVE:
            print("\nWARNING: Environment check found issues (see logs). Please resolve for full functionality.")
        return False
    else:
        logger.info("Environment check passed. Required tools and template files appear to be available.")
        if not NON_INTERACTIVE:
            print("\nEnvironment check passed.")
        return True

def get_paths_by_side(side_label: str, component_type: str) -> str:
    """
    Get relevant paths based on hemisphere and component type.

    Parameters:
    -----------
    side_label : str
        Hemisphere ("left" or "right").
    component_type : str
        Component type. Valid options:
        "seg_mask_input" (segmentation masks for binarization - e.g. from SEG_LEFT_MASKS_DIR),
        "bin_mask_input" (binarized masks for registration - e.g. from BINARIZED_LEFT_MASKS_DIR),
        "reg_mask_input" (registered masks for SPHARM - e.g. from REGISTERED_LEFT_MASKS_DIR),
        "spharm_tool_output" (raw output directory for SPHARM-PDM tool),
        "spharm_extracted_models" (directory for final extracted SPHARM models),
        "sphere_template",
        "flip_template",
        "reference_image" (for FLIRT registration)

    Returns:
    --------
    str
        The requested path.

    Raises:
    -------
    ValueError
        If side_label or component_type is invalid.
    """
    if side_label not in ["left", "right"]:
        raise ValueError(f"Invalid side_label: {side_label}. Must be 'left' or 'right'.")

    path_map = {
        "left": {
            "seg_mask_input": SEG_LEFT_MASKS_DIR,
            "bin_mask_input": BINARIZED_LEFT_MASKS_DIR,
            "reg_mask_input": REGISTERED_LEFT_MASKS_DIR,
            "spharm_tool_output": SPHARM_TOOL_LEFT_OUTPUT_DIR,
            "spharm_extracted_models": SPHARM_EXTRACTED_MODELS_LEFT_DIR,
            "sphere_template": LEFT_SPHERE_TEMPLATE,
            "flip_template": LEFT_FLIP_TEMPLATE,
            "reference_image": LEFT_REFERENCE_IMAGE,
        },
        "right": {
            "seg_mask_input": SEG_RIGHT_MASKS_DIR,
            "bin_mask_input": BINARIZED_RIGHT_MASKS_DIR,
            "reg_mask_input": REGISTERED_RIGHT_MASKS_DIR,
            "spharm_tool_output": SPHARM_TOOL_RIGHT_OUTPUT_DIR,
            "spharm_extracted_models": SPHARM_EXTRACTED_MODELS_RIGHT_DIR,
            "sphere_template": RIGHT_SPHERE_TEMPLATE,
            "flip_template": RIGHT_FLIP_TEMPLATE,
            "reference_image": RIGHT_REFERENCE_IMAGE,
        }
    }

    try:
        return path_map[side_label][component_type]
    except KeyError:
        raise ValueError(f"Invalid component_type '{component_type}' for side '{side_label}'. Valid types are: {list(path_map['left'].keys())}")
