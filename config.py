"""
Master configuration file for Hippocampal Morphometry Pipeline.
Edit this file to configure paths and parameters for your analysis.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
import logging 

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===============================================================================
# GENERAL CONFIGURATION
# ===============================================================================
BASE_DIR = "/home/jeremy/Desktop/hippo_morph"
MAX_WORKERS = 4 
SPHARM_TIMEOUT_SECONDS = 3600 * 4 # Example: 4 hours. Adjust as needed.
MFSDA_TIMEOUT_SECONDS = 120 
FAIL_PIPELINE_ON_EMPTY_INPUT_MASKS = False
ROI_NAME = "hippo" # Still used for ROI-specific subdirectories in SPHARM output and potentially other logic

# --- Configuration for Automated Covariate Merging (if used later) ---
MASTER_COVARIATES_CSV = os.path.join(BASE_DIR, "project_data", "master_covariates.csv") 
MASTER_COVARIATES_ID_COLUMN = "SubjectID" 
SUBJECT_ID_REGEX_PATTERN = r"(ADNI_\d{3}_S_\d{4})"

# ===============================================================================
# RUNTIME CONFIGURATION 
# ===============================================================================
SKIP_SEGMENTATION = False 
SKIP_BINARIZATION = False
SKIP_REGISTRATION = False 
SKIP_SPHARM = False
# SKIP_STATS = False # This was the old general stats skip flag
SKIP_MFSDA_STATS = True # <<< ADD THIS LINE (Set to True to skip MFSDA)
SKIP_LONGITUDINAL_PREP = False # If you have a separate flag for M2MD prep
# SKIP_LONGITUDINAL = False # This was the old general longitudinal skip flag
WITH_GUI = False        
NON_INTERACTIVE = True 

SIDE = "both"  
ALIGNMENT_TYPE = "SPHARM_procalign" 

# ===============================================================================
# EXTERNAL TOOLS
# ===============================================================================
SLICERSALT_PATH = os.path.join(BASE_DIR, "SlicerSALT-5.0.0-linux-amd64/SlicerSALT")
HIPPODEEP_PATH = os.path.join(BASE_DIR, "hippodeep_pytorch/deepseg1.sh") 

def get_slicersalt_module_path(module_name: str, slicer_version_specific_path_part: str, module_type: str = "CommandLineTool") -> str:
    slicer_root = os.path.dirname(SLICERSALT_PATH) 
    if module_type == "CommandLineTool":
        return os.path.join(slicer_root, "share", slicer_version_specific_path_part, "CommandLineTool", f"{module_name}.py")
    elif module_type == "qt-scripted-modules":
        return os.path.join(slicer_root, "lib", slicer_version_specific_path_part, "qt-scripted-modules", f"{module_name}.py")
    else:
        raise ValueError(f"Unknown SlicerSALT module type: {module_type}")

SPHARM_PDM_PYTHON_SCRIPT_PATH = get_slicersalt_module_path("SPHARM-PDM", "SlicerSALT-5.3", "CommandLineTool")
MFSDA_RUN_PATH = get_slicersalt_module_path("MFSDA_run", "SlicerSALT-5.3", "qt-scripted-modules") 
MFSDA_CREATE_SHAPES_PATH = get_slicersalt_module_path("MFSDA_createShapes", "SlicerSALT-5.3", "qt-scripted-modules")

print(MFSDA_RUN_PATH)

# ===============================================================================
# DIRECTORY STRUCTURE
# ===============================================================================
def build_path(*parts: str) -> str:
    return os.path.join(BASE_DIR, *parts)

T1_IMAGES_DIR_ROOT_FOR_SPHARM_INPUT = "/home/jeremy/Desktop/hippo_morph/Input_T1s/" 

# Segmentation outputs (where Hippodeep initially places files or where they are moved)
SEG_OUTPUT_ROOT_DIR = build_path("data", "segmentation_output")
SEG_LEFT_MASKS_DIR = os.path.join(SEG_OUTPUT_ROOT_DIR, "masks_L") # Referenced by preprocessing.py
SEG_RIGHT_MASKS_DIR = os.path.join(SEG_OUTPUT_ROOT_DIR, "masks_R") # Referenced by preprocessing.py
SEG_CEREBRUM_MASKS_DIR = os.path.join(SEG_OUTPUT_ROOT_DIR, "masks_cerebrum") # Referenced by preprocessing.py
SEG_BRAIN_MASKS_DIR = os.path.join(SEG_OUTPUT_ROOT_DIR, "masks_brain") # Referenced by preprocessing.py
SEG_VOLUME_REPORTS_DIR = os.path.join(SEG_OUTPUT_ROOT_DIR, "volume_reports") # Referenced by preprocessing.py

# SPHARM Processing and Output Directories
SPHARM_COMPUTATION_BASE_DIR = build_path("output", "spharm_computation_base") # DEFINITION ADDED HERE
SPHARM_PIPELINE_OUTPUT_ROOT_DIR = build_path("output", "spharm_processing")
SPHARM_TOOL_LEFT_OUTPUT_DIR = os.path.join(SPHARM_PIPELINE_OUTPUT_ROOT_DIR, "tool_output_L") 
SPHARM_TOOL_RIGHT_OUTPUT_DIR = os.path.join(SPHARM_PIPELINE_OUTPUT_ROOT_DIR, "tool_output_R")
SPHARM_EXTRACTED_MODELS_ROOT_DIR = build_path("output", "spharm_extracted_models_adapted") 
SPHARM_EXTRACTED_MODELS_LEFT_DIR = os.path.join(SPHARM_EXTRACTED_MODELS_ROOT_DIR, "left")
SPHARM_EXTRACTED_MODELS_RIGHT_DIR = os.path.join(SPHARM_EXTRACTED_MODELS_ROOT_DIR, "right")

# Other Processing Directories
BINARIZED_MASKS_ROOT_DIR = build_path("data", "binarized_masks")
BINARIZED_LEFT_MASKS_DIR = os.path.join(BINARIZED_MASKS_ROOT_DIR, "masks_L")
BINARIZED_RIGHT_MASKS_DIR = os.path.join(BINARIZED_MASKS_ROOT_DIR, "masks_R")
REGISTERED_MASKS_ROOT_DIR = build_path("data", "registered_masks")
REGISTERED_LEFT_MASKS_DIR = os.path.join(REGISTERED_MASKS_ROOT_DIR, "masks_L")
REGISTERED_RIGHT_MASKS_DIR = os.path.join(REGISTERED_MASKS_ROOT_DIR, "masks_R")
STATS_ROOT_DIR = build_path("output", "statistics")
STATS_STATIC_LEFT_DIR = os.path.join(STATS_ROOT_DIR, "static", "left") 
STATS_STATIC_RIGHT_DIR = os.path.join(STATS_ROOT_DIR, "static", "right") 
STATS_LONGITUDINAL_LEFT_DIR = os.path.join(STATS_ROOT_DIR, "longitudinal", "left") 
STATS_LONGITUDINAL_RIGHT_DIR = os.path.join(STATS_ROOT_DIR, "longitudinal", "right") 
CSV_REPORTS_DIR = build_path("output", "csv_reports_adapted")
TEMPLATES_DIR = build_path("templates") 

# ===============================================================================
# TEMPLATES AND REFERENCES 
# Define the *actual filenames* of your templates here.
# ===============================================================================
ACTUAL_LEFT_SPHERE_TEMPLATE_FILENAME = "MCALT_T1_mask_L_bin_pp_surf_SPHARM.vtk"
ACTUAL_RIGHT_SPHERE_TEMPLATE_FILENAME = "MCALT_T1_mask_R_bin_pp_surf_SPHARM.vtk"
ACTUAL_LEFT_FLIP_TEMPLATE_FILENAME = "MCALT_T1_mask_L_bin_pp_surf_SPHARM.coef"
ACTUAL_RIGHT_FLIP_TEMPLATE_FILENAME = "MCALT_T1_mask_R_bin_pp_surf_SPHARM.coef"
LEFT_REFERENCE_IMAGE_FLIRT = os.path.join(TEMPLATES_DIR, "MCALT_T1_mask_L_bin.nii.gz") 
RIGHT_REFERENCE_IMAGE_FLIRT = os.path.join(TEMPLATES_DIR, "MCALT_T1_mask_R_bin.nii.gz") 

# ===============================================================================
# PROCESSING PARAMETERS
# ===============================================================================
FLIRT_OPTIONS = "-searchcost labeldiff -cost labeldiff -dof 6 -interp nearestneighbour"

SPHARM_PARAMS = {
    "rescale": "True",
    "space": "0.5,0.5,0.5",
    "label": "0",  
    "gauss": "False",
    "var": "10,10,10",
    "iter": "1000", 
    "subdivLevel": "10",
    "spharmDegree": "15",
    "medialMesh": "True", 
    "phiIteration": "100",
    "thetaIteration": "100",
    "regParaTemplateFileOn": "True", 
    "flipTemplateOn": "True",      
    "flip": "0" 
}

SPHARM_COMPLETION_PHRASE = "ParaToSPHARMMesh completed without errors" 

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
    "spharm_computation_base": SPHARM_COMPUTATION_BASE_DIR, # Now defined above
    "spharm_extracted_models_root": SPHARM_EXTRACTED_MODELS_ROOT_DIR,
    "spharm_extracted_left": SPHARM_EXTRACTED_MODELS_LEFT_DIR,
    "spharm_extracted_right": SPHARM_EXTRACTED_MODELS_RIGHT_DIR,
    "csv_reports": CSV_REPORTS_DIR,
    "segmentation_output_root": SEG_OUTPUT_ROOT_DIR, 
    "binarized_masks_root": BINARIZED_MASKS_ROOT_DIR,
    "registered_masks_root": REGISTERED_MASKS_ROOT_DIR,
    "statistics_output_root": STATS_ROOT_DIR,
}

def create_directories(non_interactive_flag: bool = NON_INTERACTIVE):
    logger.info("Initializing base project directories...")
    created_count = 0
    skipped_count = 0
    if not os.path.exists(BASE_DIR):
        logger.error(f"CRITICAL: Base directory {BASE_DIR} does not exist. Cannot proceed.")
        return False 
    if not os.path.exists(T1_IMAGES_DIR_ROOT_FOR_SPHARM_INPUT): 
        logger.warning(f"Input T1 directory for SPHARM {T1_IMAGES_DIR_ROOT_FOR_SPHARM_INPUT} does not exist.")
    if not os.path.exists(TEMPLATES_DIR):
        logger.warning(f"Templates directory {TEMPLATES_DIR} does not exist.")

    def _create_dirs_recursive(item):
        nonlocal created_count, skipped_count
        if isinstance(item, dict):
            for k, v_list in item.items(): 
                if isinstance(v_list, dict): 
                    for sub_k, sub_v in v_list.items():
                         _create_dirs_recursive(sub_v) 
                elif isinstance(v_list, str): 
                     _create_dirs_recursive(v_list)
        elif isinstance(item, str): 
            if not os.path.exists(item):
                try:
                    os.makedirs(item, exist_ok=True)
                    logger.info(f"Created directory: {item}")
                    created_count += 1
                except OSError as e:
                    logger.error(f"Failed to create directory {item}: {e}")
            else:
                skipped_count += 1
    
    _create_dirs_recursive(ALL_DIRECTORIES_TO_CREATE)
    
    summary_message = f"Base directory initialization complete. Created: {created_count}, Already existed/Skipped: {skipped_count}."
    logger.info(summary_message)
    return True

def check_environment():
    logger.info("Performing environment check...")
    issues = []
    # Directly use global variables defined in this module
    if not os.path.exists(SLICERSALT_PATH) or not os.access(SLICERSALT_PATH, os.X_OK):
        issues.append(f"SlicerSALT executable not found or not executable at {SLICERSALT_PATH}")
    if not os.path.exists(SPHARM_PDM_PYTHON_SCRIPT_PATH): 
        issues.append(f"SPHARM_PDM.py script not found at {SPHARM_PDM_PYTHON_SCRIPT_PATH}")
    
    if hasattr(sys.modules[__name__], 'HIPPODEEP_PATH'):
        if HIPPODEEP_PATH and (not os.path.exists(HIPPODEEP_PATH) or not os.access(HIPPODEEP_PATH, os.X_OK)):
            issues.append(f"Hippodeep script not found or not executable at {HIPPODEEP_PATH}")
    else:
        logger.debug("HIPPODEEP_PATH not defined in config, skipping its check in environment setup.")

    try: 
        flirt_check = subprocess.run(["flirt", "-version"], capture_output=True, text=True, check=False)
        if flirt_check.returncode != 0: issues.append("FSL FLIRT (flirt -version) command failed or not found in PATH.")
    except FileNotFoundError: issues.append("FSL FLIRT (flirt) command not found.")
    try:
        fslmaths_check = subprocess.run(["fslmaths", "-version"], capture_output=True, text=True, check=False)
        if fslmaths_check.returncode != 0 and not fslmaths_check.stdout and not fslmaths_check.stderr : 
             issues.append("FSL MATHS (fslmaths -version) command failed or not found in PATH.")
    except FileNotFoundError: issues.append("FSL MATHS (fslmaths) command not found.")

    if SPHARM_PARAMS.get("regParaTemplateFileOn", "False").lower() == "true":
        left_reg_template = os.path.join(TEMPLATES_DIR, ACTUAL_LEFT_SPHERE_TEMPLATE_FILENAME)
        right_reg_template = os.path.join(TEMPLATES_DIR, ACTUAL_RIGHT_SPHERE_TEMPLATE_FILENAME)
        if not os.path.exists(left_reg_template):
            issues.append(f"Left sphere template missing: {left_reg_template}")
        if not os.path.exists(right_reg_template):
            issues.append(f"Right sphere template missing: {right_reg_template}")
            
    if SPHARM_PARAMS.get("flipTemplateOn", "False").lower() == "true":
        left_flip_template = os.path.join(TEMPLATES_DIR, ACTUAL_LEFT_FLIP_TEMPLATE_FILENAME)
        right_flip_template = os.path.join(TEMPLATES_DIR, ACTUAL_RIGHT_FLIP_TEMPLATE_FILENAME)
        if not os.path.exists(left_flip_template):
            issues.append(f"Left flip template missing: {left_flip_template}")
        if not os.path.exists(right_flip_template):
            issues.append(f"Right flip template missing: {right_flip_template}")
        
    if issues:
        logger.warning("--- ENVIRONMENT CHECK FOUND ISSUES ---")
        for issue in issues: logger.warning(f"  - {issue}")
        return False
    else:
        logger.info("Environment check passed. Required tools and template files appear to be available.")
        return True

def get_paths_by_side(side_label: str, component_type: str) -> str:
    if side_label not in ["left", "right"]:
        raise ValueError(f"Invalid side_label: {side_label}. Must be 'left' or 'right'.")
    
    path_map = {
        "left": {
            "seg_mask_input": SEG_LEFT_MASKS_DIR, 
            "bin_mask_input": BINARIZED_LEFT_MASKS_DIR, 
            "reg_mask_input": REGISTERED_LEFT_MASKS_DIR, 
            "spharm_tool_output": SPHARM_TOOL_LEFT_OUTPUT_DIR, 
            "spharm_extracted_models": SPHARM_EXTRACTED_MODELS_LEFT_DIR, 
            "sphere_template": os.path.join(TEMPLATES_DIR, ACTUAL_LEFT_SPHERE_TEMPLATE_FILENAME), 
            "flip_template": os.path.join(TEMPLATES_DIR, ACTUAL_LEFT_FLIP_TEMPLATE_FILENAME),   
            "reference_image_flirt": LEFT_REFERENCE_IMAGE_FLIRT, 
            "static_csv": STATIC_LEFT_CSV, 
            "longit_m2md_csv": LONGITUDINAL_LEFT_M2MD_CSV,
            "longit_stats_csv": LONGITUDINAL_STATS_LEFT_CSV,
            "stats_static_output": STATS_STATIC_LEFT_DIR, 
            "stats_longit_output": STATS_LONGITUDINAL_LEFT_DIR, 
            "m2md_output": os.path.join(build_path("output", "m2md"), "left"), 
            "delta_models_output": os.path.join(build_path("output", "delta_models"), "left") 
        },
        "right": {
            "seg_mask_input": SEG_RIGHT_MASKS_DIR,
            "bin_mask_input": BINARIZED_RIGHT_MASKS_DIR,
            "reg_mask_input": REGISTERED_RIGHT_MASKS_DIR, 
            "spharm_tool_output": SPHARM_TOOL_RIGHT_OUTPUT_DIR, 
            "spharm_extracted_models": SPHARM_EXTRACTED_MODELS_RIGHT_DIR,
            "sphere_template": os.path.join(TEMPLATES_DIR, ACTUAL_RIGHT_SPHERE_TEMPLATE_FILENAME), 
            "flip_template": os.path.join(TEMPLATES_DIR, ACTUAL_RIGHT_FLIP_TEMPLATE_FILENAME),   
            "reference_image_flirt": RIGHT_REFERENCE_IMAGE_FLIRT,
            "static_csv": STATIC_RIGHT_CSV,
            "longit_m2md_csv": LONGITUDINAL_RIGHT_M2MD_CSV,
            "longit_stats_csv": LONGITUDINAL_STATS_RIGHT_CSV,
            "stats_static_output": STATS_STATIC_RIGHT_DIR, 
            "stats_longit_output": STATS_LONGITUDINAL_RIGHT_DIR, 
            "m2md_output": os.path.join(build_path("output", "m2md"), "right"),
            "delta_models_output": os.path.join(build_path("output", "delta_models"), "right")
        }
    }
    try:
        return path_map[side_label][component_type]
    except KeyError:
        valid_keys = list(path_map['left'].keys()) 
        raise ValueError(f"Invalid component_type '{component_type}' for side '{side_label}'. Valid types are: {sorted(list(set(valid_keys)))}")