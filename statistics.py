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

import config
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
    
    # Get paths based on side
    csv_path = config.get_paths_by_side(side, "static_csv")
    output_dir = config.get_paths_by_side(side, "stats_static")
    shape_template = config.get_paths_by_side(side, "shape_template")
    sphere_template = config.get_paths_by_side(side, "sphere_template")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if the CSV file exists
    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found at {csv_path}")
        return False
    
    logger.info(f"Running statistical analysis using CSV: {csv_path}")
    
    # Check if CSV contains the necessary data
    try:
        df = pd.read_csv(csv_path)
        if len(df.columns) < 2:
            logger.error(f"CSV file {csv_path} must have at least 2 columns (subjects and at least one covariate)")
            return False
            
        # Check if subject paths exist
        subject_column = df.columns[0]
        missing_subjects = [path for path in df[subject_column] if not os.path.exists(path)]
        if missing_subjects:
            logger.warning(f"Found {len(missing_subjects)} missing subject files")
            logger.warning(f"First missing file: {missing_subjects[0]}")
    except Exception as e:
        logger.error(f"Error reading CSV file {csv_path}: {str(e)}")
        return False
    
    # Run MFSDA analysis
    success = run_mfsda(
        csv_path=csv_path,
        output_dir=output_dir,
        shape_template=shape_template,
        sphere_template=sphere_template,
        mfsda_run_path=config.MFSDA_RUN_PATH,
        mfsda_createshapes_path=config.MFSDA_CREATE_SHAPES_PATH,
        slicersalt_path=config.SLICERSALT_PATH,
        max_workers=config.MAX_WORKERS
    )
    
    if success:
        logger.info(f"Statistical analysis completed. Results saved to {output_dir}")
    else:
        logger.error(f"Statistical analysis failed. Check logs for details.")
    
    return success

def run_longitudinal_analysis(side: str = "left") -> bool:
    """
    Run longitudinal statistical analysis on delta models.
    
    Parameters:
    -----------
    side : str
        Hemisphere to process ("left" or "right").
        
    Returns:
    --------
    bool
        True if analysis was successful, False otherwise
    """
    print_section_header(f"LONGITUDINAL STATISTICAL ANALYSIS: {side.upper()} HEMISPHERE")
    
    # Get paths based on side
    csv_path = config.get_paths_by_side(side, "longit_stats_csv")
    output_dir = config.get_paths_by_side(side, "stats_longit")
    shape_template = config.get_paths_by_side(side, "shape_template")
    sphere_template = config.get_paths_by_side(side, "sphere_template")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if the CSV file exists
    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found at {csv_path}")
        return False
    
    logger.info(f"Running longitudinal statistical analysis using CSV: {csv_path}")
    
    # Check if CSV contains the necessary data
    try:
        df = pd.read_csv(csv_path)
        if len(df.columns) < 2:
            logger.error(f"CSV file {csv_path} must have at least 2 columns (delta models and at least one covariate)")
            return False
            
        # Check if subject paths exist
        subject_column = df.columns[0]
        missing_subjects = [path for path in df[subject_column] if not os.path.exists(path)]
        if missing_subjects:
            logger.warning(f"Found {len(missing_subjects)} missing subject files")
            logger.warning(f"First missing file: {missing_subjects[0]}")
    except Exception as e:
        logger.error(f"Error reading CSV file {csv_path}: {str(e)}")
        return False
    
    # Run MFSDA analysis
    success = run_mfsda(
        csv_path=csv_path,
        output_dir=output_dir,
        shape_template=shape_template,
        sphere_template=sphere_template,
        mfsda_run_path=config.MFSDA_RUN_PATH,
        mfsda_createshapes_path=config.MFSDA_CREATE_SHAPES_PATH,
        slicersalt_path=config.SLICERSALT_PATH,
        max_workers=config.MAX_WORKERS
    )
    
    if success:
        logger.info(f"Longitudinal statistical analysis completed. Results saved to {output_dir}")
    else:
        logger.error(f"Longitudinal statistical analysis failed. Check logs for details.")
    
    return success

def run_mfsda(csv_path: str, 
             output_dir: str, 
             shape_template: str, 
             sphere_template: str, 
             mfsda_run_path: str, 
             mfsda_createshapes_path: str, 
             slicersalt_path: str, 
             max_workers: int) -> bool:
    """
    Run MFSDA (Multivariate Functional Shape Data Analysis) on shape models.
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file containing subject data and covariates.
    output_dir : str
        Directory to store analysis results.
    shape_template : str
        Path to the shape template file.
    sphere_template : str
        Path to the sphere template file.
    mfsda_run_path : str
        Path to the MFSDA_run.py module.
    mfsda_createshapes_path : str
        Path to the MFSDA_createShapes.py module.
    slicersalt_path : str
        Path to the SlicerSALT executable.
    max_workers : int
        Maximum number of concurrent processes.
        
    Returns:
    --------
    bool
        True if MFSDA was successful, False otherwise
    """
    # Check if necessary files exist
    for filepath in [csv_path, shape_template, sphere_template, slicersalt_path]:
        if not os.path.exists(filepath):
            logger.error(f"Required file not found: {filepath}")
            return False
    
    # Create a temporary directory to store individual CSV files
    temp_dir = tempfile.mkdtemp(prefix="mfsda_")
    logger.info(f"Created temporary directory for analysis: {temp_dir}")
    
    try:
        # Read the main CSV file
        df = pd.read_csv(csv_path)
        
        # Get the first column name (should contain subject paths)
        subject_column = df.columns[0]
        
        # Get covariate names (all columns except the first)
        covariates = df.columns[1:].tolist()
        if not covariates:
            logger.error(f"No covariates found in {csv_path}")
            return False
            
        covariates_str = ' '.join(covariates)
        
        logger.info(f"Found subject column: {subject_column}")
        logger.info(f"Found covariates: {covariates_str}")
        
        # Create the command for MFSDA_run.py
        mfsda_run_command = [
            slicersalt_path,
            "--no-main-window",
            "--python-script",
            mfsda_run_path,
            "--shapeData", csv_path,
            "--coordData", sphere_template,
            "--outputDir", output_dir
        ]
        
        # Run MFSDA_run.py
        logger.info("Running MFSDA_run.py...")
        result = run_command(
            mfsda_run_command,
            description="Running MFSDA statistical analysis",
            check=False
        )
        
        if result.returncode != 0:
            logger.error(f"MFSDA_run.py failed with return code {result.returncode}")
            logger.error(f"Error output: {result.stderr}")
            return False
            
        # Check if output files were created
        pvalues_file = os.path.join(output_dir, 'pvalues.json')
        efit_file = os.path.join(output_dir, 'efit.json')
        
        if not os.path.exists(pvalues_file) or not os.path.exists(efit_file):
            logger.error(f"MFSDA_run.py did not create expected output files")
            return False
            
        # Display basic statistics from the pvalues.json file
        try:
            with open(pvalues_file, 'r') as f:
                pvalues_data = json.load(f)
                min_pval = min(pvalues_data) if isinstance(pvalues_data, list) else "unknown"
                logger.info(f"Minimum p-value: {min_pval}")
        except Exception as e:
            logger.warning(f"Could not read p-values file: {str(e)}")
        
        # Prepare the arguments for MFSDA_createShapes.py
        output_vtk = os.path.join(output_dir, 'output.vtk')
        
        # Create the command for MFSDA_createShapes.py
        mfsda_create_shapes_command = [
            slicersalt_path,
            "--no-main-window",
            "--python-script",
            mfsda_createshapes_path,
            "--shape", shape_template,
            "--pvalues", pvalues_file,
            "--efit", efit_file,
            "--covariates", covariates_str,
            "--output", output_vtk
        ]
        
        # Run MFSDA_createShapes.py
        logger.info("Running MFSDA_createShapes.py...")
        result = run_command(
            mfsda_create_shapes_command,
            description="Creating shape visualization for statistical analysis",
            check=False
        )
        
        if result.returncode != 0:
            logger.error(f"MFSDA_createShapes.py failed with return code {result.returncode}")
            logger.error(f"Error output: {result.stderr}")
            return False
        
        logger.info("MFSDA analysis completed successfully.")
        return True
    
    except Exception as e:
        logger.error(f"Error during MFSDA analysis: {str(e)}")
        return False
    
    finally:
        # Clean up temporary directory
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

def prepare_longitudinal_data(side: str = "left") -> bool:
    """
    Prepare data for longitudinal analysis by generating delta models.
    
    Parameters:
    -----------
    side : str
        Hemisphere to process ("left" or "right").
        
    Returns:
    --------
    bool
        True if preparation was successful, False otherwise
    """
    print_section_header(f"LONGITUDINAL DATA PREPARATION: {side.upper()} HEMISPHERE")
    
    # Get paths based on side
    csv_path = config.get_paths_by_side(side, "longit_csv")
    output_dir = config.get_paths_by_side(side, "m2md")
    delta_models_dir = config.get_paths_by_side(side, "delta")
    
    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(delta_models_dir, exist_ok=True)
    
    # Check if the CSV file exists
    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found at {csv_path}")
        return False
    
    # Check if CSV has the required columns
    try:
        df = pd.read_csv(csv_path)
        required_columns = ['Timepoint 1', 'Timepoint 2', 'Output']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"CSV file is missing required columns: {', '.join(missing_columns)}")
            logger.error(f"CSV must have columns: 'Timepoint 1', 'Timepoint 2', 'Output'")
            return False
            
        # Check if files exist
        missing_tp1 = [path for path in df['Timepoint 1'] if not os.path.exists(path)]
        missing_tp2 = [path for path in df['Timepoint 2'] if not os.path.exists(path)]
        
        if missing_tp1 or missing_tp2:
            logger.warning(f"Found {len(missing_tp1)} missing Timepoint 1 files and {len(missing_tp2)} missing Timepoint 2 files")
            if missing_tp1:
                logger.warning(f"First missing Timepoint 1 file: {missing_tp1[0]}")
            if missing_tp2:
                logger.warning(f"First missing Timepoint 2 file: {missing_tp2[0]}")
                
    except Exception as e:
        logger.error(f"Error reading CSV file {csv_path}: {str(e)}")
        return False
    
    # Run model-to-model distance (M2MD)
    m2md_success = run_m2md(csv_path, output_dir, config.SLICERSALT_PATH)
    
    if not m2md_success:
        return False
    
    # Generate delta models
    m2md_results_csv = os.path.join(output_dir, "M2MD_results.csv")
    if not os.path.exists(m2md_results_csv):
        logger.error(f"M2MD results CSV file not found at {m2md_results_csv}")
        return False
        
    delta_success = generate_delta_models(m2md_results_csv, delta_models_dir)
    
    if not delta_success:
        return False
        
    logger.info(f"Longitudinal data preparation completed. Delta models saved to {delta_models_dir}")
    return True

def run_m2md(csv_path: str, output_dir: str, slicersalt_path: str) -> bool:
    """
    Run Model-to-Model Distance (M2MD) analysis.
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file containing timepoint data.
        The CSV should have columns: 'Timepoint 1', 'Timepoint 2', 'Output'.
    output_dir : str
        Directory to store M2MD results.
    slicersalt_path : str
        Path to the SlicerSALT executable.
        
    Returns:
    --------
    bool
        True if M2MD was successful, False otherwise
    """
    logger.info("Running Model-to-Model Distance analysis")
    
    # Create a temporary script file to run M2MD within SlicerSALT
    fd, script_path = tempfile.mkstemp(suffix='.py', prefix='m2md_script_')
    os.close(fd)
    
    try:
        # Write the M2MD script content
        with open(script_path, 'w') as f:
            f.write(f"""
# M2MD Script for SlicerSALT
import os
import numpy as np
import pandas as pd
import slicer

# Input and output paths
csv_file = "{csv_path}"
output_dir = "{output_dir}"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load the CSV file with headers
try:
    data = pd.read_csv(csv_file)
    
    # Check for required columns
    required_columns = ['Timepoint 1', 'Timepoint 2', 'Output']
    for col in required_columns:
        if col not in data.columns:
            print(f"Error: Required column '{col}' not found in CSV")
            exit(1)
            
    # Get data from columns
    tp1_subjects = data['Timepoint 1']
    tp2_subjects = data['Timepoint 2']
    output_subjects = data['Output']
    
    # Create results dataframe
    results_array = pd.DataFrame(columns=[
        'Timepoint 1 Model', 
        'Timepoint 2 Model', 
        'Output M2MD Model', 
        'MagNormVectors File'
    ])
    
    # Process each pair of models
    for i, (tp1_subj, tp2_subj, output_subj) in enumerate(zip(tp1_subjects, tp2_subjects, output_subjects)):
        try:
            print(f"\\nProcessing model pair {i+1}/{len(tp1_subjects)}: {output_subj}")
            
            # Load timepoint models in Slicer
            model1 = slicer.util.loadModel(tp1_subj)
            model2 = slicer.util.loadModel(tp2_subj)
            
            if not model1 or not model2:
                print(f"Skipped {output_subj} due to failure in loading models.")
                continue
                
            # Create output node
            output_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", output_subj)
            
            # Set parameters for Model to Model Distance module
            params = {{
                'vtkFile1': model1.GetID(), 
                'vtkFile2': model2.GetID(), 
                'vtkOutput': output_node.GetID(), 
                'distanceType': 'corresponding_point_to_point', 
                'targetInFields': False
            }}
            
            # Execute Model to Model Distance module
            slicer.cli.runSync(slicer.modules.modeltomodeldistance, None, parameters=params)
            
            # Get model node for data extraction
            model_node = slicer.util.getNode(output_node.GetID())
            
            # Extract MagNormVector data
            MagNormVectors = slicer.util.arrayFromModelPointData(model_node, 'MagNormVector')
            
            # Define output paths
            output_model_path = os.path.join(output_dir, output_subj + ".vtk")
            magNormVectors_path = os.path.join(output_dir, output_subj + "_MagNormVectors.csv")
            
            # Save model and vector data
            slicer.util.saveNode(output_node, output_model_path)
            print(f"Output model created: {output_model_path}")
            
            np.savetxt(magNormVectors_path, MagNormVectors, delimiter=",", comments='')
            print(f"MagNormVector file created: {magNormVectors_path}")
            
            # Add result to dataframe
            new_row = {{
                'Timepoint 1 Model': tp1_subj,
                'Timepoint 2 Model': tp2_subj,
                'Output M2MD Model': output_model_path,
                'MagNormVectors File': magNormVectors_path
            }}
            results_array = pd.concat([results_array, pd.DataFrame([new_row])], ignore_index=True)
            
        except Exception as e:
            print(f"Error processing {output_subj}: {str(e)}")
            continue
    
    # Save results as CSV
    results_csv_path = os.path.join(output_dir, "M2MD_results.csv")
    results_array.to_csv(results_csv_path, index=False)
    print(f"\\nResults saved to: {results_csv_path}")
    print(f"\\nProcessed {len(results_array)} model pairs successfully")
    
except Exception as e:
    print(f"Error in M2MD processing: {str(e)}")
    exit(1)
""")
        
        # Execute the M2MD script through SlicerSALT
        slicer_command = [
            slicersalt_path,
            "--no-main-window",
            "--python-script",
            script_path
        ]
        
        result = run_command(
            slicer_command,
            description="Running Model-to-Model Distance analysis (this may take a while)",
            check=False
        )
        
        if result.returncode != 0:
            logger.error(f"M2MD analysis failed with return code {result.returncode}")
            logger.error(f"Error output: {result.stderr}")
            return False
            
        # Check if output file was created
        m2md_results_csv = os.path.join(output_dir, "M2MD_results.csv")
        if not os.path.exists(m2md_results_csv):
            logger.error("M2MD analysis did not create the expected results file")
            return False
            
        # Count number of processed models
        try:
            results_df = pd.read_csv(m2md_results_csv)
            logger.info(f"M2MD analysis completed with {len(results_df)} processed model pairs")
        except Exception as e:
            logger.warning(f"Could not read M2MD results file: {str(e)}")
        
        logger.info(f"Model-to-Model Distance analysis completed. Results saved to {output_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Error running Model-to-Model Distance analysis: {str(e)}")
        return False
        
    finally:
        # Clean up the temporary script
        try:
            os.remove(script_path)
        except:
            pass

def generate_delta_models(m2md_results_csv: str, output_dir: str) -> bool:
    """
    Generate delta models from M2MD results.
    
    Parameters:
    -----------
    m2md_results_csv : str
        Path to the M2MD_results.csv file.
    output_dir : str
        Directory to store delta models.
        
    Returns:
    --------
    bool
        True if delta model generation was successful, False otherwise
    """
    logger.info("Generating delta models from M2MD results")
    
    # Check if the results CSV file exists
    if not os.path.exists(m2md_results_csv):
        logger.error(f"M2MD results CSV file not found at {m2md_results_csv}")
        return False
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a temporary script file for processing
    fd, script_path = tempfile.mkstemp(suffix='.py', prefix='delta_models_script_')
    os.close(fd)
    
    try:
        # Write the script content
        with open(script_path, 'w') as f:
            f.write(f"""
import os
import vtk
import numpy as np
import pandas as pd
import sys

# Define input and output paths
csv_file = "{m2md_results_csv}"
output_dir = "{output_dir}"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Define helper functions
def read_vtk(filename):
    '''Read points from a VTK file.'''
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    
    polydata = reader.GetOutput()
    points = polydata.GetPoints()
    numPoints = points.GetNumberOfPoints()
    
    pointCoordinates = np.zeros((numPoints, 3))
    for i in range(numPoints):
        pointCoordinates[i, :] = points.GetPoint(i)
    
    return pointCoordinates, polydata

def write_vtk_from_csv(filename, csv_path, base_vtk_path):
    '''Create a new VTK file with points from CSV and polygons from base VTK.'''
    # Read vector data from CSV
    try:
        vectors = np.loadtxt(csv_path, delimiter=",")
    except Exception as e:
        print(f"Error reading CSV file {{csv_path}}: {{str(e)}}")
        return False
    
    # Read polygon data from base VTK
    try:
        _, base_polydata = read_vtk(base_vtk_path)
        polygons = base_polydata.GetPolys()
    except Exception as e:
        print(f"Error reading base VTK file {{base_vtk_path}}: {{str(e)}}")
        return False
    
    # Create new polydata with vectors as points
    polydata = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    
    for i in range(vectors.shape[0]):
        points.InsertNextPoint(vectors[i, 0], vectors[i, 1], vectors[i, 2])
    
    polydata.SetPoints(points)
    polydata.SetPolys(polygons)
    
    # Write to file
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(polydata)
    writer.SetFileVersion(42)  # VTK file format version
    writer.Write()
    
    return True

# Main script execution
try:
    # Load the M2MD results CSV
    print(f"Reading M2MD results from {{csv_file}}")
    if not os.path.exists(csv_file):
        print(f"Error: M2MD results file not found: {{csv_file}}")
        sys.exit(1)
        
    m2md_data = pd.read_csv(csv_file)
    
    # Check for required columns
    required_columns = ['Timepoint 1 Model', 'MagNormVectors File']
    missing_columns = [col for col in required_columns if col not in m2md_data.columns]
    if missing_columns:
        print(f"Error: M2MD results file is missing required columns: {{', '.join(missing_columns)}}")
        sys.exit(1)
    
    # Get column data
    base_VTKs = m2md_data['Timepoint 1 Model']
    mnv_files = m2md_data['MagNormVectors File']
    
    # Process each model pair
    success_count = 0
    failed_count = 0
    
    for i, (base_VTK, mnv_file) in enumerate(zip(base_VTKs, mnv_files)):
        print(f"\\nProcessing pair {{i+1}}/{{len(base_VTKs)}}")
        print(f"Base VTK: {{os.path.basename(base_VTK)}}")
        print(f"Vector file: {{os.path.basename(mnv_file)}}")
        
        # Check if files exist
        if not os.path.exists(base_VTK):
            print(f"Error: Base VTK file not found: {{base_VTK}}")
            failed_count += 1
            continue
            
        if not os.path.exists(mnv_file):
            print(f"Error: MagNormVectors file not found: {{mnv_file}}")
            failed_count += 1
            continue
        
        # Create output filename
        vtk_basename = os.path.splitext(os.path.basename(mnv_file))[0]
        vtk_basename = vtk_basename.replace("_MagNormVectors", "")
        delta_vtk_filename = f"deltas_{{vtk_basename}}.vtk"
        delta_vtk_path = os.path.join(output_dir, delta_vtk_filename)
        
        # Create delta VTK model
        if write_vtk_from_csv(delta_vtk_path, mnv_file, base_VTK):
            print(f"Created delta model: {{delta_vtk_path}}")
            success_count += 1
        else:
            print(f"Failed to create delta model for {{vtk_basename}}")
            failed_count += 1
    
    print(f"\\nDelta model generation complete")
    print(f"Successfully created {{success_count}} delta models")
    if failed_count > 0:
        print(f"Failed to create {{failed_count}} delta models")
    print(f"Output directory: {{output_dir}}")
    
except Exception as e:
    print(f"Error generating delta models: {{str(e)}}")
    sys.exit(1)
""")
        
        # Execute the delta models generation script
        result = run_command(
            ["python", script_path],
            description="Generating delta models",
            check=False
        )
        
        if result.returncode != 0:
            logger.error(f"Delta model generation failed with return code {result.returncode}")
            logger.error(f"Error output: {result.stderr}")
            return False
        
        # Check if any delta models were created
        delta_models = [f for f in os.listdir(output_dir) if f.startswith("deltas_") and f.endswith(".vtk")]
        if not delta_models:
            logger.warning("No delta models were created")
            return False
            
        logger.info(f"Created {len(delta_models)} delta models in {output_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Error generating delta models: {str(e)}")
        return False
        
    finally:
        # Clean up the temporary script
        try:
            os.remove(script_path)
        except:
            pass
