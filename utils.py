import subprocess
import sys
import os
import re 
import ast
import json
import tempfile
import numpy as np 
def merge_close_genes(original_positions, dithered_positions, threshold=10):
    original_positions = [np.array(pos) for pos in original_positions]
    dithered_positions = [np.array(pos) for pos in dithered_positions]
    changed = True
    while changed:
        changed = False
        n = len(original_positions)
        i = 0
        while i < n:
            j = i + 1
            merged = False
            while j < n:
                dist = np.linalg.norm(original_positions[i] - original_positions[j])
                if dist < threshold:
                    # merge: average original and dithered
                    new_orig = (original_positions[i] + original_positions[j]) / 2
                    new_dith = (dithered_positions[i] + dithered_positions[j]) / 2
                    # new_dith = random.choice([dithered_positions[i],dithered_positions[j]])

                    # replace i with merged, remove j
                    original_positions[i] = new_orig
                    dithered_positions[i] = new_dith
                    del original_positions[j]
                    del dithered_positions[j]

                    n -= 1
                    changed = True
                    merged = True
                    break
                j += 1
            if not merged:
                i += 1
    original_positions = [list(pos) for pos in original_positions]
    dithered_positions = [list(pos) for pos in dithered_positions]
    return original_positions, dithered_positions

def generate_face_mesh(
    blender_executable_path: str,
    analyzer_script_path: str,
    blend_file_to_open: str,
    genes: list,
    output_file: str,
    timeout_seconds: int = 300 # 5 minutes
    ):
    """
    Runs the Blender puddle analysis script headlessly.

    Args:
        blender_executable_path (str): Full path to the Blender executable.
        analyzer_script_path (str): Full path to the 'puddle_analyzer_headless.py'.
        blend_file_to_open (str): Path to the .blend file (e.g., 'face_blend.blend').
        object_name_in_blend (str): Name of the object to analyze.
        voxel_s (float): Voxel size.
        create_debug (bool): If True, tells Blender script to create debug meshes.
        verbose_blender_output (bool): If True, tells Blender script to print verbose logs.
        timeout_seconds (int): Timeout for the Blender process.

    Returns:
        dict or None: Parsed results (nx, ny, nz, solid_volume, trapped_water_vol, cupped_water_vol)
                      or None if an error occurred or output couldn't be parsed.
    """

    if not os.path.exists(blender_executable_path):
        print(f"Error: Blender executable not found at '{blender_executable_path}'")
        return None
    if not os.path.exists(analyzer_script_path):
        print(f"Error: Blender analyzer script not found at '{analyzer_script_path}'")
        return None
    if not os.path.exists(blend_file_to_open):
        print(f"Error: Specified .blend file not found at '{blend_file_to_open}'")
        return None
    # if not os.path.exists(dither_config):
    #     print(f"Error: Specified config file not found at '{dither_config}'")
    #     return None
    
    # with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
    #     genes_json_path = tmp_file.name
    genes_json_path = 'temp_genes_name_1.json'

    with open(genes_json_path, 'w') as f:
        # `genes` will now be a list of dictionaries
        json.dump({"genes": genes}, f)


    command = [
        blender_executable_path,
        "-b", # Run in background (headless)
        # Removed opening .blend file from here; script handles it via its own arg
        # blend_file_to_open, # This would open it BEFORE the script runs
        "-P", analyzer_script_path,
        "--", # Separator for script arguments
        "--input", blend_file_to_open, # Pass blend file as arg to script
        "--genes", genes_json_path,
        "--output", output_file,
    ]
    print(f"Running face making: {' '.join(command)}")

    try:
        # Setting PYTHONNOUSERSITE and PYTHONSAFEPATH can sometimes help avoid conflicts
        # with user's local Python packages if Blender's Python environment is sensitive.
        env = os.environ.copy()
        env["PYTHONNOUSERSITE"] = "1"
        env["PYTHONSAFEPATH"] = "1"

        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        stdout, stderr = process.communicate(timeout=timeout_seconds)

        if process.returncode != 0:
            print(f"Error running Blender script. Return code: {process.returncode}")
            return None
    
        return True
        
    except subprocess.TimeoutExpired:
        print(f"Error: Blender script timed out after {timeout_seconds} seconds.")
        if process: process.kill() # Ensure process is killed
        # stdout_on_timeout, stderr_on_timeout = process.communicate() # Get any remaining output
        # print("--- STDOUT (on timeout) ---\n", stdout_on_timeout)
        # print("--- STDERR (on timeout) ---\n", stderr_on_timeout)
        return None
    except Exception as e:
        print(f"An unexpected error occurred while running Blender script: {e}")
        import traceback
        traceback.print_exc()
        return None
    

def run_emotion_analysis(
    blender_executable_path: str,
    analyzer_script_path: str,
    blend_file_to_open: str,
    image_file_name: str,
    timeout_seconds: int = 300 # 5 minutes
    ):

    # /Applications/Blender.app/Contents/MacOS/Blender -b eyes_open_landmark_EMOTE.blend -P face_emote.py
    if not os.path.exists(blender_executable_path):
        print(f"Error: Blender executable not found at '{blender_executable_path}'")
        return None
    if not os.path.exists(analyzer_script_path):
        print(f"Error: Blender analyzer script not found at '{analyzer_script_path}'")
        return None
    if not os.path.exists(blend_file_to_open):
        print(f"Error: Specified .blend file not found at '{blend_file_to_open}'")
        return None

    command = [
        blender_executable_path,
        "-b", # Run in background (headless)
        "-P", analyzer_script_path,
        "--", # Separator for script arguments
        "--blend_file", blend_file_to_open, # Pass blend file as arg to script
        "--image_file_name", image_file_name,
    ]
    
    print(f"Executing command: {' '.join(command)}")

    try:
        # Setting PYTHONNOUSERSITE and PYTHONSAFEPATH can sometimes help avoid conflicts
        # with user's local Python packages if Blender's Python environment is sensitive.
        env = os.environ.copy()
        env["PYTHONNOUSERSITE"] = "1"
        env["PYTHONSAFEPATH"] = "1"

        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        

        if process.returncode != 0:
            print(f"Error running Blender script. Return code: {process.returncode}")
            return None
        else:
            # {'angry': 1.2861880309097786, 'disgust': 0.000818852325814356, 'fear': 29.925556774470614, 'happy': 0.22580385157041857, 'sad': 7.912326330681225, 'surprise': 0.010981416489519788, 'neutral': 60.63832406034017}
            # Parse the specific output line
            # Expected format: 'nx <val> ny <val> nz <val> solid_volume <val> trapped_water_vol <val> cupped_water_vol <val>'
            output_lines = stdout.strip().split('\n')
            results_line = None
            for line in reversed(output_lines): # Find the last line that matches
                if 'angry' in line:
                    results_line = line
                    break
            try:
                return ast.literal_eval(results_line)
            except:
                return None

    except subprocess.TimeoutExpired:
        print(f"Error: Blender script timed out after {timeout_seconds} seconds.")
        if process: process.kill() # Ensure process is killed
        # stdout_on_timeout, stderr_on_timeout = process.communicate() # Get any remaining output
        # print("--- STDOUT (on timeout) ---\n", stdout_on_timeout)
        # print("--- STDERR (on timeout) ---\n", stderr_on_timeout)
        return None
    except Exception as e:
        print(f"An unexpected error occurred while running Blender script: {e}")
        import traceback
        traceback.print_exc()
        return None



def run_blender_analysis(
    blender_executable_path: str,
    analyzer_script_path: str,
    blend_file_to_open: str,
    object_name_in_blend: str,
    voxel_s: float,
    create_debug: bool = False,
    verbose_blender_output: bool = False,
    timeout_seconds: int = 300 # 5 minutes
    ):
    """
    Runs the Blender puddle analysis script headlessly.

    Args:
        blender_executable_path (str): Full path to the Blender executable.
        analyzer_script_path (str): Full path to the 'puddle_analyzer_headless.py'.
        blend_file_to_open (str): Path to the .blend file (e.g., 'face_blend.blend').
        object_name_in_blend (str): Name of the object to analyze.
        voxel_s (float): Voxel size.
        create_debug (bool): If True, tells Blender script to create debug meshes.
        verbose_blender_output (bool): If True, tells Blender script to print verbose logs.
        timeout_seconds (int): Timeout for the Blender process.

    Returns:
        dict or None: Parsed results (nx, ny, nz, solid_volume, trapped_water_vol, cupped_water_vol)
                      or None if an error occurred or output couldn't be parsed.
    """

    if not os.path.exists(blender_executable_path):
        print(f"Error: Blender executable not found at '{blender_executable_path}'")
        return None
    if not os.path.exists(analyzer_script_path):
        print(f"Error: Blender analyzer script not found at '{analyzer_script_path}'")
        return None
    if not os.path.exists(blend_file_to_open):
        print(f"Error: Specified .blend file not found at '{blend_file_to_open}'")
        return None

    command = [
        blender_executable_path,
        "-b", # Run in background (headless)
        # Removed opening .blend file from here; script handles it via its own arg
        # blend_file_to_open, # This would open it BEFORE the script runs
        "-P", analyzer_script_path,
        "--", # Separator for script arguments
        "--blend_file", blend_file_to_open, # Pass blend file as arg to script
        "--object_name", object_name_in_blend,
        "--voxel_size", str(voxel_s),
    ]

    if create_debug:
        command.append("--create_debug_meshes")
    if verbose_blender_output:
        command.append("--verbose")
    
    print(f"Executing command: {' '.join(command)}")

    try:
        # Setting PYTHONNOUSERSITE and PYTHONSAFEPATH can sometimes help avoid conflicts
        # with user's local Python packages if Blender's Python environment is sensitive.
        env = os.environ.copy()
        env["PYTHONNOUSERSITE"] = "1"
        env["PYTHONSAFEPATH"] = "1"

        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        
        if verbose_blender_output or process.returncode != 0:
            print("--- Blender Script STDOUT ---")
            print(stdout)
            print("--- Blender Script STDERR ---")
            print(stderr)

        if process.returncode != 0:
            print(f"Error running Blender script. Return code: {process.returncode}")
            return None
        else:
            # Parse the specific output line
            # Expected format: 'nx <val> ny <val> nz <val> solid_volume <val> trapped_water_vol <val> cupped_water_vol <val>'
            output_lines = stdout.strip().split('\n')
            results_line = None
            for line in reversed(output_lines): # Find the last line that matches
                if line.startswith("nx ") and "solid_volume" in line:
                    results_line = line
                    break
        
            results_line

            # Split the string into a list
            items = results_line.split()

            # Convert to dictionary by pairing adjacent items
            result = {items[i]: int(items[i+1]) for i in range(0, len(items), 2)}

            return result

    except subprocess.TimeoutExpired:
        print(f"Error: Blender script timed out after {timeout_seconds} seconds.")
        if process: process.kill() # Ensure process is killed
        # stdout_on_timeout, stderr_on_timeout = process.communicate() # Get any remaining output
        # print("--- STDOUT (on timeout) ---\n", stdout_on_timeout)
        # print("--- STDERR (on timeout) ---\n", stderr_on_timeout)
        return None
    except Exception as e:
        print(f"An unexpected error occurred while running Blender script: {e}")
        import traceback
        traceback.print_exc()
        return None

