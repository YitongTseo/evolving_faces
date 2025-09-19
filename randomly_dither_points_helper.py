import subprocess
import json
import os
import tempfile

def get_dithered_vertices(input, num_points_to_dither=20, min_dither=0.5, max_dither=5, exempt_vertices=[], blender_executable='/Applications/Blender.app/Contents/MacOS/Blender'):
    """
    Wrapper function to call the Blender vertex dithering script.
    
    Args:
        input_file (str): Path to input .blend file
        num_points_to_dither (int): Number of vertices to dither
        min_dither (float): Minimum dithering magnitude
        max_dither (float): Maximum dithering magnitude  
        exempt_vertices (list): List of vertex indices to exempt from dithering
        seed (int): Random seed for reproducibility
        blender_executable (str): Path to Blender executable
    
    Returns:
        list: List of gene dictionaries with vertex dithering information
    """
    # Create temporary file for exempt_vectors 
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
        exempt_vertices_json_path = tmp_file.name
    # exempt_vertices_json_path = 'exempt_vertices_json_path.json'
    with open(exempt_vertices_json_path, 'w') as f:
        # `genes` will now be a list of dictionaries
        json.dump({"exempt_vertices": exempt_vertices}, f)
    
    
    # Create temporary file for output
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
        output_json_path = tmp_file.name

    # output_json_path = 'output_gene_json_path_.json'
    try:
        # Prepare command arguments
        cmd = [
            blender_executable,
            '--background',  # Run in background
            '--python', 'randomly_dither_points.py',  # Your Blender script
            '--',  # Arguments separator
            '--input', input,
            '--output', output_json_path,
            '--num_points', str(num_points_to_dither),
            '--min_dither', str(min_dither),
            '--max_dither', str(max_dither),
            '--exempt_vertices', exempt_vertices_json_path
        ]
        
        print(f"Running Blender vertex dithering: {' '.join(cmd)}")
        
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            print(f"Blender script failed with return code {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            raise RuntimeError(f"Blender vertex dithering script failed: {result.stderr}")
        
        # Load the results
        if not os.path.exists(output_json_path):
            raise RuntimeError("Output JSON file was not created")
        
        with open(output_json_path, 'r') as f:
            output_data = json.load(f)
        
        genes = output_data.get('genes', [])
        print(f"Successfully generated {len(genes)} vertex dithering genes")
        return genes
        
    except subprocess.TimeoutExpired:
        raise RuntimeError("Blender vertex dithering script timed out")
    except Exception as e:
        print(f"Error in get_dithered_vertices wrapper: {e}")
        raise
    finally:
        # Clean up temporary file
        if os.path.exists(output_json_path):
            os.unlink(output_json_path)