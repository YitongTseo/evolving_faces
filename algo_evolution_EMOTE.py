import json
import os
import csv
import random
import shutil
import uuid
import numpy as np
import functools
# from deap import base, creator, tools, algorithms
from utils import generate_face_mesh, run_emotion_analysis, merge_close_genes
from randomly_dither_points_helper import get_dithered_vertices
import pandas as pd
from tqdm import tqdm


# --- Configuration ---
BLENDER_EXECUTABLE_PATH = '/Applications/Blender.app/Contents/MacOS/Blender'
GENERATION_HELPER_SCRIPT_PATH = 'rando_blender_rbf_script.py'
ANALYSIS_SCRIPT_PATH = 'face_emote.py'
BASE_INPUT_BLEND_FILE = 'eyes_open_landmark_EMOTE.blend'
OBJECT_NAME_TO_ANALYZE = "eyes_open_mask"

POPULATION_SIZE = 2000
N_GENERATIONS = 3
TOP_PARENTS_TOTAL = 12  # Top 25 of parents for breeding
GEN0_FACES = ['284136g8', '32159b42', 'd9346748', 'df1aaea1', '656029da', '5a44c7da', 'a01b13cd', 'a33c6fe5','7a5d46ab']
GEN1_FACES = [
    '71157398',  'f94ec936', '8ebe8b02',
    'ab83bbf7',   '4a2e5720', 'adfd099e', 
    '9576e2df',  '1b81e25d',  'e54cbcce',
    '6e8de1d9', 'e52b237e', 'd53a9a9c',  
]

# These are the params for the dithering...
NUM_POINTS_TO_DITHER = [10, 20, 30, 100]
MIN_DITHER = [1.5, 0.5, 0.0, 0.0]
MAX_DITHER = [8.5, 5, 7, 10]

STARTING_GEN = 2
ENDING_GEN = 3

# --- Temporary Directory for EA files ---
TEMP_DIR = "INDIVIDUALS_EVOLVING_EMOTE_BIG_RUYN" # Changed by user
# if os.path.exists(TEMP_DIR):
#     shutil.rmtree(TEMP_DIR)
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR, exist_ok=True)

import hashlib

def hash_id(text, length=8):
    return hashlib.md5(text.encode()).hexdigest()[:length]

def evaluate_and_log_individual(
    individual_genes, current_generation_num,
    parent1_individual_id_for_log, parent2_individual_id_for_log, #input_blend_for_this_eval,
     csv_writer_object, csv_file_handle
):
    # we always use the base_input_blend_file now...
    global BASE_INPUT_BLEND_FILE
    input_blend_for_this_eval = BASE_INPUT_BLEND_FILE
    # Maybe this was the better way of making ids?
    # individual_id = str(uuid.uuid4()) # Unique ID for this new individual being evaluated
    individual_id = hash_id(str(sorted(list(individual_genes))))
    blend_filename = f"output_gen{current_generation_num}_{individual_id}.blend" # Output of this eval
    render_facepic = f"output_gen{current_generation_num}_{individual_id}.png" # Output of this eval

    generated_blend_path = os.path.join(TEMP_DIR, blend_filename)
    render_facepic_path = os.path.join(TEMP_DIR, render_facepic)

    log_entry = {
        "generation": current_generation_num,
        "individual_id": individual_id,
        "parent1_individual_id": parent1_individual_id_for_log,
        "parent2_individual_id": parent2_individual_id_for_log,
        'face_pic': render_facepic_path,
        "blend_file": None, # Output blend file of this individual
        'emotion_key': None,
        "fitness": None,
        "angry": None, "disgust": None, "fear": None, "happy": None,
        "sad": None, "surprise": None, "neutral": None,
        "generation_status": "pending", "analysis_status": "pending",
        "genes": sorted(list(individual_genes)), # Genes of this individual
    }

    success_generation = False
    try:
        # Use the passed input_blend_for_this_eval
        success_generation = generate_face_mesh(
            blender_executable_path=BLENDER_EXECUTABLE_PATH,
            analyzer_script_path=GENERATION_HELPER_SCRIPT_PATH,
            blend_file_to_open=input_blend_for_this_eval, # IMPORTANT: Use parent's output or base
            genes=individual_genes,
            output_file=generated_blend_path,
        )
    except Exception as e:
        print(f"  Exception during generate_face_mesh for {individual_id} (using {os.path.basename(input_blend_for_this_eval)}): {e}")
        log_entry["generation_status"] = f"error: {e}"

    current_fitness = -float('inf') 
    
    if not success_generation or not os.path.exists(generated_blend_path):
        if success_generation and log_entry["generation_status"] == "pending":
             log_entry["generation_status"] = "failed_no_output"
    else:
        log_entry["generation_status"] = "success"
        log_entry["blend_file"] = blend_filename # This individual's output blend

        analysis_results = None
        try:
            analysis_results = run_emotion_analysis(
                blender_executable_path=BLENDER_EXECUTABLE_PATH,
                analyzer_script_path=ANALYSIS_SCRIPT_PATH,
                blend_file_to_open=generated_blend_path, # Analyze this individual's output
                image_file_name=render_facepic_path,
            )
        except Exception as e:
            print(f"  Exception during run_blender_analysis for {individual_id}: {e}")
            log_entry["analysis_status"] = f"error: {e}"
        
        # deleting the blender file to save space... I think this is OK since we can just recreate it from the genes...
        os.remove(generated_blend_path)
        
        if analysis_results is None:
            if log_entry["analysis_status"] == "pending":
                log_entry["analysis_status"] = "failed_no_results"
        else:
            log_entry["analysis_status"] = "success"
            # NOTE: no neutral...
            full_emotional_cast= ['angry', 'disgust', 'fear',  'sad', 'happy', 'surprise']
            emotional_cast = ['angry', 'disgust', 'fear',  'sad'] # NO surprise or happy... 'surprise', 'happy',
            emotions = [(emote, analysis_results[emote]) for emote in emotional_cast]
            # Subtract out the neutral-ness of the face...
            max_emotion = max(emotions, key=lambda x: x[1]) 
            log_entry['emotion_key'] = max_emotion[0]
            log_entry.update(analysis_results)
            other_emotions = sum([analysis_results[emote] for emote in full_emotional_cast if max_emotion[0] != emote])
            current_fitness = max_emotion[1] - other_emotions - ((3 * analysis_results['neutral']))
            print('top emotion rating:', max_emotion[1],  ' other_emotions:', other_emotions, ' neutral:', ((3 * analysis_results['neutral'])))
            print('current_fitness:', current_fitness)

            

    log_entry['fitness'] = current_fitness

    # detailed_log_list_in_memory.append(log_entry)
    csv_writer_object.writerow(log_entry)
    csv_file_handle.flush()
    
    return log_entry

def convert_row(row: dict) -> dict:
    converted = {}
    for key, val in row.items():
        if val is None:
            converted[key] = None
            continue

        # Try JSON decoding first (for things like lists)
        try:
            parsed = json.loads(val)
            converted[key] = parsed
            continue
        except (json.JSONDecodeError, TypeError):
            pass

        # Try integer
        try:
            converted[key] = int(val)
            continue
        except ValueError:
            pass

        # Try float
        try:
            converted[key] = float(val)
            continue
        except ValueError:
            pass

        # Leave as string if nothing else worked
        converted[key] = val

    return converted


def get_top_parents(all_individuals_detailed_log_in_memory, current_gen, total):
    successful_individuals = []
    for entry in all_individuals_detailed_log_in_memory:
        # TODO: grab all of the successful individuals from this generation and pull them up by fitness
        if entry["generation"] == current_gen and entry["generation_status"] == "success" and entry["analysis_status"] == "success":
            successful_individuals.append(entry)

    # Sort by fitness (descending for maximization)
    successful_individuals.sort(key=lambda x: x['fitness'], reverse=True)
    # Return the top total parents
    num_top_parents = max(2, total)  # At least 2 parents
    return successful_individuals[:num_top_parents]


def main():
    print("Evolutionary Algorithm: Two-Parent Crossover with Top 25% Parent Selection")
    print(f"Output and logs will be in: {os.path.abspath(TEMP_DIR)}")
    print("---")

    all_individuals_detailed_log_in_memory = []

    detailed_log_csv_path = os.path.join(TEMP_DIR, "all_individuals_evaluation_log.csv")
    csv_fieldnames = [
        "generation", "individual_id", "parent1_individual_id", "parent2_individual_id", 
        'face_pic', "blend_file", "emotion_key", "fitness", "angry", "disgust", "fear", "happy",
        "sad", "surprise", "neutral",
        "generation_status", "analysis_status", "genes",
    ]

    # with open(detailed_log_csv_path, 'a+', newline='') as csvfile_handle:
    #     csv_writer = csv.DictWriter(csvfile_handle, fieldnames=csv_fieldnames)
    #     csvfile_handle.seek(0)
    #     is_empty = not csvfile_handle.read(1)
    #     if is_empty:
    #         csvfile_handle.seek(0)
    #         csv_writer.writeheader()
    with open(detailed_log_csv_path, 'a+', newline='') as csvfile_handle:
        csv_writer = csv.DictWriter(csvfile_handle, fieldnames=csv_fieldnames)

        # Check if file is empty
        csvfile_handle.seek(0)
        is_empty = not csvfile_handle.read(1)

        if is_empty:
            csvfile_handle.seek(0)
            csv_writer.writeheader()
        else:
            # Read entire CSV into dict
            csvfile_handle.seek(0)
            csv_reader = csv.DictReader(csvfile_handle)
            for row in csv_reader:
                all_individuals_detailed_log_in_memory.append(convert_row(row))
        all_individuals_detailed_log_in_memory = [ind for ind in all_individuals_detailed_log_in_memory if ((ind['generation'] != 0) or (ind['individual_id'] in GEN0_FACES))]
        all_individuals_detailed_log_in_memory = [ind for ind in all_individuals_detailed_log_in_memory if ((ind['generation'] != 1) or (ind['individual_id'] in GEN1_FACES))]

        # Generation 0
        print("Evaluating initial population (Generation 0)...")
        if 0 == STARTING_GEN:
            for i in tqdm(range(POPULATION_SIZE)): # ind is a list of genes
                # print('whats ind? ', ind)
                print(f"  Gen 0, Eval Ind {i+1}/{POPULATION_SIZE}")
                individual_genes = get_dithered_vertices(
                    input=BASE_INPUT_BLEND_FILE,
                    num_points_to_dither=NUM_POINTS_TO_DITHER[0],
                    min_dither=MIN_DITHER[0],
                    max_dither=MAX_DITHER[0],
                    exempt_vertices=[] # nothing here yet since its the base run
                )
                original_positions, dithered_positions = [], []
                for orig_position, dith_position in individual_genes:
                    original_positions.append(orig_position)
                    dithered_positions.append(dith_position)

                merged_orig, merged_dith = merge_close_genes(original_positions, dithered_positions)
                print(
                    'merging cut: ', 
                    len(original_positions) - len(merged_orig), 
                    ' original and cut: ',
                    len(dithered_positions) - len(merged_dith), 
                )
                individual_genes = [[o, d] for o, d in zip(merged_orig, merged_dith)]
                log_entry = evaluate_and_log_individual(
                    individual_genes=individual_genes, current_generation_num=0,
                    parent1_individual_id_for_log="INITIAL",
                    parent2_individual_id_for_log="INITIAL",
                    csv_writer_object=csv_writer, csv_file_handle=csvfile_handle
                )
                all_individuals_detailed_log_in_memory.append(log_entry)
                print('fitness value: ', log_entry['fitness'], ' for ', log_entry['emotion_key'])
        
        # Subsequent Generations
        for gen in range(1, N_GENERATIONS + 1):
            if gen < STARTING_GEN or gen >= ENDING_GEN:
                continue

            print(f"\n--- Generation {gen}/{N_GENERATIONS} ---")
            
            # Get top parents from previous generation
            top_parent_logs = get_top_parents(all_individuals_detailed_log_in_memory, gen - 1, TOP_PARENTS_TOTAL)

            print(f"  Generating and evaluating {POPULATION_SIZE} offspring for generation {gen}...")
            for i in tqdm(range(POPULATION_SIZE)):
                # Select two random parents from the top parents
                parent1_log, parent2_log = random.sample(top_parent_logs, 2)
                parent1_id, parent2_id = parent1_log['individual_id'], parent2_log['individual_id']
                child_genes = parent1_log['genes'] + parent2_log['genes']
                new_mutations = get_dithered_vertices(
                    input=BASE_INPUT_BLEND_FILE,
                    num_points_to_dither=NUM_POINTS_TO_DITHER[gen],
                    min_dither=MIN_DITHER[gen],
                    max_dither=MAX_DITHER[gen],
                    exempt_vertices=[gene[0] for gene in child_genes]
                )
                child_genes = child_genes + new_mutations


                original_positions, dithered_positions = [], []
                for orig_position, dith_position in child_genes:
                    original_positions.append(orig_position)
                    dithered_positions.append(dith_position)

                merged_orig, merged_dith = merge_close_genes(original_positions, dithered_positions)
                print(
                    'merging cut: ', 
                    len(original_positions) - len(merged_orig), 
                    ' original and cut: ',
                    len(dithered_positions) - len(merged_dith), 
                )
                child_genes = [[o, d] for o, d in zip(merged_orig, merged_dith)]
                
                # Evaluate the child
                log_entry = evaluate_and_log_individual(
                    individual_genes=child_genes, 
                    current_generation_num=gen,
                    parent1_individual_id_for_log=parent1_id,
                    parent2_individual_id_for_log=parent2_id,
                    csv_writer_object=csv_writer, csv_file_handle=csvfile_handle
                )
                all_individuals_detailed_log_in_memory.append(log_entry)

            print(f"Detailed CSV log: {os.path.abspath(detailed_log_csv_path)}")

    print("\n--- Evolution Finished ---")
    print(f"Detailed incremental log saved to: {os.path.abspath(detailed_log_csv_path)}")

if __name__ == "__main__":
    main()
