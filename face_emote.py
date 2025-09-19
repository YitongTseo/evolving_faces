# /Applications/Blender.app/Contents/MacOS/Blender -b eyes_open_landmark_EMOTE.blend -P face_emote.py
import tensorflow as tf
from deepface import DeepFace
import bpy
import sys
import argparse # Added for command-line arguments

def eval_emotion(img_path="quick_render.png"):
    result = DeepFace.analyze(img_path, actions=['emotion'])
    print(result[0]['emotion'])


def take_picture(name="quick_render.png"):
    # Set render resolution (optional)
    bpy.context.scene.render.resolution_x = 500
    bpy.context.scene.render.resolution_y = 500  # tall/skinnier
    bpy.context.scene.render.resolution_percentage = 100

    # Set output path and format
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.filepath = name

    camera = bpy.data.objects.get("Camera")
    if camera and camera.type == 'CAMERA':
        cam_data = camera.data
        cam_data.sensor_fit = 'HORIZONTAL'  # or 'VERTICAL'
        cam_data.sensor_width = 36  # adjust width
        cam_data.sensor_height = 48  # adjust height

    bpy.ops.render.render(write_still=True)


# Helper to get script-specific arguments after '--'
def get_blender_script_args():
    try:
        return sys.argv[sys.argv.index("--") + 1:]
    except ValueError:
        return []


if __name__ == "__main__":
    script_args = get_blender_script_args()

    parser = argparse.ArgumentParser(description="Blender Headless Puddle Analyzer")
    parser.add_argument("--blend_file", type=str, help="Path to the .blend file to open.")
    parser.add_argument("--image_file_name", type=str, required=True, help="image_file_name.")

    args = parser.parse_args(args=script_args)

    if args.blend_file:
        try:
            bpy.ops.wm.open_mainfile(filepath=args.blend_file)
        except RuntimeError as e:
            print(f"Error opening .blend file '{args.blend_file}': {e}")
            sys.exit(1) # Exit if file cannot be opened

    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    
    take_picture(args.image_file_name)
    eval_emotion(args.image_file_name)
