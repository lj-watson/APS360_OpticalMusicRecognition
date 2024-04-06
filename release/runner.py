'''
@breif main python file to run other scripts
Place the input image in the release folder and name it input.png
'''

import subprocess
import os

def run_script(path):
    # Extract the directory path relative to this script
    script_dir = os.path.dirname(path)
    # Get the script name
    script_name = os.path.basename(path)
    # Get the full path to the directory
    full_script_dir_path = os.path.join(os.getcwd(), script_dir)
    # Run the script using Python, changing the working directory to the script's directory
    subprocess.run(['python', script_name], cwd=full_script_dir_path, check=True)

run_script('../classify/staff_removal.py')
run_script('../classify/region_proposal.py')
run_script('../classify/get_staff_y_values.py')
run_script('../classify/clean_input.py')
run_script('../classify/classify_folder.py')
run_script('../audio/get_octaves.py')
run_script('../audio/midi.py')