import os


def rename_files_in_directory(directory):
    # List all files in the given directory
    files = [f for f in os.listdir(directory) if f.endswith('.jpg')]

    # Sort files to maintain a consistent order
    files.sort()

    # Rename each file
    for index, filename in enumerate(files):
        # Construct new file name
        new_name = f"Healthy{index + 1}.jpg"

        # Get full path for the old and new file names
        old_file = os.path.join(directory, filename)
        new_file = os.path.join(directory, new_name)

        # Rename the file
        os.rename(old_file, new_file)
        print(f'Renamed: {old_file} to {new_file}')


# Specify the directory where your files are located
directory_path = 'rice_leaf_diseases/Healthy'
rename_files_in_directory(directory_path)
