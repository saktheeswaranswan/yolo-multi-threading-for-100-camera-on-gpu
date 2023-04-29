import os

# Replace the "folder_path" variable with the path to your folder
folder_path = "/home/saktheeswaran/borewell baby rescue/teachelec/TRACKERCLONEIMOO/bulkyo"

# Loop through all files in the folder
for file_name in os.listdir(folder_path):
    # Check if the file is a text file
    if file_name.endswith(".txt"):
        # Get the full path to the file
        file_path = os.path.join(folder_path, file_name)

        # Open the file in read mode
        with open(file_path, "r") as file:
            # Read the first line of the file
            first_line = file.readline().rstrip("\n")

        # Replace the first '0' in the leftmost corner with '4'
        updated_first_line = "8" + first_line[1:].replace("0", "", 1)

        # Open the file in write mode
        with open(file_path, "r+") as file:
            # Move the file pointer to the beginning of the file
            file.seek(0)

            # Replace the first line with the updated first line
            file.write(updated_first_line)

            # Move the file pointer to the end of the first line
            file.seek(len(updated_first_line))

            # Copy the rest of the file to a temporary buffer
            temp_buffer = file.read()

            # Move the file pointer back to the beginning of the first line
            file.seek(0)

            # Write the updated first line followed by the rest of the file
            file.write(updated_first_line + temp_buffer)

