import os
import sys

def convert_xyzrgb_to_ply(input_file, output_file):
    """
    Converts a file in XYZRGB format to a PLY file.

    XYZRGB format is a text file where each line contains 6 values: x, y, z, r, g, b.
    x, y, z are the point coordinates and r, g, b are the point colors in RGB format.

    The generated PLY file is a text file in ASCII format, with a header followed by
    the vertex data. The header contains the number of vertices, and the vertex data
    contains the x, y, z coordinates and the RGB color values for each vertex.

    :param input_file: The input file in XYZRGB format
    :param output_file: The output file in PLY format
    """
    
    # Read the XYZRGB file
    points = []
    with open(input_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:  # Make sure we have 6 values (x, y, z, r, g, b)
                x, y, z, r, g, b = map(float, parts[:6])
                # Convert RGB values to integers if they're not already
                r, g, b = int(r), int(g), int(b)
                points.append((x, y, z, r, g, b))
    
    # Write the PLY file
    with open(output_file, 'w') as f:
        # Write header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # Write vertex data
        for x, y, z, r, g, b in points:
            f.write(f"{x} {y} {z} {r} {g} {b}\n")

    print(f"Converted: {input_file} → {output_file}")

def batch_convert_folder(input_folder):
    """
    Converts all .xyzrgb files in a given folder to PLY files.

    :param input_folder: The path to the folder containing XYZRGB files
    """
    
    # Create output folder if it doesn't exist
    output_folder = os.path.join(input_folder, "ply_output")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")
    
    # Counter for tracking files processed
    files_processed = 0
    
    # Process each .xyzrgb file in the folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.xyzrgb'):
            input_path = os.path.join(input_folder, filename)
            output_filename = os.path.splitext(filename)[0] + '.ply'
            output_path = os.path.join(output_folder, output_filename)
            
            try:
                convert_xyzrgb_to_ply(input_path, output_path)
                files_processed += 1
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    print(f"\nConversion complete. Processed {files_processed} files.")
    print(f"PLY files saved to: {output_folder}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python batch_convert_xyzrgy_to_ply.py input_folder")
        print("Or run interactively by executing the script without arguments.")
        
        # Interactive mode
        input_folder = input("Enter the path to the folder containing XYZRGB files: ")
    else:
        input_folder = sys.argv[1]
    
    if not os.path.isdir(input_folder):
        print(f"Error: {input_folder} is not a valid directory.")
        sys.exit(1)
        
    batch_convert_folder(input_folder)