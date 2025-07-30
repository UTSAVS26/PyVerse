#!/usr/bin/env python3

import os
import argparse
from typing import List

def list_valid_dirs(path: str) -> List[str]:
    """
    Scans a given path and returns a sorted list of valid sub-directories.

    Args:
        path (str): The directory path to scan.

    Returns:
        List[str]: A sorted list of directory names that pass the filter rules.
    """
    valid_dirs = []
    if not os.path.isdir(path):
        return valid_dirs

    for name in os.listdir(path):
        # Check if the item is a directory
        if os.path.isdir(os.path.join(path, name)):
            # Apply filtering rules to exclude utility/hidden folders
            if name != "scripts" and not name.startswith(('.', '_')) and "venv" not in name:
                valid_dirs.append(name)
    
    valid_dirs.sort()
    return valid_dirs

def print_project_list_md(top_dir: str = ".") -> None:
    """
    Prints a nested Markdown list of project directories.
    It scans a two-level hierarchy (category/project).

    Args:
        top_dir (str): The root directory to start searching from.
    """
    print("## Project Directory\n")
    
    # Get the top-level category directories (e.g., 'dir1', 'dir2')
    category_dirs = list_valid_dirs(top_dir)
    
    if not category_dirs:
        print("No category directories found.")
        return

    # Loop through each top-level category directory
    for category_dir in category_dirs:
        category_title = category_dir.replace('_', ' ').title()
        
        # Print the category name as a bolded, top-level list item
        print(f"* **{category_title}**")
        
        # Now find the project sub-directories inside the category
        path_to_category = os.path.join(top_dir, category_dir)
        project_dirs = list_valid_dirs(path_to_category)
        
        if project_dirs:
            # Loop through each project in the category and print it as an indented list item
            for project_dir in project_dirs:
                project_title = project_dir.replace('_', ' ').title()
                
                # Create a URL-safe link to the project directory
                # The path must be relative to the root (e.g., './dir1/p1')
                url = f"./{category_dir}/{project_dir}".replace(" ", "%20")
                
                # Print the indented list item with the hyperlink
                print(f"  * [{project_title}]({url})")
        else:
            # Optional: Note if a category directory is empty
            print("  * (No projects found in this category)")
            
        print() # Add a blank line for better spacing between categories

def main():
    """Main function to parse arguments and run the script."""
    parser = argparse.ArgumentParser(description="Generate a nested Markdown list of project directories.")
    parser.add_argument("directory", nargs="?", default=".", help="The root directory to scan (default: current directory).")
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Error: Directory '{args.directory}' does not exist.")
        return

    print_project_list_md(args.directory)

if __name__ == "__main__":
    main()
