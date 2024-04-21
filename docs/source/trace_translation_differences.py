# This script is made for purpose of detecting what files were added or changed
# to detect the differences in translation and update them in an efficient way.

import os
import sys

def print_stats(title_of_stats, files):
	print(f"{title_of_stats}:")
	for file in files:
		print(f" * {file}")
	print('')

def compare_folders(source_dir, target_dir):
	def list_files(directory):
		file_paths = {}
		for root, _, files in os.walk(directory):
			for file in files:
				full_path = os.path.join(root, file)
				relative_path = os.path.relpath(full_path, directory)
				file_paths[relative_path] = os.path.getmtime(full_path)
		return file_paths

	source_files = list_files(source_dir)
	target_files = list_files(target_dir)

	added = [file for file in source_files if file not in target_files]
	removed = [file for file in target_files if file not in source_files]
	modified = [file for file in source_files if file in target_files and source_files[file] > target_files[file]]

	return added, modified, removed

translation_dir = input("Enter the translation directory (like 'pl'): ")

if not os.path.isdir(translation_dir):
	print(f"Directory '{translation_dir}' does not exists.")
	sys.exit(1)

added, modified, removed = compare_folders('en', translation_dir)

print_stats("Added files", added)
print_stats("Modified files", modified)
print_stats("Removed files", removed)
