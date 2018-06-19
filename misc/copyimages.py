#!/bin/bash - unused since this is now Python

# Idea: Take a txt file and two directories as arguments.
# Read the txt file and take file names.
# Move all file names from first directory to second directory.
# Repeat for second group of files.

import os
from shutil import copyfile

filelistAP = input("List of grouped files: ")
dir1 = input("Source directory: ")
dir2 = input("New group directory: ")

# for i in range(6):
# 	AP_line = AP.read()

AP = open(filelistAP, "r")
for filename in AP:
	try:
		filename_noline = filename.replace("\n", "")
		copyfile(dir1 + "/" + filename_noline, dir2 + "/" + filename_noline)
	except FileNotFoundError:
		continue

AP.close()
