#!/bin/bash - unused since this is now Python

# Idea: Take a txt file and two directories as arguments.
# Read the txt file and take file names.
# Move all file names from first directory to second directory.
# Repeat for second group of files.

import os

filelistAP = input("List of AP files: ")
dir1 = input("Source directory: ")
dir2 = input("New AP directory: ")

# for i in range(6):
# 	AP_line = AP.read()

AP = open(filelistAP, "r")
for filename in AP:
	filename_noline = filename.replace("\n", "")
	os.rename(dir1 + "/" + filename_noline, dir2 + "/" + filename_noline)

AP.close()

filelistPA = input("List of PA files: ")
dir2 = input("New PA directory: ")

PA = open(filelistPA, "r")
for filename in PA:
	filename_noline = filename.replace("\n", "")
	os.rename(dir1 + "/" + filename_noline, dir2 + "/" + filename_noline)

PA.close()
