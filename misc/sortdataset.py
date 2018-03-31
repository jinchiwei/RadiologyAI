#!/bin/bash - unused since this is now Python

# Idea: Take a txt file and 4 directories as arguments.
# Read the txt file and take file names.
# Move all file names from first directory to second directory, etc.
# Repeat for second group of files, etc.

import os

# datalistfile = input("List of dataset files: ")

dir1 = input("Source directory: ")
dir2 = input("New training directory: ")
dir3 = input("New validation directory: ")
dir4 = input("New testing directory: ")
num1 = int(input("Number of training files: "))
num2 = int(input("Number of validation files: "))
num3 = int(input("Number of test files: "))

onlyfiles = [f for f in os.listdir(dir1) if os.path.isfile(os.path.join(dir1, f))]

# for i in range(6):
# 	AP_line = AP.read()

# datalist = open(datalistfile, "r")

for i in range(num1):
	os.rename(dir1 + "/" + onlyfiles[i], dir2 + "/" + onlyfiles[i])

for i in range(num1, num1 + num2):
	os.rename(dir1 + "/" + onlyfiles[i], dir3 + "/" + onlyfiles[i])

for i in range(num1 + num2, num1 + num2 + num3):
	os.rename(dir1 + "/" + onlyfiles[i], dir4 + "/" + onlyfiles[i])


# for filename in AP:
#	try:
#		filename_noline = filename.replace("\n", "")
#		os.rename(dir1 + "/" + filename_noline, dir2 + "/" + filename_noline)
#	except FileNotFoundError:
#		continue

# datalist.close()
