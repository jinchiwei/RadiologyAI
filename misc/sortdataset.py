# Idea: Take source and new image directories as arguments.
# Move all file names from first directory to second directory, etc.
# Repeat for second group of files, etc.

import os

dir1 = input("Source directory: ")
dir2 = input("New training directory: ")
dir3 = input("New validation directory: ")
dir4 = input("New testing directory: ")
num1 = int(input("Number of training files: "))
num2 = int(input("Number of validation files: "))
num3 = int(input("Number of test files: "))

onlyfiles = [f for f in os.listdir(dir1) if os.path.isfile(os.path.join(dir1, f))]

# for i in range(6):
# 	AP_line = AP.readline()

for i in range(num1):
	os.rename(dir1 + "/" + onlyfiles[i], dir2 + "/" + onlyfiles[i])

for i in range(num1, num1 + num2):
	os.rename(dir1 + "/" + onlyfiles[i], dir3 + "/" + onlyfiles[i])

for i in range(num1 + num2, num1 + num2 + num3):
	os.rename(dir1 + "/" + onlyfiles[i], dir4 + "/" + onlyfiles[i])

