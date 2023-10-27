#!/bin/bash

# remove any previously unzipped copies of Shell1/
if [ -d Shell1 ];
then
  echo "Removing old copies of Shell1/..."
  rm -r Shell1
  echo "Done"
fi

# unzip a fresh copy of Shell1/
echo "Unzipping Shell1.zip..."
unzip -q Shell1
echo "Done"

: ' Problem 1: In the space below, write commands to change into the
Shell1/ directory and print a string telling you the current working
directory. '



: ' Problem 2: Use ls with flags to print one list of the contents of
Shell1/, including hidden files and folders, listing contents in long
format, and sorting output by file size. '


: ' Problem 3: Inside the Shell1/ directory, delete the Audio/ folder
along with all its contents. Create Documents/, Photos/, and
Python/ directories. Rename the Random/ folder as Files/. '



: ' Problem 4: Using wildcards, move all the .jpg files to the Photos/
directory, all the .txt files to the Documents/ directory, and all the
.py files to the Python/ directory. '



: ' Problem 5: Move organize_photos.sh to Scripts/, add executable
permissions to the script, and run the script. '



: ' Problem 6: Copy img_649.jpg from UnixShell1/ to Shell1/Photos, making
sure to leave a copy of the file in UnixShell1/.'



# remove any old copies of UnixShell1.tar.gz
if [ ! -d Shell1 ];
then
  cd ..
fi

if [ -f UnixShell1.tar.gz ];
then
  echo "Removing old copies of UnixShell1.tar.gz..."
  rm -v UnixShell1.tar.gz
  echo "Done"
fi

# archive and compress the Shell1/ directory
echo "Compressing Shell1/ Directory..."
tar -zcpf UnixShell1.tar.gz Shell1/*
echo "Done"
