#!/bin/bash

# Check if a filename is provided as an argument
if [ -z "$1" ]; then
  echo "Usage: $0 <filename>"
  exit 1
fi

# Assign the filename to a variable
filename="$1"

# Check if the file exists
if [ ! -f "$filename" ]; then
  echo "File '$filename' does not exist."
  exit 1
fi

# Use sed to replace spaces with commas and save the result to a temporary file
sed 's/ /,/g' "$filename" > "${filename}.tmp"

# Replace the original file with the modified file
mv "${filename}.tmp" "$filename"

echo "Spaces replaced with commas in '$filename'."
