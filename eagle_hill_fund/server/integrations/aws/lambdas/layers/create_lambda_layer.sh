#!/bin/bash

LIBRARY_NAME=$1

# Install the required packages in a temporary directory
mkdir -p temp/python
pip3 install --target temp/python $LIBRARY_NAME

# Create the ZIP file for the Lambda layer
cd temp || exit
zip -r ../$LIBRARY_NAME-layer.zip *

# Clean up
cd ..
rm -rf temp




