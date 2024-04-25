#!/bin/bash
#
# Define the name of the environment and the Python version
#
ENV_NAME="dmsuds"
PYTHON_VERSION="3.12"
#
# Check if the conda command is available
#
if ! command -v conda &> /dev/null
then
    echo "conda could not be found. Please ensure that it is installed and added to your PATH."
    exit
fi
#
# Create the Conda environment with a specific Python version
#
echo "Creating Conda environment: $ENV_NAME with Python $PYTHON_VERSION"
conda create --name $ENV_NAME python=$PYTHON_VERSION -y
#
# Activate the environment
#
echo "Activating the environment: $ENV_NAME"
source activate $ENV_NAME
#
# Check if the requirements.txt file exists
#
if [ -f "requirements.txt" ]; then
    # Install packages from requirements.txt using pip
    echo "Installing packages from requirements.txt"
    pip install -r requirements.txt
else
    echo "requirements.txt not found."
fi

echo "Setup completed successfully."
#
# folder for diffusion models
#
cd models
mkdir diffusion_models
cd ..