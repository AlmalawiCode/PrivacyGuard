#!/bin/bash
# PrivacyGuard Run Script (Linux/Mac)
# Author: Abdulmohsen Almalawi <balmalowy@kau.edu.sa>

echo "========================================"
echo "         PrivacyGuard Launcher"
echo "========================================"
echo

# Check if Ant is installed
if ! command -v ant &> /dev/null; then
    echo "Error: Apache Ant is not installed."
    echo "Please install Ant: sudo apt install ant"
    exit 1
fi

# Check if Java is installed
if ! command -v java &> /dev/null; then
    echo "Error: Java is not installed."
    echo "Please install Java JDK 18+: sudo apt install openjdk-18-jdk"
    exit 1
fi

# Compile and run
echo "Building project..."
ant compile

if [ $? -eq 0 ]; then
    echo
    echo "Starting PrivacyGuard..."
    ant run
else
    echo "Build failed. Please check the error messages above."
    exit 1
fi
