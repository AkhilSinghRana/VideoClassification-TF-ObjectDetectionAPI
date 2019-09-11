## Python Argument parser Script for Terminals
import argparse

def parseArguments():
    # Creates argument parser
    parser = argparse.ArgumentParser()


    # Define the basic Arguments for processing the data
    parser.add_argument('--input_dir', help='Path for input data/s', default='./', type=str)
    parser.add_argument('--output_dir', help='Path to save the procesed Data directory(DIR)', default='./', type=str)


    args = parser.parse_args()
    return args