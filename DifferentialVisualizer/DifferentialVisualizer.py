import cv2
import configparser
import glob

config = configparser.ConfigParser()
config.read('config.txt')
source_folder = config['DEFAULT']['SourceFolder']
ground_truth_naming_schema = config['DEFAULT']['GroundTruthNamingSchema']

def process_folder(folder:str):
    
    for file in glob.glob('{}/*{}*.png'.format(folder, ground_truth_naming_schema)):
        print(file)


def main():



    for folder in glob.glob('{}/*'.format(source_folder)):
        process_folder(folder)


if __name__ == '__main__':
    main()