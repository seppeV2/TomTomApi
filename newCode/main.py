from operational_functions import od_matrix_from_move_csv
from retreiveOD import get_od_matrix_from_database

import pathlib


def main():
    #get the od matrix from the database 
    odRightMatrix = get_od_matrix_from_database()

    #convert the csv file from tomtom move to a od matrix
    path = str(pathlib.Path(__file__).parent)+'/data/move_results/Leuven.csv'
    print(od_matrix_from_move_csv(path))

main()
    