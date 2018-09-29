import csv

def get_parameter(m = 8, n = 8):
    # get data_list from raw exp data

    ifile = open(file_name.csv, "r")
    # reader is a list [[row1],[row2],...]
    reader = csv.reader(ifile)

    point0_I = []
    point0_Q = []
    point1_I = []
    point1_Q = []
    for idx, row in enumerate(reader):
        if int(row[2]) == 1:
            point1_I.append(int(row[0]))
            point1_Q.append(int(row[1]))
        elif int(row[2]) == 0:
            point0_I.append(int(row[0]))
            point0_Q.append(int(row[1]))
        else:
            print('pro_data is wrong,neither 0 nor 1')

    return point0_I, point0_Q, point1_I, point1_Q
