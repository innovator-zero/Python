import csv

filename='simulation.csv'
with open(filename) as f:
    reader=csv.reader(f)

    highs=[]
    for row in reader:
        highs.append(row[1])

    print(highs)