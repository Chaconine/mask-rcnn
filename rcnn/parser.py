import csv


def parse():
    path = "/Users/josephgmaa/mask-rcnn/mask-rcnn/data/white18_trainingData.txt"
    target = {}
    with open(path, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        next(reader)
        for row in reader:
            row = list(filter("".__ne__, row))
            filename = path[:-4] + "/" + row[0].split("\\")[-1]
            target[filename] = {"boxes": [[row[4*i], row[1+4*i], row[2+4*i], row[3+4*i]] for i in range(len(row)//4)]}
    return target
