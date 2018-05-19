import math


def get_average(circles):
    avg, count = 0, 0
    for i in circles[0,:]:
        avg+= i[2]
        count+=1
    avg = avg/count
    return math.pi*avg*avg
