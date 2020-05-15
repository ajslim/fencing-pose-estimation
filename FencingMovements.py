import math

def angleBetween(point1, point2):
    return math.atan2((-point2[1] - -point1[1]),(point2[0] - point1[0]))

def within(value1, value2, range):
    return value1 + range > value2 and value1 - range < value2

def detectLunge(fencerPosition):
    try:
        return angleBetween(fencerPosition[12], fencerPosition[13]) > -1
    except IndexError:
        return False

