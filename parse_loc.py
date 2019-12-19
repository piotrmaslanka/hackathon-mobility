import logging
import typing as tp
from collections import namedtuple

Giveup = namedtuple('Giveup', ('Month', 'Lat', 'Lon'))

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    with open('LocationsScootersOnRzeszow07-09.csv', 'r') as fin:
        data = fin.readlines()[1:]

    def parseLine(line):
        mon, rest = line.split(',')
        rest = rest[6:-2]
        lon, lat = rest.split(' ')
        lon, lat = float(lon), float(lat)
        return Giveup(int(mon), lat, lon)

    new_data = []
    for line in data:
        new_data.append(parseLine(line))
