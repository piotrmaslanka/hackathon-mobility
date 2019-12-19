# Heatmap

import json
import skimage.morphology
import typing as tp
import scipy.ndimage
import numpy as np
import numpy.ma as ma
from deap.base import Fitness
import itertools
from collections import namedtuple

Giveup = namedtuple('Giveup', ('month', 'point'))

DISCRETIZATION_INTERVAL = 0.05  # 100 meters


class Point:
    lon: float
    lat: float

    def __init__(self, lon, lat):
        self.lon = lon
        self.lat = lat

    def __eq__(self, other) -> bool:
        return self.lon == other.lon and self.lat == other.lat

    def __hash__(self) -> int:
        return self.lon.__hash__() ^ self.lat.__hash__()

    def get_kilometres(self) -> tp.Tuple[float, float]:
        """Return point representation as a point on a kilometer grid"""
        return 85.39 * self.lon, 111.03 * self.lat

    @staticmethod
    def from_km(km_lon: float, km_lat: float) -> 'Point':
        return Point(1 / 85.39 * km_lon, 1 / 111.03 * km_lat)


class Zone:
    canonical_type: str
    points: tp.List[Point]

    def __init__(self, canonical_type: str, points: tp.List[Point]):
        self.canonical_type = canonical_type
        self.points = points
        self.points_km = [point.get_kilometres() for point in points]

    def get_area(self) -> float:
        point_list = [point.get_kilometres() for point in self.points]
        points_natural_order = list(range(len(point_list)))
        points_shifted_one = points_natural_order[1:] + [points_natural_order[0]]

        area = 0.0
        for p_no, p_so in zip(points_natural_order, points_shifted_one):
            area += (point_list[p_no][0] * point_list[p_so][1] - point_list[p_no][1] * point_list[p_so][0]) / 2

        return abs(area)

    def is_inside_km(self, kilo_lon: float, kilo_lat: float) -> bool:
        n = len(self.points_km)
        inside = False

        p1x, p1y = self.points_km[0]
        for i in range(n + 1):
            p2x, p2y = self.points_km[i % n]
            if kilo_lat > min(p1y, p2y):
                if kilo_lat <= max(p1y, p2y):
                    if kilo_lon <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (kilo_lat - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or kilo_lon <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def is_inside(self, point: Point) -> bool:

        n = len(self.points)
        inside = False

        p1x, p1y = self.points[0]
        for i in range(n + 1):
            p2x, p2y = self.points[i % n]
            if point.lat > min(p1y, p2y):
                if point.lat <= max(p1y, p2y):
                    if point.lon <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (point.lat - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or point.lon <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside


class SquareGrid:
    def __init__(self, grid: np.array):
        self.grid = grid

    def is_inside_km(self, km_lon: float, km_lat: float) -> bool:
        min_point = MINIMUM_POINT.get_kilometres()
        km_lon, km_lat = km_lon - min_point[0], km_lat - min_point[1]
        km_lon /= DISCRETIZATION_INTERVAL
        km_lat /= DISCRETIZATION_INTERVAL
        try:
            return self.grid[int(km_lon), int(km_lat)]
        except IndexError:
            return False

    def get_total_percentage_giveups(self) -> float:
        return np.sum(ma.masked_array(GIVEUPS, mask=np.logical_not(self.grid))) / TOTAL_GIVEUPS

    def get_total_percentage_heatmaps(self) -> float:
        return np.sum(ma.masked_array(HEATMAP, mask=np.logical_not(self.grid))) / TOTAL_HEATMAP

    def get_disjointed_area_number(self) -> int:
        labels, num = skimage.morphology.label(self.grid, return_num=True)
        return num

    def get_fitness(self) -> float:
        a = self.get_total_percentage_giveups() * 2 + \
            self.get_total_percentage_heatmaps() * 4
        b = (2.42 - self.count_surface()) / 2.42
        if b < 0:
            a += b
        c = self.get_disjointed_area_number()
        if c > 6:
            a -= c * 0.2
        return a

    def count_surface(self) -> float:
        nbones = np.count_nonzero(self.grid)
        return nbones * DISCRETIZATION_INTERVAL ** 2

    @property
    def fitness(self):
        return creator.Fitness((self.get_fitness(),))


def get_giveups(scooter_type: str, particular_month: tp.Optional[int] = None) -> tp.Iterator[Giveup]:
    def parseLine(line):
        mon, rest = line.split(',')
        rest = rest[6:-2]
        lon, lat = rest.split(' ')
        lon, lat = float(lon), float(lat)
        return Giveup(int(mon), Point(lon=lon, lat=lat))

    with open(f'Locations{scooter_type}sOnRzeszow07-09.csv', 'r') as fin:
        for line in fin.readlines()[1:]:
            a = parseLine(line)
            if particular_month is not None:
                if a.month == particular_month:
                    yield a
            else:
                yield a


class Zone:
    canonical_type: str
    points: tp.List[Point]

    def __init__(self, canonical_type: str, points: tp.List[Point]):
        self.canonical_type = canonical_type
        self.points = points
        self.points_km = [point.get_kilometres() for point in points]

    def get_area(self) -> float:
        point_list = [point.get_kilometres() for point in self.points]
        points_natural_order = list(range(len(point_list)))
        points_shifted_one = points_natural_order[1:] + [points_natural_order[0]]

        area = 0.0
        for p_no, p_so in zip(points_natural_order, points_shifted_one):
            area += (point_list[p_no][0] * point_list[p_so][1] - point_list[p_no][1] * point_list[p_so][0]) / 2

        return abs(area)

    def is_inside_km(self, kilo_lon: float, kilo_lat: float) -> bool:
        n = len(self.points_km)
        inside = False

        p1x, p1y = self.points_km[0]
        for i in range(n + 1):
            p2x, p2y = self.points_km[i % n]
            if kilo_lat > min(p1y, p2y):
                if kilo_lat <= max(p1y, p2y):
                    if kilo_lon <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (kilo_lat - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or kilo_lon <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside


def as_points(lst: tp.List[float]) -> tp.Iterator[Point]:
    for lon, lat in lst:
        yield Point(lon=lon, lat=lat)


def get_zones() -> tp.Iterator[Zone]:
    with open('strefy.json', 'r') as fin:
        data = json.load(fin)

    for zone in data['data']['layers']['data']:
        for subzone in zone['area']['coordinates']:
            yield Zone(canonical_type=zone['canonical_type'],
                       points=list(as_points(subzone[0])))


GIVEUP = list(get_giveups('KickScooter'))

MINIMUM_POINT = Point(lat=min(giveup.point.lat for giveup in GIVEUP), lon=min(giveup.point.lon for giveup in GIVEUP))
MAXIMUM_POINT = Point(lat=max(giveup.point.lat for giveup in GIVEUP), lon=max(giveup.point.lon for giveup in GIVEUP))


def get_heatmap() -> tp.Iterator[Giveup]:
    def parseLine(line):
        lon, lat, wth = line.split(',')
        lon, lat = float(lon), float(lat)
        return Giveup(int(wth), Point(lon=lon, lat=lat))

    with open(f'Heat.csv', 'r') as fin:
        for line in fin.readlines()[1:]:
            yield parseLine(line)


HEAT = list(get_heatmap())


class DiscretizatorEngine:

    def get_grid_size(self) -> tp.Tuple[int, int]:
        min_kms = MINIMUM_POINT.get_kilometres()
        max_kms = MAXIMUM_POINT.get_kilometres()

        diff = max_kms[0] - min_kms[0], max_kms[1] - min_kms[1]
        return int(diff[0] / DISCRETIZATION_INTERVAL), \
               int(diff[1] / DISCRETIZATION_INTERVAL)

    def discretize(self, zone: Zone, existing_grid: tp.Optional[np.array] = None) -> SquareGrid:
        grid_size = self.get_grid_size()
        base_np = existing_grid or np.zeros(grid_size, dtype=np.bool)

        start_point_km = MINIMUM_POINT.get_kilometres()

        for point in zone.points:
            pt = point.get_kilometres()
            intx, inty = int((pt[0] - start_point_km[0]) / DISCRETIZATION_INTERVAL), \
                         int((pt[1] - start_point_km[1]) / DISCRETIZATION_INTERVAL)
            try:
                base_np[intx, inty] = True
            except IndexError:
                pass

        return SquareGrid(base_np)


import math

min_km = MINIMUM_POINT.get_kilometres()
max_km = MAXIMUM_POINT.get_kilometres()

scope = int((max_km[0] - min_km[0]) / DISCRETIZATION_INTERVAL), int((max_km[1] - min_km[1]) / DISCRETIZATION_INTERVAL)

HEATMAP = np.zeros(DiscretizatorEngine().get_grid_size(), dtype=np.int32)
for index, heat in enumerate(HEAT):
    pk = heat.point.get_kilometres()
    act_km = int((pk[0] - min_km[0]) / DISCRETIZATION_INTERVAL), int((pk[1] - min_km[1]) / DISCRETIZATION_INTERVAL)

    if (act_km[0] > scope[0]) or (act_km[1] > scope[1]):
        print(heat.point.lat, heat.point.lon)
        print(index, 'Point beyond scope of testing')
    else:
        try:
            HEATMAP[act_km[0], act_km[1]] += heat.month
        except IndexError:
            print(index, 'Point ', heat.point.lon, heat.point.lat, 'not seen on the grid')


def f(x):
    return x > 0


vf = np.vectorize(f)


def array_for(x):
    return np.array([f(xi) for xi in x])


TOTAL_HEATMAP = np.sum(HEATMAP)
from matplotlib import pyplot as plt
plt.imshow(HEATMAP)

GIVEUPS = np.zeros(DiscretizatorEngine().get_grid_size(), dtype=np.int32)
TOTAL_GIVEUPS = len(GIVEUP)
for index, giveup in enumerate(GIVEUP):
    min_km = MINIMUM_POINT.get_kilometres()
    pk = giveup.point.get_kilometres()
    min_km = int((pk[0]- min_km[0])/DISCRETIZATION_INTERVAL), int((pk[1]- min_km[1])/DISCRETIZATION_INTERVAL)
    try:
        GIVEUPS[min_km[0], min_km[1]] += 1
    except IndexError:
        print(index, 'Point ',giveup.point.lon,giveup.point.lat,'not seen on the grid')

from deap import creator, base, tools, algorithms
from numpy.random import random, seed, randint
from random import choice

seed(0)

creator.create("Fitness", base.Fitness, weights=(1.0,))
creator.create("Individual", SquareGrid, fitness=creator.Fitness)

toolbox = base.Toolbox()


def cxTwoPointCopy(sq1: SquareGrid, sq2: SquareGrid) -> SquareGrid:
    sq1 = sq1.grid.copy()
    sq2 = sq2.grid.copy()
    size = DiscretizatorEngine().get_grid_size()
    st = np.zeros(size)

    st[randint(size[0]), randint(size[1])] = True
    st = scipy.ndimage.morphology.binary_dilation(st)
    st = scipy.ndimage.morphology.binary_dilation(st)
    st = scipy.ndimage.morphology.binary_dilation(st)
    st = scipy.ndimage.morphology.binary_dilation(st)
    st = scipy.ndimage.morphology.binary_dilation(st)

    sq1 = (np.logical_not(sq1) & sq2) | (sq1 & np.logical_not(st)) | (sq1 & sq2)
    sq2 = (st & sq1) | (sq1 & sq2) | (np.logical_not(st) & sq2)

    return SquareGrid(sq1), SquareGrid(sq2)


def evaluate(sq: SquareGrid) -> float:
    return sq.get_fitness()


def mutate(sq: SquareGrid) -> tp.Tuple[SquareGrid]:
    sq = sq.grid.copy()

    size = DiscretizatorEngine().get_grid_size()
    st = np.zeros(size)

    p = scipy.ndimage.morphology.binary_dilation(scipy.ndimage.morphology.binary_dilation(sq)) & np.logical_not(sq)
    x, y = choice(np.argwhere(p == True))

    st[x, y] = True
    st = scipy.ndimage.morphology.binary_dilation(st)
    st = scipy.ndimage.morphology.binary_dilation(st)
    st = scipy.ndimage.morphology.binary_dilation(st)
    st = scipy.ndimage.morphology.binary_dilation(st)
    st = scipy.ndimage.morphology.binary_dilation(st)

    if random() > 0.9:
        sq = sq | st
    else:
        sq = sq & np.logical_not(st)

    return SquareGrid(sq),


size = DiscretizatorEngine().get_grid_size()
st = np.zeros(size, dtype=np.bool)
for grid in [DiscretizatorEngine().discretize(zone) for zone in get_zones() if zone.canonical_type == 'roller']:
    st |= grid.grid
st = scipy.ndimage.morphology.binary_dilation(st)


def createIndividual() -> SquareGrid:
    global st
    sd = SquareGrid(st.copy())
    for x in range(10):
        sd, = mutate(sd)
    return sd


toolbox.register("individual", createIndividual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", cxTwoPointCopy)
toolbox.register("evaluate", evaluate)
toolbox.register("mutate", mutate)
toolbox.register("select", tools.selTournament, tournsize=5)


def main():
    NGEN = 200  # 20   # parameter: number of epochs
    MU = 400  # parameter: population size
    LAMBDA = 100
    CXPB = 0.5  # parameter: fraction of new population that will be construed by cross-overing current best
    MUTPB = 0.5  # parameter: fraction of new population that will be construed by mutating current best ones

    pop = toolbox.population(n=MU)
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.get_fitness())

    algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats,
                              halloffame=hof, verbose=True)

    return pop, stats, hof


pop, stats, hof = main()

from matplotlib import pyplot as plt

plt.imshow(pop[0].grid)
print('Total giveup coverage % :', pop[0].get_total_percentage_giveups())
print('Total heatmap coverage % :', pop[0].get_total_percentage_heatmaps())
print('Disjointed area number : ', pop[0].get_disjointed_area_number())
print('Surface area : ', pop[0].count_surface())

stdOffset = MINIMUM_POINT

outfile = open("/output/dataPts.csv", "w")

for i in range(len(pop[0].grid[0])):
    try:
        for j in range(len(pop[0].grid[i])):
            if pop[0].grid[i][j]:
                pt = Point.from_km(float(i*DISCRETIZATION_INTERVAL), float(j*DISCRETIZATION_INTERVAL))
                print(str(pt.lat+stdOffset.lat)+','+str(pt.lon+stdOffset.lon), file=outfile)
    except IndexError:
        pass

outfile.close()