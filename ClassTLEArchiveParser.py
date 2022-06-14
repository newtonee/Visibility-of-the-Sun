import pandas as pd
import pickle as pkl
import numpy as np
from os import listdir
from os.path import isfile, join
from skyfield.api import load


class TLEArchiveParser():
    def __init__(self, foldername, filename):
        self.foldername = foldername
        self.filename = filename

    def __call__(self):
        self.parse()
        tles = self.get_dict()
        return tles

    def parse(self, path=None):
        path = self.filename if path is None else path
        lines_per_file = 2
        smallfile = None
        with open(path) as bigfile:
            for lineno, line in enumerate(bigfile):
                if lineno % lines_per_file == 0:
                    if smallfile:
                        smallfile.close()
                    small_filename = (self.foldername + "/" + 'small_file_{}.txt').format(lineno + lines_per_file)
                    smallfile = open(small_filename, "w")
                smallfile.write(line)
            if smallfile:
                smallfile.close()

    def get_dict(self, sat_id=48274):
        onlyfiles = [f for f in listdir(self.foldername) if isfile(join(self.foldername, f))]
        # ISS TLEs dict forming
        tles = {}
        for tle_filename in onlyfiles:
            tle_filename = self.foldername + "/" + tle_filename
            satellites = load.tle(tle_filename)
            tles.update({satellites[sat_id].epoch.utc_datetime(): satellites[sat_id]})

        # sorting ISS TLEs dict by its key - date
        dates = list(tles.keys())
        dates.sort()
        tles_unsorted = tles
        tles = {}
        for date in dates:
            tles.update({date: tles_unsorted[date]})

        return tles


if __name__ == "__main__":
    parser = TLEArchiveParser("tianhe_tle_archive_parsed", "Tianhe_TLE.txt")
    parser()




