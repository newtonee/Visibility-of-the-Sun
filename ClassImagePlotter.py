import warnings

# calculating and data containing
import numpy as np
from numpy import linalg as LA
import pandas as pd
import swifter

# time
from datetime import timezone, timedelta, datetime

# celestial bodies motion calculating
import skyfield
from skyfield.api import Topos, load
from skyfield.positionlib import Geocentric
from skyfield.nutationlib import iau2000b

# plotting
import os
# os.environ['PROJ_LIB'] = r'C:\Users\Egor\AppData\Local\Continuum\anaconda2\pkgs\proj4-4.9.2-vc10_0\Library\share'
# %matplotlib inline
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.basemap import Basemap
import matplotlib.cm as cm
import matplotlib.tri as tri
import seaborn as sns

# multiprocessing
# from dask import dataframe as dd
# from multiprocessing import cpu_count, Pool
# import multiprocessing as mp
# from pandarallel import pandarallel
# pandarallel.initialize()
from ClassTLEArchiveParser import TLEArchiveParser
from IPython.display import clear_output


# matplotlib settings
import matplotlib.pyplot as plt

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 20

plt.rc('font', **{'family' : 'normal',
                'weight' : 'bold',
                'size'   : SMALL_SIZE})          # controls default text sizes
plt.rc("axes", **{'labelsize'   : MEDIUM_SIZE,
                  'titlesize'   : SMALL_SIZE,
                 "labelweight" : "bold"})

plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# constants
R_earth = 6371000  # m
R_sun = 0  # 1.392E9/2 #m

# visibility indexes
visible = {'false': 0, 'true': 1, 'up': 2, 'down': 3}

class image_plotter:
    def __init__(self, sat, sun, earth, dt=10, T=[3, 0, 0, 0], h=[20, 80], start_date=[2019, 12, 20, 0, 0, 0], GMT=0,
                 sat_subpoint=False, full_sun=False):
        self.sat = sat  # EarthSatellite or {datetime : EarthSatellite}
        self.sun = sun
        self.earth = earth
        self.dt = dt  # time step, [seconds]
        self.T = T  # time period condsidered, [days, hours, minutes, seconds]
        self.h = [h[0] * 1E3, h[1] * 1E3]  # heights of atmospheric layer considered, initially in [km], from now in [m]
        self.start_date = start_date  # [year, month, day, hour, minute, second]
        self.GMT = GMT  # [hours]
        self.sat_subpoint = sat_subpoint  # Does the position of the spacecraft display during the flyby?
        self.full_sun = full_sun  # Only consider the full visibility of the sun or partial too?

        self.ts = load.timescale()
        dates = self.get_dates()
        self.df = pd.DataFrame({'Date': dates})  # vector of the dates to calculate positions of sat,
        # sun and sun's visibility

    def __call__(self, terminator=True, sat_trajectory=True, save=False, projection='cyl', scatter=True,
                 num_points=40,
                 plot_flyby_info=False):
        # operating pipeline
        self.get_positions()
        self.calculate_visibility()
        self.get_flyby_info(plot=plot_flyby_info)
        self.get_beam_path(num_points=num_points)

    #         self.plot(terminator=terminator, sat_trajectory=sat_trajectory, save=save, projection=projection, scatter=scatter)

    def decode_start_date(self):
        Y = self.start_date[0]
        M = self.start_date[1]
        D = self.start_date[2]
        h = self.start_date[3]
        m = self.start_date[4]
        s = self.start_date[5]
        return Y, M, D, h, m, s

    def decode_period(self):
        dD = self.T[0]
        dh = self.T[1]
        dm = self.T[2]
        ds = self.T[3]
        return dD, dh, dm, ds

    def get_dates(self):
        # start date
        Y, M, D, h, m, s = self.decode_start_date()
        # period considered
        dD, dh, dm, ds = self.decode_period()

        # getting the vector of dates
        ts = load.timescale()  # create skyfield timescale object
        tz = timezone(timedelta(hours=self.GMT))  # whatever your timezone offset from UTC is
        start = datetime(Y, M, D, h, m, s, tzinfo=tz)  # timezone-aware start time
        end = start + timedelta(seconds=ds, minutes=dm, hours=dh, days=dD)  # one day's worth of times
        delta = timedelta(seconds=self.dt)  # your interval over which you evaluate
        dates = [start]
        now = start
        while now <= end:
            now += delta
            dates.append(now)
        return dates

    def get_sun_position(self, row):  # get position of the Sun
        t = row["Date"]
        t = self.ts.utc(t)
        t._nutation_angles = iau2000b(t.tt)  # reducing accuracy of Sun's position determ. to improve speed
        pos = self.earth.at(t).observe(self.sun).apparent().position.m
        return pos

    def get_sat_position_and_subpoint(self, row):  # get position of sat
        # and its current latitude, longitude and elevation
        t = self.ts.utc(row["Date"])
        geocentric = row["sat"].at(t)
        row["sat_pos"] = geocentric.position.m
        subpoint = geocentric.subpoint()
        row["sat_subpoint"] = np.array([self.convert_angle(subpoint.latitude),
                                        self.convert_angle(subpoint.longitude), subpoint.elevation.m])
        return row

    def get_sat_position(self, row):  # get position of sat
        t = self.ts.utc(row["Date"])
        geocentric = row["sat"].at(t)
        pos = geocentric.position.m
        return pos

    def get_positions(self):
        self.df["closest TLE date"] = None
        self.df["sat"] = None

        self.df = self.df.swifter.apply(self.get_sat_for_date, axis=1)
        if self.sat_subpoint:
            self.df = self.df.swifter.apply(self.get_sat_position_and_subpoint, axis=1)
        else:
            self.df["sat_pos"] = self.df.swifter.apply(self.get_sat_position, axis=1)
        self.df["sun_pos"] = self.df.swifter.apply(self.get_sun_position, axis=1)
        self.df["sat_to_sun"] = self.df["sun_pos"] - self.df["sat_pos"]  # vector between the sat and the Sun,
        # pointing to the Sun
        self.df = self.df.swifter.apply(self.get_WGS84_R_Earth, axis=1)

    def get_nearest_datetime(self, items, pivot):
        nearest_datetime = min(items, key=lambda x: abs(x - pivot))
        return nearest_datetime

    def get_sat_for_date(self, row):
        t = row["Date"]
        t = pd.to_datetime(t)# t.to_pydatetime()
        if type(self.sat) == dict:
            row["closest TLE date"] = self.get_nearest_datetime(self.sat.keys(), t)
            row["sat"] = self.sat[row["closest TLE date"]]
        elif type(self.sat) == skyfield.sgp4lib.EarthSatellite:
            row["closest TLE date"] = self.sat.epoch.utc_datetime()
            row["sat"] = self.sat
        else:
            raise TypeError('Incorrect type of self.sat')
        return row

    def calculate_visibility(self):  # compute if sun is visible for whole dataframe
        print("Shape of dataframe before determining if the Sun is visible is", self.df.shape)
        self.df = self.df.swifter.apply(self.get_horizont_earth_radius, axis=1)
        self.df["sun_vis"] = self.df.swifter.apply(self.sun_is_visible, axis=1)
        if self.full_sun:
            self.df = self.df[self.df["sun_vis"] == 1]
        else:
            self.df = self.df[self.df["sun_vis"] != 0]
        self.df = self.df.reset_index(drop=True)
        print("Shape of dataframe after determining if the Sun is visible is", self.df.shape)

    def get_horizont_earth_radius(self, row):
        r_sat = row["sat_pos"]
        r_ss = row["sat_to_sun"]
        r_ss_norm = LA.norm(r_ss)
        perpendicular_to_ss = r_sat - np.dot(r_ss / r_ss_norm, r_sat) * r_ss / r_ss_norm
        point = Geocentric(position_au=perpendicular_to_ss / 149597870691,
                           t=self.ts.utc(row["Date"]))  # высота рассчитывается из WGS 84
        subpoint = point.subpoint()
        row["horizont_R_earth"] = LA.norm(perpendicular_to_ss) - subpoint.elevation.m
        return row

    def sun_is_visible(self, row):  # compute if the sun is visible for current date
        # radius-vectors of sat and sun
        r_sat = row["sat_pos"]
        r_sun = row["sun_pos"]
        r_ss = row["sat_to_sun"]
        # getting norms
        r_sat = LA.norm(r_sat)
        r_sun = LA.norm(r_sun)
        r_ss = LA.norm(r_ss)
        # angles of visibility
        rho_up = np.arcsin((row["horizont_R_earth"] + self.h[1]) / r_sat)  # R_earth acceptable
        rho_down = np.arcsin((row["horizont_R_earth"] + self.h[0]) / r_sat)  # R_earth acceptable
        rho_sun = np.arcsin(R_sun / r_ss)
        d = np.arccos((r_sat ** 2 + r_ss ** 2 - r_sun ** 2) / (2 * r_sat * r_ss))
        # determing if sun is visible
        if rho_up >= d + rho_sun and rho_down <= d - rho_sun:
            return visible["true"]
        #         if rho_up >= d - rho_sun and rho_down <= d + rho_sun : return visible["true"]
        elif rho_up >= d - rho_sun and rho_down <= d - rho_sun:
            return visible["down"]
        elif rho_up >= d + rho_sun and rho_down <= d + rho_sun:
            return visible["up"]
        else:
            return visible["false"]

    def get_WGS84_R_Earth(self, row):
        sat_vec = row["sat_pos"]
        lat, lon, elev = row["sat_subpoint"]
        row["WGS_84_R_Earth"] = LA.norm(sat_vec) - elev
        #         row["WGS_84_R_Earth"] = R_earth
        return row

    def get_entry_vec(self, row):  # Light beam entry position
        b1, b2, b3 = row["sat_to_sun"]
        c1, c2, c3 = row["sat_pos"]
        local_R_earth = row["WGS_84_R_Earth"]  # constant R_Earth unacceptable. use WGS 84
        R = local_R_earth + self.h[1]
        # from Mathematica
        a1 = (-2 * b1 ** 2 * c1 - 2 * b1 * b2 * c2 - 2 * b1 * b3 * c3 - np.sqrt(
            (2 * b1 ** 2 * c1 + 2 * b1 * b2 * c2 + 2 * b1 * b3 * c3) ** 2 - 4 * (b1 ** 2 + b2 ** 2 + b3 ** 2) * (
                        b1 ** 2 * c1 ** 2 + b1 ** 2 * c2 ** 2 + b1 ** 2 * c3 ** 2 - b1 ** 2 * R ** 2))) / (
                         2. * (b1 ** 2 + b2 ** 2 + b3 ** 2))
        a2 = (-((b1 ** 2 * b2 * c1) / (b1 ** 2 + b2 ** 2 + b3 ** 2)) - (b1 * b2 ** 2 * c2) / (
                    b1 ** 2 + b2 ** 2 + b3 ** 2) - (b1 * b2 * b3 * c3) / (b1 ** 2 + b2 ** 2 + b3 ** 2) - (b2 * np.sqrt(
            (2 * b1 ** 2 * c1 + 2 * b1 * b2 * c2 + 2 * b1 * b3 * c3) ** 2 - 4 * (b1 ** 2 + b2 ** 2 + b3 ** 2) * (
                        b1 ** 2 * c1 ** 2 + b1 ** 2 * c2 ** 2 + b1 ** 2 * c3 ** 2 - b1 ** 2 * R ** 2))) / (
                          2. * (b1 ** 2 + b2 ** 2 + b3 ** 2))) / b1
        a3 = (-((b1 ** 2 * b3 * c1) / (b1 ** 2 + b2 ** 2 + b3 ** 2)) - (b1 * b2 * b3 * c2) / (
                    b1 ** 2 + b2 ** 2 + b3 ** 2) - (b1 * b3 ** 2 * c3) / (b1 ** 2 + b2 ** 2 + b3 ** 2) - (b3 * np.sqrt(
            (2 * b1 ** 2 * c1 + 2 * b1 * b2 * c2 + 2 * b1 * b3 * c3) ** 2 - 4 * (b1 ** 2 + b2 ** 2 + b3 ** 2) * (
                        b1 ** 2 * c1 ** 2 + b1 ** 2 * c2 ** 2 + b1 ** 2 * c3 ** 2 - b1 ** 2 * R ** 2))) / (
                          2. * (b1 ** 2 + b2 ** 2 + b3 ** 2))) / b1
        if np.isnan(a1) or np.isnan(a2) or np.isnan(a3):
            return np.nan
        else:
            return np.array([a1, a2, a3]) + row["sat_pos"]

    def get_exit_vec(self, row):  # Light beam exit position
        b1, b2, b3 = row["sat_to_sun"]
        c1, c2, c3 = row["sat_pos"]
        local_R_earth = row["WGS_84_R_Earth"]  # constant R_Earth unacceptable. use WGS 84
        R = local_R_earth + self.h[1]
        # from Mathematica
        a1 = (-2 * b1 ** 2 * c1 - 2 * b1 * b2 * c2 - 2 * b1 * b3 * c3 + np.sqrt(
            (2 * b1 ** 2 * c1 + 2 * b1 * b2 * c2 + 2 * b1 * b3 * c3) ** 2 - 4 * (b1 ** 2 + b2 ** 2 + b3 ** 2) * (
                        b1 ** 2 * c1 ** 2 + b1 ** 2 * c2 ** 2 + b1 ** 2 * c3 ** 2 - b1 ** 2 * R ** 2))) / (
                         2. * (b1 ** 2 + b2 ** 2 + b3 ** 2))
        a2 = (-((b1 ** 2 * b2 * c1) / (b1 ** 2 + b2 ** 2 + b3 ** 2)) - (b1 * b2 ** 2 * c2) / (
                    b1 ** 2 + b2 ** 2 + b3 ** 2) - (b1 * b2 * b3 * c3) / (b1 ** 2 + b2 ** 2 + b3 ** 2) + (b2 * np.sqrt(
            (2 * b1 ** 2 * c1 + 2 * b1 * b2 * c2 + 2 * b1 * b3 * c3) ** 2 - 4 * (b1 ** 2 + b2 ** 2 + b3 ** 2) * (
                        b1 ** 2 * c1 ** 2 + b1 ** 2 * c2 ** 2 + b1 ** 2 * c3 ** 2 - b1 ** 2 * R ** 2))) / (
                          2. * (b1 ** 2 + b2 ** 2 + b3 ** 2))) / b1
        a3 = (-((b1 ** 2 * b3 * c1) / (b1 ** 2 + b2 ** 2 + b3 ** 2)) - (b1 * b2 * b3 * c2) / (
                    b1 ** 2 + b2 ** 2 + b3 ** 2) - (b1 * b3 ** 2 * c3) / (b1 ** 2 + b2 ** 2 + b3 ** 2) + (b3 * np.sqrt(
            (2 * b1 ** 2 * c1 + 2 * b1 * b2 * c2 + 2 * b1 * b3 * c3) ** 2 - 4 * (b1 ** 2 + b2 ** 2 + b3 ** 2) * (
                        b1 ** 2 * c1 ** 2 + b1 ** 2 * c2 ** 2 + b1 ** 2 * c3 ** 2 - b1 ** 2 * R ** 2))) / (
                          2. * (b1 ** 2 + b2 ** 2 + b3 ** 2))) / b1
        if np.isnan(a1) or np.isnan(a2) or np.isnan(a3):
            return np.nan
        else:
            return np.array([a1, a2, a3]) + row["sat_pos"]

    def convert_angle(self, angle):  # converts angle value from int(deg), int(min), int(sec) to float(deg)
        angle = angle.dms(warn=True)
        return angle[0] + angle[1] / 60 + angle[2] / 3600

    def beam_path_points_with_subpoints(self, row):
        num_points = row["num_points"]  # - 1
        points_skyfield = []
        points = []
        ponts_direct_coords = []
        for i in range(num_points):
            point = row["entry_vec"] + row["diff"] * i
            ponts_direct_coords.append(point)
            point = Geocentric(position_au=point / 149597870691,
                               t=self.ts.utc(row["Date"]))  # высота рассчитывается из WGS 84
            subpoint = point.subpoint()
            points_skyfield.append([subpoint.latitude, subpoint.longitude, subpoint.elevation.m])
            points.append(
                [self.convert_angle(subpoint.latitude), self.convert_angle(subpoint.longitude), subpoint.elevation.m])

        row["beam_path_points_with_subpoints"] = np.array(points_skyfield)
        row["beam_path"] = np.array(points)
        row["beam_path_points_direct_coords"] = np.array(ponts_direct_coords)
        return row

    def get_beam_path(self, num_points=50):  # generate point lying on beam path
        # get vectors, pointing to the entry to atm and to the exit from atm of light beam
        self.df["entry_vec"] = self.df.swifter.apply(self.get_entry_vec, axis=1)
        self.df["exit_vec"] = self.df.swifter.apply(self.get_exit_vec, axis=1)
        self.df = self.df.dropna()

        # get points forming light trajectory
        self.df["diff"] = self.df["exit_vec"] - self.df[
            "entry_vec"]  # оба имеют макс. высоту в h[1]. При этом на их же концах и есть наибольшая ошибка
        self.df["num_points"] = num_points
        self.df["diff"] /= self.df["num_points"]
        self.df = self.df.swifter.apply(self.beam_path_points_with_subpoints, axis=1)

        # clean data
        self.df.drop(columns=["diff"])
        self.df = self.df.reset_index(drop=True)

    def total_seconds(self, t):
        return t.total_seconds()

    def get_flyby_info(self, plot=True, lookup=True):  # get information about flyby: duration [sec], start and end date
        self.df["dt"] = self.df["Date"].diff().apply(self.total_seconds)
        Duration = []
        Start = []
        End = []
        for index, row in self.df.iterrows():
            #             print(row)
            dt = row["dt"]
            date = row["Date"]
            if np.isnan(dt):
                Duration.append(0)
                Start.append(date)
                End.append(date)
            elif dt <= 2 * self.dt:
                Duration[-1] += dt
                if End[-1] < date: End[-1] = date
            else:
                Duration.append(0)
                Start.append(date)
                End.append(date)

        assert len(Duration) == len(Start) and len(Duration) == len(End), print(
            "Sizes of the lists must be equal! Duration", len(Duration), "Start", len(Start), "End", len(End))
        flyby_info = pd.DataFrame({"Duration": Duration, "Start": Start, "End": End})
        self.flyby_info = flyby_info

        if lookup:
            flyby_info.head()
            print("Shape is ", flyby_info.shape)

        if plot:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(12, 6)
            fig.suptitle('Flyby duration plots')
            ax1.plot(flyby_info["Duration"])
            ax1.set_xlabel("Number of flyby")
            ax1.set_ylabel("Duration of flyby, sec")

            ax2 = sns.distplot(flyby_info["Duration"])
            ax2.set_ylabel("Fraction")
            ax2.set_xlabel("Duration of flyby, sec")

            plt.show()

        return flyby_info

    def plot(self, terminator=False, sat_trajectory=False, save=False, projection='cyl',
             scatter=False, min_height=True, figsize=(12, 6)):  # get final map
        # start date
        Y, M, D, h, m, s = self.decode_start_date()
        # period considered
        dD, dh, dm, ds = self.decode_period()

        # datetimes
        tz = timezone(timedelta(hours=self.GMT))  # whatever your timezone offset from UTC is
        date = datetime(Y, M, D, h, m, s, tzinfo=tz)  # timezone-aware start time
        ddate = timedelta(days=dD, hours=dh, minutes=dm, seconds=ds)
        date_mid = date + ddate / 2
        print("Start date is", date)
        print("Middle date is", date_mid)
        print("Final date is", date + ddate)

        # prepare data for plotting
        # beam path points
        #         ar = np.array(self.df["beam_path"][0])
        ar = np.array(self.df["beam_path"].loc[self.df["beam_path"].index[0]])
        for i in self.df["beam_path"]:
            if min_height: i[:, 2] = i[:, 2].min()
            ar = np.concatenate((ar, i), axis=0)
        #         ar = ar[np.logical_and(ar[:,2] > self.h[0]*0.98, ar[:,2] < self.h[1]*1.02)]
        #         ar = ar[ar[:,2] < self.h[1]*1.02]

        # satellite trajectory points
        if sat_trajectory:
            #             sat = np.array(self.df["sat_subpoint"][0])
            sat = np.array(self.df["sat_subpoint"].loc[self.df["sat_subpoint"].index[0]])
            sat = sat[np.newaxis]
            for i in self.df["sat_subpoint"]:
                i = i[np.newaxis]
                sat = np.concatenate((sat, i), axis=0)

        # plotting
        # map
        #         fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        fig = plt.figure(figsize=figsize, edgecolor='w')
        ax = fig.add_axes([0, 0, 1, 1])
        mp = Basemap(projection=projection, lat_0=0, lon_0=0, ax=ax)

        # relief, water and etc.
        mp.drawcoastlines(color='gray', zorder=0)
        mp.shadedrelief(zorder=0)

        # plotting terminator
        if terminator:
            term_speed = 360. / (24 * 3600)
            total_sec = dD * 24 * 3600 + dh * 3600 + dm * 60 + ds
            print("During the considered time period, the terminator shifts by", term_speed * total_sec, "degrees.")
            print("Position of the terminator for the middle of the period is shown.")
            CS = mp.nightshade(date_mid, zorder=1)

        # plotting atmospheric cut
        lons = ar[:, 1]
        lats = ar[:, 0]
        elev = ar[:, 2]
        print("min elev of the beam =", min(elev), "m , max elev of the beam =", max(elev), "m")
        x, y = mp(lons, lats)
        if scatter:
            lol = mp.scatter(x, y, marker=".", c=elev / 1000, s=2)  # cmap="plasma",
        else:
            lol = ax.tricontourf(x, y, elev / 1000, levels=15)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.2)
        cbar = fig.colorbar(lol, cax=cax)
        cbar.set_label('Beam elevation, km', rotation=90)

        # plotting sattelite trajectory
        if sat_trajectory:
            lats = sat[:, 0]
            lons = sat[:, 1]
            elev = sat[:, 2]
            x, y = mp(lons, lats)
            lol = mp.scatter(x, y, marker=".", c='red', s=4)

        # saving
        if save:
            plt.savefig('map.png', quality=100, dpi=300, bbox_inches='tight')
            print("The image has been saved.")

        # showing
        # plt.tight_layout()
        plt.show()

    def plot_heatmap(self, bins=30, alpha=0.33, save=False,
                     projection='cyl', relief=True, figsize=(12, 6),
                     coastlines_color="black", coastlines_linewidth=2):  # get final map
        # start date
        Y, M, D, h, m, s = self.decode_start_date()
        # period considered
        dD, dh, dm, ds = self.decode_period()

        # datetimes
        tz = timezone(timedelta(hours=self.GMT))  # whatever your timezone offset from UTC is
        date = datetime(Y, M, D, h, m, s, tzinfo=tz)  # timezone-aware start time
        ddate = timedelta(days=dD, hours=dh, minutes=dm, seconds=ds)
        date_mid = date + ddate / 2
        print("Start date is", date)
        print("Middle date is", date_mid)
        print("Final date is", date + ddate)

        # prepare data for plotting
        # beam path points
        #         ar = np.array(self.df["beam_path"][0])
        #         for i in self.df["beam_path"]:
        #             ar = np.concatenate((ar, i), axis=0)
        #         ar = ar[np.logical_and(ar[:,2] > self.h[0]*0.98, ar[:,2] < self.h[1]*1.02)]
        #         ar = ar[ar[:,2] < self.h[1]*1.02]

        # plotting
        # map
        #         fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        fig = plt.figure(figsize=figsize, edgecolor='w')
        ax = fig.add_axes([0, 0, 1, 1])
        mp = Basemap(projection=projection, lat_0=0, lon_0=0, ax=ax)

        # relief, water and etc.
        mp.drawcoastlines(color=coastlines_color, zorder=11, linewidth=coastlines_linewidth)
        #         if relief : mp.shadedrelief(zorder=zorder)
        #         mp.drawmapboundary(fill_color='skyblue')
        #         mp.fillcontinents(color='white', lake_color='skyblue', zorder=1)

        # creating array of points
        list_of_points = self.df["beam_path"].to_list()
        ar = np.concatenate(list_of_points, axis=0)
        lons = ar[:, 1]
        lats = ar[:, 0]
        #         elev = ar[:, 2]
        x, y = mp(lons, lats)

        lons_back, lats_back = np.meshgrid(np.arange(-180, 181, 1), np.arange(-90, 90.5, 0.5), sparse=False)
        lons_back = np.ravel(lons_back)
        lats_back = np.ravel(lats_back)

        # plotting hexbin
        CS = mp.hexbin(*mp(lons_back, lats_back), gridsize=bins, edgecolors='none', alpha=alpha, linewidths=None,
                       zorder=9, cmap=plt.cm.jet, C=x * 0.0, )  # plt.cm.jet, )
        CS = mp.hexbin(x, y, gridsize=bins, edgecolors='none', alpha=alpha, linewidths=None, zorder=10,
                       cmap=plt.cm.jet)  # plt.cm.jet, )

        #         # plotting hist2d
        #         # remove points outside projection limb.
        #         bincount, xedges, yedges = np.histogram2d(x, y, bins=bins)
        #         mask = bincount == 0
        #         # reset zero values to one to avoid divide-by-zero
        #         bincount = np.where(bincount == 0, 1, bincount)
        #         H, xedges, yedges = np.histogram2d(x, y, bins=bins)#, weights=z)
        #         H = np.ma.masked_where(mask, H/bincount)
        #         # set color of masked values to axes background (hexbin does this by default)
        #         palette = plt.cm.jet
        # #         palette.set_bad(ax.get_axis_bgcolor(), 1.0)
        #         CS = mp.pcolormesh(xedges,yedges,H.T,shading='flat',cmap=palette, zorder=10)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.2)
        #         cbar = fig.colorbar(CS * self.dt, cax=cax, cmap=plt.cm.jet, )
        cbar = fig.colorbar(CS, cax=cax, cmap=plt.cm.jet, )
        #         cbar.set_label('Exposition, sec', rotation=90)
        cbar.set_label('Exposition, sec', rotation=90)

        # saving
        if save:
            plt.savefig('map.png', quality=100, dpi=300, bbox_inches='tight')
            print("The image has been saved.")

        # showing
        plt.show()

    def calculate_and_save_every_day(self, start, total_days, foldername="pkls"):
        days = range(total_days)
        for dday in days:
            delta_t = timedelta(days=dday)
            date = start + delta_t
            date = [date.year, date.month, date.day, date.hour, date.minute, date.second]
            ip = image_plotter(self.sat, sun, earth, dt=self.dt, T=[0, 1, 0, 0], h=[self.h[0]/1e3, self.h[1]/1e3], start_date=date,
                               sat_subpoint=True, full_sun=False);
            ip();
            ip.df.drop("sat", axis=1).to_pickle(foldername + "/" + str(date) + ".pkl")
            clear_output()
            print(date, "done")

    def read_df(self, df):
        self.df = df

    def plot_duration(self, df_indexes=None):
        a = self.get_flyby_info(plot=False)
        if df_indexes is None: # если график можно рисовать непрерывно
            df = [a]
        else: # если график состоит из отдельных кривых
            df = []
            for index_pair in df_indexes:
                df.append(a.iloc[index_pair[0]:index_pair[1]])

        fig, ax = plt.subplots(figsize=(8, 6))
        # fig.subplots_adjust(right=0.75)
        for df_ in df:
            ax.plot(df_["Start"], df_["Duration"],
                    color="steelblue")  # , linestyle = 'None', marker=".")        # plot x and y using default line style and color
        ax.set_xlabel('Date')
        ax.set_ylabel('Duration, sec')
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        # ax.yaxis.label_right()
        ax.set_ylim([10, 1000])
        ax.set_yscale("log")

        ax.grid(b=True, which='major', color='gray', linestyle='-', linewidth=1, alpha=0.5)
        ax.grid(b=True, which='minor', color='gray', linestyle='-', linewidth=1, alpha=0.1)
        plt.savefig('Duration_5-50km.png', dpi=600)
        plt.show()


if __name__ == "__main__":
    # stations_url = 'http://celestrak.com/NORAD/elements/stations.txt'
    # satellites = load.tle(stations_url)
    # iss = satellites['ISS (ZARYA)']
    # print(iss)

    planets = load('de421.bsp')
    sun = planets['sun']
    earth = planets['earth']


    # # ЕСЛИ ЧИТАЕШЬ СТАРЫЙ ФАЙЛ
    # ip = image_plotter(tles, sun, earth, dt=1, T=[0, 0, 11, 0], h=[5, 50], start_date=[2020, 5, 15, 12, 36, 35],
    #                    sat_subpoint=True, full_sun=False)
    # big_df = pd.read_pickle("2019-10-10_2020-10-08_dt1.pkl")
    # ip.read_df(big_df) # пример большого файла
    #
    # # # пример с выделением кусочка из большого файла
    # # tz = timezone(timedelta(hours=0))  # whatever your timezone offset from UTC is
    # # fb_df = big_df[(big_df["Date"] >= datetime(2020, 4, 15, 0, 0, 0, tzinfo=tz)) &
    # #                (big_df["Date"] <= datetime(2020, 4, 16, 0, 0, 0, tzinfo=tz))]
    # # # fb_df.iloc[::30, :]
    # # ip.read_df(fb_df)
    #
    # # a = ip.get_flyby_info(plot=True)
    # ip.plot_duration() # нарисовать график длительности пролетов в зависимости от даты
    # # ip.plot(terminator=False, sat_trajectory=False, save=True, scatter=True, figsize=(8,4)) # нарисовать область на карте, покрываемую за какой-то пролёт
    # # ip.plot_heatmap(bins=100, alpha=1, save=True, figsize=(8, 4), coastlines_color="white") # нарисовать карту суммарных длительностей пролётов


    # ЕСЛИ НОВЫЙ РАСЧЁТ
    tle_parser = TLEArchiveParser("tianhe_tle_archive_parsed", "Tianhe_TLE.txt")
    tles = tle_parser()

    ip = image_plotter(tles, sun, earth, dt=10, T=[1, 0, 0, 0], h=[5, 50], start_date=[2021, 6, 12, 0, 0, 0], sat_subpoint=True, full_sun=False)
    ip()
    ip.calculate_and_save_every_day(start=datetime(year=2021, month=6, day=12), total_days=2)
    # fb_info = ip.get_flyby_info()
    # print(fb_info)
    # ip.plot(sat_trajectory=True, scatter=True, terminator=True)

