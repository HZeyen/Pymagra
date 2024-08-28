# -*- coding: utf-8 -*-
"""
Last modified on Aug. 26, 2024

@author: Hermann Zeyen <hermann.zeyen@universite-paris-saclay.fr>
         Université Paris-Saclay, France

"""

import os
from copy import deepcopy
import numpy as np
from scipy import interpolate
from scipy.interpolate import griddata
from sklearn.preprocessing import QuantileTransformer

# from ..in_out.geometrics import Geometrics
from ..in_out import io


class Utilities:
    """
    Contains the following utility functions for program PyMaGra:

        - extract
        - clean_data
        - diurnal_correction
        - diurnal_variation
        - interpol_line
        - interpol_2D
        - extrapolate
        - gauss_transform
        - justify_lines_gaussian
        - matrixExtension
        - pole_Reduction
        - continuation
        - analyticSignal
        - horizontalDerivative2
        - horizontalDerivative
        - horizontalDerivative2
        - tilt

    """

    def __init__(self, main):
        self.main = main
        self.month_base = []
        self.day_base = []
        self.year_base = []
        self.hour_base = []
        self.minute_base = []
        self.second_base = []
        self.time_base = []
        self.jday_base = []
        self.base = []
        self.grad_inter = []

    def extract(self, choice):
        """
        Mark data segments for treatment as function of odd or even line number

        Parameters
        ----------
        choice : tuple of str
            may be:

            - "all" (choose all lines)
            - "odd" (choose odd lines, natural counting)
            - "even" (choose even line numbers, natural counting)
            - "N", "S", "W" or "E"

        """
        if choice == "all":
            for key, val in self.main.dat[self.main.actual_plotted_file].data.items():
                if isinstance(key, (str)):
                    break
                val["mask"] = True
            self.main.line_choice = "all"
        else:
            self.main.line_choice = choice
            for key, val in self.main.dat[self.main.actual_plotted_file].data.items():
                if isinstance(key, (str)):
                    break
                if choice == "odd":
                    if key % 2 == 1:
                        val["mask"] = True
                    else:
                        val["mask"] = False
                elif choice == "even":
                    if key % 2 == 0:
                        val["mask"] = True
                    else:
                        val["mask"] = False

    def data_flatten(self, data):
        """
        Data that are stored in dictionary data with one entrance per line are
        concatenated into one 1D numpy array

        Parameters
        ----------
        data : dictionary with keys equal to line number
            contains for every line itself a dictionary with the following keys:

            - "s1" : Data of sensor 1
            - "s2" : Data of sensor 2
            - "x", "y", "z": Coordinates of data

        Returns
        -------
        s1 : Numpy 1D float array
             Concatenated data of sensor 1
        s2 : Numpy 1D float array
             Concatenated data of sensor 2
        x  : Numpy 1D float array
             Concatenated x-coordinates of all data
        y  : Numpy 1D float array
             Concatenated y-coordinates of all data
        z  : Numpy 1D float array
             Concatenated z-coordinates of all data
        """
        s1 = []
        s2 = []
        x = []
        y = []
        z = []
        for key, val in data.items():
            if isinstance(key, (str)):
                break
            s1 += list(val["s1"])
            s2 += list(val["s2"])
            x += list(val["x"])
            y += list(val["y"])
            z += list(val["z"])
        return s1, s2, x, y, z

    def clean_data(
        self, min_fix=None, max_fix=None, percent_down=None, percent_up=None
    ):
        """
        Set data to np.nan under certain conditions which may be:

        Parameters
        ----------
        data : dictionary
            Contains data to be cleaned (see io.get_line).
        min_fix : float
            All data below this value are set to nan.
        max_fix : float
            All data above this value are set to nan.
        percent_down : float
            The lowermost percentile values are set to nan
            A value of 0.01 means that all values lower than the
            1% quantile are set to None.
        percent_up : float
            The uppermost percentile values are set to nan
            A value of 0.01 means that all values higher than the
            99% quantile are set to None.

        Returns
        -------
        data : dictionary
            Same structure as inpput data, but with with data outside defined
            limits set to np.nan
        """
        print("\n")
        data = self.main.dat[self.main.actual_plotted_file].data
        grad_data = data["grad_data"]
        s1, s2, _, _, _ = self.data_flatten(data)
        if min_fix:
            for key, val in data.items():
                if isinstance(key, (str)):
                    break
                v = np.copy(val["s1"])
                v[v < min_fix] = np.nan
                val["s1"] = np.copy(v)
                if grad_data:
                    v = np.copy(val["s2"])
                    v[v < min_fix] = np.nan
                    val["s2"] = np.copy(v)
            print(f"Clip below {np.round(min_fix,1)}")
        if max_fix:
            for key, val in data.items():
                if isinstance(key, (str)):
                    break
                v = np.copy(val["s1"])
                v[v > max_fix] = np.nan
                val["s1"] = np.copy(v)
                if grad_data:
                    v = np.copy(val["s2"])
                    v[v > max_fix] = np.nan
                    val["s2"] = np.copy(v)
            print(f"Clip above {np.round(max_fix,1)}")
        if percent_down:
            vmin1 = np.nanquantile(s1, percent_down)
            if grad_data:
                vmin2 = np.nanquantile(s2, percent_down)
            for key, val in data.items():
                if isinstance(key, (str)):
                    break
                v = np.copy(val["s1"])
                v[v < vmin1] = np.nan
                val["s1"] = np.copy(v)
                if grad_data:
                    v = np.copy(val["s2"])
                    v[v < vmin2] = np.nan
                    val["s2"] = np.copy(v)
            if grad_data:
                print(
                    f"Clip below {np.round(vmin1,1)} for sensor 1 and "
                    + f"{np.round(vmin2,1)} for sensor 2"
                )
            else:
                print(f"Clip below {np.round(vmin1,1)}")
        if percent_up:
            vmax1 = np.nanquantile(s1, 1.0 - percent_up)
            if grad_data:
                vmax2 = np.nanquantile(s2, 1.0 - percent_up)
            for key, val in data.items():
                if isinstance(key, (str)):
                    break
                v = np.copy(val["s1"])
                v[v > vmax1] = np.nan
                val["s1"] = np.copy(v)
                if grad_data:
                    v = np.copy(val["s2"])
                    v[v > vmax2] = np.nan
                    val["s2"] = np.copy(v)
            if grad_data:
                print(
                    f"Clip above {np.round(vmax1,1)} for sensor 1 and "
                    + f"{np.round(vmax2,1)} for sensor 2"
                )
            else:
                print(f"Clip above {np.round(vmax1,1)}")
        self.main.dat[self.main.actual_plotted_file].data = data

    def julian2date(self, j_day, year):
        """
        Function translates Julian day number to standard date.
        1st of January is Julian day number 1.

        Parameters
        ----------
        j_day : int
            Number of Julian day
        year : int
            Year in which to do the calculation (important to know whether
            it is a leap year). May be 2 or 4 ciphers

        Returns
        -------
        day: int
            Day of month
        month: int
            Month in year
        """
        day_month = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
        if year % 4 == 0:
            day_month[2:] += 1
        month = np.where(day_month >= j_day)[0][0]
        day = j_day - day_month[month - 1]
        return day, month

    def date2julian(self, day, month, year):
        """
        Function translates month and day of month to Julian day of year.
        1st of January is Julian day number 1.

        Parameters
        ----------
        day : int
            Day of month (natural counting, starts at 1)
        month : int
            Month of year (natural counting, starts at 1 for January)
        year : int
            Year in which to do the calculation (important to know whether
            it is a leap year). May be 2 or 4 ciphers

        Returns
        -------
        j_day: int
            Julian day of year
        """
        day_month = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
        if year % 4 == 0:
            day_month[2:] += 1
        return day_month[month - 1] + day

    def time2data(self, time, year):
        """
        Convert seconds into julian day of year, hour, minute and second

        Parameters
        ----------
        time : may be a single float or a 1D numpy array
            Time to be converted [s], based on julian day of year, i.e.
            time = julian_day*86000+hour*3600+minute*60+second.

        Returns
        -------
        All returned parameters have the same shape as time.

        month :  int
            month of year
        day : int
            day of month.
        h : int
            Hour.
        m : int
            minute.
        s : float
            second.

        """
        try:
            _ = len(time)
            d = np.array(time / 86400.0, dtype=int)
            t = time - d * 86400.0
            h = np.array(t / 3600.0, dtype=int)
            t -= h * 3600.0
            m = np.array(t / 60.0, dtype=int)
            s = t - m * 60.0
            day = np.zeros_like(d)
            month = np.zeros_like(d)
            for i, dd in enumerate(d):
                day[i], month[i] = self.julian2date(dd, year)
        except TypeError:
            day = int(time / 86400.0)
            t = time - day * 86400.0
            h = int(t / 3600.0)
            t -= h * 3600.0
            m = int(t / 60.0)
            s = t - m * 60.0
            day, month = self.julian2date(day, year)
        return month, day, h, m, s

    def diurnal_correction(self, degree=3, diff_weight=5.0):
        """
        Apply diurnal corrections
        The diurnal variations may come from base station data (function read_base)
        or, if no base station data exist, they are calculated in function
        diurnal_variation by fitting a polynomial of degree "degree" to the
        measured data. Data of different days are then fitted independently.

        Base station data (measured or calculated) are interpolated onto
        measurement times and simply subtracted from data. The process is done
        in situ, i.e. the values of arrays self.sensor1 and self.sensor2 are
        modified. If you need to keep the original data, you must copy them to
        other arrays before applying diurnal_correction

        Parameters
        ----------
        degree : int, optional
            Degree of polynom to be fitted to data. The default is 3.
            This parameter is only used if no base station data exist
        diff_weight : float, optional
            If day_joint_flag is true, diff_weight defines the relative weight
            for the fit of the differences along the block edges with respect
            to the fit of the medians.

        """
        # Test whether base station data have already been read
        data = deepcopy(self.main.dat[self.main.actual_plotted_file].data)
        if len(self.main.base_files) == 0:
            # If no base station data exist, use median values of sensor 1 of every line as
            #    indicator for diurnal variations
            #    Add median value at the time of the beginning and the end of every line
            print("\nWARNING in function diurnal_correction:")
            print(
                f"   No base station data exist. Polynom of degree {degree}"
                + " is fitted to data\n"
            )
            self.month_base = []
            self.day_base = []
            self.year_base = []
            self.hour_base = []
            self.minute_base = []
            self.second_base = []
            self.time_base = []
            self.jday_base = []
            self.base = []
            year = data["year"]
            keys = list(data.keys())[: -self.main.string_keys]
            day = np.zeros(max(keys) + 1)
            month = np.zeros_like(day)
            hour1 = np.zeros_like(day)
            hour2 = np.zeros_like(day)
            minute1 = np.zeros_like(day)
            minute2 = np.zeros_like(day)
            second1 = np.zeros_like(day)
            second2 = np.zeros_like(day)
            time1 = np.zeros_like(day)
            time2 = np.zeros_like(day)
            for key, val in data.items():
                if isinstance(key, (str)):
                    break
                month[key], day[key], hour1[key], minute1[key], second1[key] = (
                    self.time2data(val["time"][0], year)
                )
                time1[key] = min(val["time"][0], val["time"][-1])
                _, _, hour2[key], minute2[key], second2[key] = self.time2data(
                    val["time"][-1], year
                )
                time2[key] = max(val["time"][0], val["time"][-1])
            days = np.unique(day)
            for d in days:
                lines = np.where(day == d)[0]
                fit_parameters, tmn = self.diurnal_variation(
                    data, np.array(lines), degree=degree, diff_weight=diff_weight
                )
                npar = degree + 1
                for k in lines:
                    self.month_base.append(month[k])
                    self.month_base.append(month[k])
                    self.day_base.append(day[k])
                    self.day_base.append(day[k])
                    self.year_base.append(year)
                    self.year_base.append(year)
                    self.hour_base.append(hour1[k])
                    self.hour_base.append(hour2[k])
                    self.minute_base.append(minute1[k])
                    self.minute_base.append(minute2[k])
                    self.second_base.append(second1[k])
                    self.second_base.append(second2[k])
                    self.jday_base.append(
                        self.date2julian(int(day[k]), int(month[k]), int(year))
                    )
                    self.time_base.append(data[k]["time"][0])
                    b = 0.0
                    t = self.time_base[-1] - tmn
                    if len(fit_parameters) > npar:
                        blk = data[k]["block"] - 1
                        fit = fit_parameters[blk * npar : (blk + 1) * npar]
                    else:
                        fit = np.copy(fit_parameters)
                    for i in range(npar):
                        b += fit[i] * t ** (degree - i)
                    self.base.append(b)
                    self.time_base.append(data[k]["time"][-1])
                    b = 0.0
                    t = self.time_base[-1] - tmn
                    for i in range(degree + 1):
                        b += fit[i] * t ** (degree - i)
                    self.base.append(b)
            self.month_base = np.array(self.month_base, dtype=int)
            self.day_base = np.array(self.day_base, dtype=int)
            self.jday_base = np.array(self.day_base, dtype=int)
            self.year_base = np.array(self.year_base, dtype=int)
            self.hour_base = np.array(self.hour_base, dtype=int)
            self.minute_base = np.array(self.minute_base, dtype=int)
            self.second_base = np.array(self.second_base)
            self.time_base = np.array(self.time_base)
            self.base = np.array(self.base)
            index = np.argsort(self.time_base)
            self.month_base = self.month_base[index]
            self.day_base = self.day_base[index]
            self.year_base = self.year_base[index]
            self.hour_base = self.hour_base[index]
            self.minute_base = self.minute_base[index]
            self.second_base = self.second_base[index]
            self.jday_base = self.jday_base[index]
            self.time_base = self.time_base[index]
            self.base = self.base[index]
            # io.base = io.Geometrics()
            # io.base.year_base = self.year_base
            # io.base.month_base = self.month_base
            # io.base.day_base = self.day_base
            # io.base.year_base = self.year_base
            # io.base.hour_base = self.hour_base
            # io.base.minute_base = self.minute_base
            # io.base.second_base = self.second_base
            # io.base.time_base = self.time_base
            # io.base.base = self.base
            with open("temp.stn", "w", encoding="utf-8") as fo:
                for i, b in enumerate(self.base):
                    if np.isnan(b):
                        continue
                    fo.write(
                        f"*  0 {self.jday_base[i]:3d} {self.hour_base[i]:02d}"
                        + f"{self.minute_base[i]:02d}{int(self.second_base[i]):02d}"
                        + f"{i:5d}{int(b*10):7d}\n"
                    )
            self.main.base = io.Data()
            self.main.base.read_base("temp.stn", year)
            os.remove("temp.stn")
        else:
            self.time_base = self.main.base.base.time_base
            self.base = self.main.base.base.base
            time1 = []
            time2 = []
            for key, val in data.items():
                if isinstance(key, (str)):
                    break
                time1.append(min(val["time"][0], val["time"][-1]))
                time2.append(max(val["time"][0], val["time"][-1]))
            time1 = np.array(time1)
            time2 = np.array(time2)
        # Check whether base station data cover all measurements.
        # If not, set base station data to zero and avoid in this way base corrections
        if self.time_base.min() > time1.min() or self.time_base.max() < time2.max():
            print("\nWARNING in function  diurnal_correction:")
            print("   Base station data do not cover all measurements.")
            print("           No base station corrections effectuated.\n")
            return
        # interpolate base station values at measurement times
        tb, ind = np.unique(self.time_base, return_index=True)
        b = self.base[ind]
        f = interpolate.interp1d(tb, b, kind=1)
        for key, val in data.items():
            if isinstance(key, (str)):
                break
            diurnal = f(val["time"])
            # Subtract base station data from field data
            val["s1"] -= diurnal
            if data["grad_data"]:
                val["s2"] -= diurnal
        self.main.dat[self.main.actual_plotted_file].data = deepcopy(data)

    def diurnal_variation(self, data, lines, degree=3, diff_weight=1.0):
        """
        Calculate a fit of degree "degree" to the data of lines "lines" which
        will be used for correction of diurnal variations if not base station
        data exist

        Parameters
        ----------
        lines : 1D numpy int array
            Numbers of the lines to be used for calculation of diurnal variations.
            If several files have been loaded, acquired at different days, the
            fit may be calculated independently for every day.
        degree : int, optional
            Degree of polynom to be fitted to data. The default is 3.
        diff_weight : float, optional
            If all blocks are inverted together, diff_weight is the weight
            given to fit of differences along the edges with respect to fit
            of medians in each block. The default is 1.

        Returns
        -------
        Polynome coefficients (1D numpy array of size degree+1)
            The polynome is calculated as
            P[degree]+P[degree-1]*x+P[degree-2]*x**2...+P[0]*x**degree
            If multiple blocks are fitted together, P contains a polynome for
            each block, i.e. len(P) = (degree+1)*number_of_blocks
        tmn : float
            For the stability of polynome fit, times (given in seconds) are
            reduced such that the minimum time is zero. tmn is this minumum time.
            To apply the coefficients to data, their time must be transformed to
            time-tmn before applying the polynome coefficients

        """
        blocks = []
        for key in lines:
            blocks.append(data["block"])
        bl = np.unique(np.array(blocks))
        n_blocks = len(bl)
        if n_blocks == 1:
            d = []
            t = []
            for k in lines:
                d += list(data[k]["s1"])
                t += list(data[k]["time"])
            d = np.array(d)
            t = np.array(t)
            tmn = t.min()
            t -= tmn
            idx = np.isfinite(d) & np.isfinite(t)
            return np.polyfit(t[idx], d[idx], deg=degree), tmn
        n_all_lines = len(blocks)
        m = []
        tm = []
        td_start = []
        td_end = []
        xd_start = []
        xd_end = []
        yd_start = []
        yd_end = []
        v_start = []
        v_end = []
        tmin = data[0]["time"][0]
        for i, b in enumerate(bl):
            m.append([])
            v_start.append([])
            v_end.append([])
            tm.append([])
            td_start.append([])
            td_end.append([])
            xd_start.append([])
            xd_end.append([])
            yd_start.append([])
            yd_end.append([])
            for key, val in data.items():
                if key in lines and val["block"] == b:
                    m[i].append(val["median1"])
                    tm[i].append((val["time"][0] + val["time"][-1]) / 2)
                    td_start[i].append(val["time"][0])
                    td_end[i].append(val["time"][-1])
                    xd_start[i].append(np.round(val["x"][0], 3))
                    xd_end[i].append(np.round(val["x"][-1], 3))
                    yd_start[i].append(np.round(val["y"][0], 3))
                    yd_end[i].append(np.round(val["y"][-1], 3))
                    v_start[i].append(val["s1"][0])
                    v_end[i].append(val["s1"][-1])
            tmin = min(tmin, td_start[i])
        ib1 = []
        ib2 = []
        v1 = []
        v2 = []
        t1 = []
        t2 = []
        for i1 in range(n_blocks - 1):
            for j in range(len(td_start[i1])):
                for i2 in range(i1 + 1, n_blocks):
                    for k in range(len(td_start[i2])):
                        if (
                            xd_start[i1][j] == xd_start[i2][k]
                            and yd_start[i1][j] == yd_start[i2][k]
                        ):
                            ib1.append(i1)
                            ib2.append(i2)
                            v1.append(v_start[i1][j])
                            v2.append(v_start[i2][k])
                            t1.append(td_start[i1][j])
                            t2.append(td_start[i2][k])
                        elif (
                            xd_start[i1][j] == xd_end[i2][k]
                            and yd_start[i1][j] == yd_end[i2][k]
                        ):
                            ib1.append(i1)
                            ib2.append(i2)
                            v1.append(v_start[i1][j])
                            v2.append(v_end[i2][k])
                            t1.append(td_start[i1][j])
                            t2.append(td_end[i2][k])
                        elif (
                            xd_end[i1][j] == xd_start[i2][k]
                            and yd_end[i1][j] == yd_start[i2][k]
                        ):
                            ib1.append(i1)
                            ib2.append(i2)
                            v1.append(v_end[i1][j])
                            v2.append(v_start[i2][k])
                            t1.append(td_end[i1][j])
                            t2.append(td_start[i2][k])
                        elif (
                            xd_end[i1][j] == xd_end[i2][k]
                            and yd_end[i1][j] == yd_end[i2][k]
                        ):
                            ib1.append(i1)
                            ib2.append(i2)
                            v1.append(v_end[i1][j])
                            v2.append(v_end[i2][k])
                            t1.append(td_end[i1][j])
                            t2.append(td_end[i2][k])
        dat = np.zeros(n_all_lines + len(v1))
        G = np.zeros((len(data), n_blocks * (degree + 1)))
        row2 = 0
        col = -1
        for i, b in enumerate(bl):
            t = np.array(tm[i]) - tmin
            row1 = row2
            row2 += len(m[i])
            dat[row1:row2] = m[i]
            for k in range(degree, -1, -1):
                col += 1
                G[row1:row2, col] = t**k
        row = row2 - 1
        for i, v in enumerate(v1):
            row += 1
            dat[row] = v2[i] - v
            col1 = ib1[i] * (degree + 1) - 1
            col2 = ib2[i] * (degree + 1) - 1
            tt1 = t1[i] - tmin
            tt2 = t2[i] - tmin
            for k in range(degree, -1, -1):
                col1 += 1
                col2 += 1
                G[row, col1] = -(tt1**k)
                G[row, col2] = tt2**k
        C = np.diag(np.ones_like(dat))
        C[n_all_lines:] = 1 / diff_weight**2
        mat1 = np.matmul(np.transpose(G), C)
        coefs = np.matmul(np.matmul(np.linalg.inv(np.matmul(mat1, G)), mat1), dat)
        return coefs, tmin

    def interpol_line(self, nsensor, i_line=0, dx=0.2, xmin=0.0, xmax=0.0, k=3):
        """
        interpolate data of one line onto a regular grid

        Parameters
        ----------
        i_line : int, optional
            Number of line to be interpolated (counting starts at 0). The default is 0.
        dx : float, optional
            Sampling step in meters for interpolated data. The default is 0.2.
        xmin : float, optional
            Position of first sample along self.direction in meters. The default is 0.
        xmax : float, optional
            Position of last sample along self.direction in meters. The default is 0.
        k : int, optional
            Degree of spline used for interpolation. The default is 3.
            See scipy.interpolate.interp1d for more information. Only splines are used.
            Correspondance between k and "kind" of scipy.interpolate.interp1d:

            - k=0: kind="zero"
            - k=1: kind="slinear"
            - k=2: kind="quadratic"
            - k=3: kind="cubic"

        If xmin == xmax, the starting and end points are calculated automatically.
        For this,the starting point is placed at the nearest multiple of dx for
        the coordinate of self.direction (see function read_stn)

        Returns
        -------
        sensor_inter: numpy float array
            Interpolated data
        x_inter: numpy float array
            Interpolated X-coordinates
        y_inter: numpy float array
            Interpolated Y-coordinates
        dmin: float
            Position of first interpolated point within line [m]
        dmax: float
            Position of last interpolated point within line [m]

        """
        kind = ["zero", "slinear", "quadratic", "cubic"]
        if k > 3 or k < 0:
            print(
                "\nWARNING Function interpol_line:\n"
                f"        Given k ({k}) is not allowed. k is set to 3 (cubic)"
            )
            k = 3
        # Extract data
        data = self.main.dat[self.main.actual_plotted_file].data
        if nsensor == 1:
            s1 = data[i_line]["s1"]
        else:
            s1 = data[i_line]["s2"]
        index = np.isfinite(s1)
        s1 = s1[index]
        xdat = data[i_line]["x"]
        xdat = xdat[index]
        ydat = data[i_line]["y"]
        ydat = ydat[index]
        # Define coordinates in principal direction along which to interpolate data
        if data["direction"] in ("N", "S"):
            ddat = np.copy(xdat)
        else:
            ddat = np.copy(ydat)
        # Define starting and end points
        if xmin == xmax:
            dmin = np.ceil(np.round(ddat.min(), 2) / dx) * dx
            dmax = np.floor(np.round(ddat.max(), 2) / dx) * dx
        else:
            dmin = xmin
            dmax = xmax
        # Calculate number of interpolated data and their positions along the principal
        #   direction
        nx = np.int((dmax - dmin) / dx + 1)
        d_inter = dx * np.arange(nx) + dmin
        # Do interpolation for first sensor
        f = interpolate.interp1d(ddat, s1, kind=kind[k], fill_value="extrapolate")
        sensor_inter = f(d_inter)
        # Do interpolation for X-coordinates
        f = interpolate.interp1d(ddat, xdat, kind=kind[k], fill_value="extrapolate")
        x_inter = f(d_inter)
        # Do interpolation for Y-coordinates
        f = interpolate.interp1d(ddat, ydat, kind=kind[k], fill_value="extrapolate")
        y_inter = f(d_inter)
        return sensor_inter, x_inter, y_inter, dmin, dmax

    def interpol_2D(self, dx=0.2, dy=0.2):
        """
        Routine interpolates data on all lines onto a regular grid. No extrapolation
        is done, i.e. if at the beginning or the end of a line data are missing
        (the line starts later than others or stops earlier), the interpolated
        array will contain nans
        The interpolation method used is scipy.interpolate.CloughTocher2DInterpolator

        Parameters
        ----------
        dx : float, optional
            Sampling step in meters in x-direction. The default is 0.2.
        dx : float, optional
            Sampling step in meters in y-direction. The default is 0.2.

        Returns
        -------
        sensor1_inter : 2D numpy float array
            Contains gridded data of sensor 1
        sensor2_inter : 2D numpy float array
            Contains gridded data of sensor 2
        grad_inter :2D numpy float array
            contains the vertical gradient

        The shape of the arrays depends on the principal direction of the lines:
        - If self.direction == 1, shape is (number_of_data_points_per_line, number_of_lines)
        - else: (number_of_lines, number_of_data_points_per_line)

        x_inter : 1D numpy float array
            x_coordinates of the columns of s1_inter and s2_inter
        y_inter : 1D numpy float array
            y_coordinates of the rows of s1_inter and s2_inter

        """
        data = self.main.dat[self.main.actual_plotted_file].data
        x = []
        y = []
        z = []
        sensor1 = []
        sensor2 = []
        for key, val in data.items():
            if isinstance(key, (str)):
                break
            x += list(val["x"])
            y += list(val["y"])
            z += list(val["z"])
            sensor1 += list(val["s1"])
            if data["grad_data"]:
                sensor2 += list(val["s2"])
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
        sensor1 = np.array(sensor1)
        if data["grad_data"]:
            sensor2 = np.array(sensor2)
        xmin = np.ceil(x.min() / dx) * dx
        xmax = np.floor(x.max() / dx) * dx
        ymin = np.ceil(y.min() / dy) * dy
        ymax = np.floor(y.max() / dy) * dy
        x_inter = xmin + np.arange(int((xmax - xmin) / dx) + 1) * dx
        y_inter = ymin + np.arange(int((ymax - ymin) / dy) + 1) * dy
        xi, yi = np.meshgrid(x_inter, y_inter)
        sensor1_inter = griddata((x, y), sensor1, (xi, yi), method="linear")
        z_inter = griddata((x, y), z, (xi, yi), method="linear")
        if data["grad_data"]:
            sensor2_inter = griddata((x, y), sensor2, (xi, yi), method="linear")
        # Calculate vertical gradient on interpolated grid
        if data["grad_data"]:
            grad_inter = (sensor1_inter - sensor2_inter) / data["d_sensor"]
            return sensor1_inter, sensor2_inter, grad_inter, x_inter, y_inter, z_inter
        dum = 0
        return sensor1_inter, dum, dum, x_inter, y_inter, z_inter

    def extrapolate(self, d, x, y):
        """
        Routine fills nans on an interpolated grid.
        For this, it searches first for every line and column the first and last
        existing (non-nan) points. Then, for every non-defined point, it searches
        the "n_nearest" nearest points (see first command line) and associates
        a weighted average value. The weight is calculated as 1/distance**2

        Parameters
        ----------
        d : 2D numpy array
               Contains data on regular grid. NaN for inexistant data. Shape: (ny,nx)
        x : 1D numpy array
            Coordiantes of the columns of data
        y : 1D numpy array
            Coordiantes of the rows of data

        Returns
        -------
        data: 2D numpy array with the same shape as input data.
            Contains full regular grid of data

        """
        n_nearest = 8
        data = np.copy(d)
        XX, YY = np.meshgrid(x, y)
        # Search all points at the beginning and the end of all rows
        zlim = []
        xlim = []
        ylim = []
        for i in range(len(x)):
            j1 = len(y)
            j2 = 0
            for j in range(len(y)):
                if np.isfinite(data[j, i]):
                    j1 = j
                    break
            for j in range(len(y) - 1, -1, -1):
                if np.isfinite(data[j, i]):
                    j2 = j
                    break
            if j1 <= j2:
                zlim.append(data[j1, i])
                xlim.append(XX[j1, i])
                ylim.append(YY[j1, i])
                zlim.append(data[j2, i])
                xlim.append(XX[j2, i])
                ylim.append(YY[j2, i])
        # Search all points at the beginning and the end of all columns
        for j in range(len(y)):
            i1 = len(x)
            i2 = 0
            for i in range(len(x)):
                if np.isfinite(data[j, i]):
                    i1 = i
                    break
            for i in range(len(x) - 1, -1, -1):
                if np.isfinite(data[j, i]):
                    i2 = i
                    break
            if i1 <= i2:
                zlim.append(data[j, i1])
                xlim.append(XX[j, i1])
                ylim.append(YY[j, i1])
                zlim.append(data[j, i2])
                xlim.append(XX[j, i2])
                ylim.append(YY[j, i2])
        zlim = np.array(zlim)
        xlim = np.array(xlim)
        ylim = np.array(ylim)
        # Do extrapolation
        for i, xx in enumerate(x):
            for j, yy in enumerate(y):
                if np.isnan(data[j, i]):
                    dist = 1 / ((xx - xlim) ** 2 + (yy - ylim) ** 2)
                    ind = np.argsort(dist)
                    data[j, i] = np.dot(
                        dist[ind[-n_nearest:]], zlim[ind[-n_nearest:]]
                    ) / np.sum(dist[ind[-n_nearest:]])
        return data

    def justify_lines_median(self, just=0, inplace=True):
        """
        Often the measurment direction has an influence on magnetic data due to
        uncorrected effects of acquisition instrumentation.
        The function calculates the median values of every line and adjusts the
        one of every second line to the average median of the neighbouring lines

        Parameters
        ----------
        just : int, optional
            If 0: Leave medians of even line (python counting, i.e. first line is even)
            untouched, justify odd lines to medians of even lines.
            If 1: Do the reverse
        inplace : bool, optional
            if True, justified data are back-copied to self.sensorN_inter and True
            is returned. If not, new arrays are returned The default is True.

        Returns
        -------
        s1_justified : 1D numpy array with justified data of first sensor
        s2_justified : 1D numpy array with justified data of second sensor

        """
        data = deepcopy(self.main.dat[self.main.actual_plotted_file].data)
        keys = np.array(list(data.keys())[: -self.main.string_keys])
        max_key = keys.max()
        if just == 0:
            d_change = data[1]["direction"]
            d_leave = data[0]["direction"]
        else:
            d_change = data[0]["direction"]
            d_leave = data[1]["direction"]
            # for day in self.lines_per_day.keys():
        for key, val in data.items():
            k = keys[1]
            k1 = keys[1]
            k2 = keys[1]
            if isinstance(key, (str)):
                break
            if val["direction"] == d_change:
                if key == 0:
                    for k in keys[1:]:
                        if data[k]["direction"] == d_leave:
                            break
                    dm1 = data[key]["median1"] - data[k]["median1"]
                    if data["grad_data"]:
                        dm2 = data[key]["median2"] - data[k]["median2"]
                elif key == max_key:
                    for k in keys[-1::-1] == d_leave:
                        break
                    dm1 = data[key]["median1"] - data[k]["median1"]
                    if data["grad_data"]:
                        dm2 = data[key]["median2"] - data[k]["median2"]
                else:
                    for k1 in keys[key + 1 :]:
                        if data[k1]["direction"] == d_leave:
                            break
                    for k2 in keys[key - 1 :: -1]:
                        if data[k2]["direction"] == d_leave:
                            break
                    dm1 = (
                        data[key]["median1"]
                        - (data[k1]["median1"] + data[k2]["median1"]) / 2.0
                    )
                    if data["grad_data"]:
                        dm2 = (
                            data[key]["median2"]
                            - (data[k1]["median2"] + data[k2]["median2"]) / 2.0
                        )
                val["s1"] -= dm1
                val["median1"] -= dm1
                if data["grad_data"]:
                    val["s2"] -= dm2
                    val["median2"] -= dm2
        if inplace:
            self.main.dat[self.main.actual_plotted_file].data = deepcopy(data)
            return True
        return data

    def gauss_transform(self, data_fix, data_move):
        """
        Transforms data sets to gaussian distribution does a projection
        of the second data set onto the distribution of the first and returns
        the back-transformed modified second data set

        Parameters
        ----------
        data_fix : numpy 1D array
            Reference data set.
        data_move : numpy 1D array
            Data set to be projected onto the gaussian distribution of data_fix.

        Returns
        -------
        numpy 1D array
            Modified data_move array.

        """
        # For the number of quantiles take the number of data of the smaller data set
        n = min(len(data_fix), len(data_move))
# It seems that the number of quantiles in sklearn.preprocessing.QuantileTransformer
#    is limited to 10000.
        n = min(n,10000)
        # Do the Gauss-transform of the reference data set
        GT_fix = QuantileTransformer(n_quantiles=n, output_distribution="normal")
        _ = GT_fix.fit_transform(data_fix)[:, 0]
        # Do the Gauss-transform of the data set to be modified
        GT_move = QuantileTransformer(n_quantiles=n, output_distribution="normal")
        v_move = GT_move.fit_transform(data_move)
        # Project data_move onto the Gauss distribution of data_fix and return the
        #     back-transformed data.
        return GT_fix.inverse_transform(v_move)[:, 0]

    def justify_lines_gaussian(self, just=0, local=1, inplace=True):
        """
        Often the measurment direction has an influence on magnetic data due to
        uncorrected effects of acquisition instrumentation.
        The function calculates the median values of every line and adjusts the
        one of every second line to the average median of the neighbouring lines

        Parameters
        ----------
        just : int, optional
            If 0: Leve medians of even line (python counting, i.e. first line is even)
            untouched, justify odd lines to medians of even lines
            If 1: Do the reverse
        local : int, optional
            If 0: apply gaussian transform to the whole data set
            If 1: apply gaussian transform only to neighboring lines
        inplace : bool, optional
            If True, justified data are back-copied to self.sensorN_inter and True
            is returned. If not, new arrays are returned The default is True.

        Returns
        -------
        s1_justified : 2D numpy float array
            Justified data of first sensor
        s2_justified : 2D numpy float array
            Justified data of second sensor

        """
        s1_justified = np.copy(self.main.sensor1_inter)
        s2_justified = np.copy(self.main.sensor2_inter)

        if local:
            if just == 0:
                if self.main.direction == 0:
                    nlines = self.main.sensor1_inter.shape[1]
                    for i in range(1, nlines, 2):
                        data_fix = self.main.sensor1_inter[
                            :, i - 1 : i + 2 : 2
                        ].reshape(-1, 1)
                        data_move = self.main.sensor1_inter[:, i].reshape(-1, 1)
                        s1_justified[:, i] = self.gauss_transform(data_fix, data_move)
                        data_fix = self.main.sensor2_inter[
                            :, i - 1 : i + 2 : 2
                        ].reshape(-1, 1)
                        data_move = self.main.sensor2_inter[:, i].reshape(-1, 1)
                        s2_justified[:, i] = self.gauss_transform(data_fix, data_move)
                else:
                    nlines = self.main.sensor1_inter.shape[0]
                    for i in range(1, nlines, 2):
                        data_fix = self.main.sensor1_inter[
                            i - 1 : i + 2 : 2, :
                        ].reshape(-1, 1)
                        data_move = self.main.ensor1_inter[i, :].reshape(-1, 1)
                        s1_justified[i, :] = self.gauss_transform(data_fix, data_move)
                        data_fix = self.main.sensor2_inter[
                            i - 1 : i + 2 : 2, :
                        ].reshape(-1, 1)
                        data_move = self.main.sensor2_inter[i, :].reshape(-1, 1)
                        s2_justified[i, :] = self.gauss_transform(data_fix, data_move)
            else:
                if self.main.direction == 0:
                    nlines = self.main.sensor1_inter.shape[1]
                    data_fix = self.main.sensor1_inter[:, 1].reshape(-1, 1)
                    data_move = self.main.sensor1_inter[:, 0].reshape(-1, 1)
                    s1_justified[:, 0] = self.gauss_transform(data_fix, data_move)
                    data_fix = self.main.sensor2_inter[:, 1].reshape(-1, 1)
                    data_move = self.main.sensor2_inter[:, 0].reshape(-1, 1)
                    s2_justified[:, 0] = self.gauss_transform(data_fix, data_move)
                    for i in range(2, nlines, 2):
                        data_fix = self.main.sensor1_inter[
                            :, i - 1 : i + 2 : 2
                        ].reshape(-1, 1)
                        data_move = self.main.sensor1_inter[:, i].reshape(-1, 1)
                        s1_justified[:, i] = self.gauss_transform(data_fix, data_move)
                        data_fix = self.main.sensor2_inter[
                            :, i - 1 : i + 2 : 2
                        ].reshape(-1, 1)
                        data_move = self.main.sensor2_inter[:, i].reshape(-1, 1)
                        s2_justified[:, i] = self.gauss_transform(data_fix, data_move)
                else:
                    nlines = self.main.sensor1_inter.shape[0]
                    data_fix = self.main.sensor1_inter[1, :].reshape(-1, 1)
                    data_move = self.main.sensor1_inter[0, :].reshape(-1, 1)
                    s1_justified[0, :] = self.gauss_transform(data_fix, data_move)
                    data_fix = self.main.sensor2_inter[1, :].reshape(-1, 1)
                    data_move = self.main.sensor2_inter[0, :].reshape(-1, 1)
                    s2_justified[0, :] = self.gauss_transform(data_fix, data_move)
                    for i in range(2, nlines, 2):
                        data_fix = self.main.sensor1_inter[
                            i - 1 : i + 2 : 2, :
                        ].reshape(-1, 1)
                        data_move = self.main.sensor1_inter[i, :].reshape(-1, 1)
                        s1_justified[i, :] = self.gauss_transform(data_fix, data_move)
                        data_fix = self.main.sensor2_inter[
                            i - 1 : i + 2 : 2, :
                        ].reshape(-1, 1)
                        data_move = self.main.sensor2_inter[i, :].reshape(-1, 1)
                        s2_justified[i, :] = self.gauss_transform(data_fix, data_move)
        else:
            if just == 0:
                if self.main.direction == 0:
                    s = self.gauss_transform(
                        self.main.sensor1_inter[:, 0::2].reshape(-1, 1),
                        self.main.sensor1_inter[:, 1::2].reshape(-1, 1),
                    )
                    s1_justified[:, 1::2] = s.reshape(
                        self.main.sensor1_inter[:, 1::2].shape
                    )
                    s = self.gauss_transform(
                        self.main.sensor2_inter[:, 0::2].reshape(-1, 1),
                        self.main.sensor2_inter[:, 1::2].reshape(-1, 1),
                    )
                    s2_justified[:, 1::2] = s.reshape(
                        self.main.sensor2_inter[:, 1::2].shape
                    )
                else:
                    s = self.gauss_transform(
                        self.main.sensor1_inter[0::2, :].reshape(-1, 1),
                        self.main.sensor1_inter[1::2, :].reshape(-1, 1),
                    )
                    s1_justified[1::2, :] = s.reshape(
                        self.main.sensor1_inter[1::2, :].shape
                    )
                    s = self.gauss_transform(
                        self.main.sensor2_inter[0::2, :].reshape(-1, 1),
                        self.main.sensor2_inter[1::2, :].reshape(-1, 1),
                    )
                    s2_justified[1::2, :] = s.reshape(
                        self.main.sensor1_inter[1::2, :].shape
                    )
            else:
                if self.main.direction == 0:
                    s = self.gauss_transform(
                        self.main.sensor1_inter[:, 1::2].reshape(-1, 1),
                        self.main.sensor1_inter[:, 0::2].reshape(-1, 1),
                    )
                    s1_justified[:, 0::2] = s.reshape(
                        self.main.sensor1_inter[:, 1::2].shape
                    )
                    s = self.gauss_transform(
                        self.main.sensor2_inter[:, 1::2].reshape(-1, 1),
                        self.main.sensor2_inter[:, 0::2].reshape(-1, 1),
                    )
                    s2_justified[:, 0::2] = s.reshape(
                        self.main.sensor2_inter[:, 1::2].shape
                    )
                else:
                    s = self.gauss_transform(
                        self.main.sensor1_inter[1::2, :].reshape(-1, 1),
                        self.main.sensor1_inter[0::2, :].reshape(-1, 1),
                    )
                    s1_justified[0::2, :] = s.reshape(
                        self.main.sensor1_inter[1::2, :].shape
                    )
                    s = self.gauss_transform(
                        self.main.sensor2_inter[1::2, :].reshape(-1, 1),
                        self.main.sensor2_inter[0::2, :].reshape(-1, 1),
                    )
                    s2_justified[0::2, :] = s.reshape(
                        self.main.sensor1_inter[1::2, :].shape
                    )

        if inplace:
            self.main.sensor1_inter = np.copy(s1_justified)
            self.main.sensor2_inter = np.copy(s2_justified)
            return True
        return s1_justified, s2_justified

    def matrixExtension(self, data):
        """
        Creation of extended matix for 2D Fourier transform.
        The toutine mirrors the lower half of the matrix and adds it at the bottom
        and mirrors the upper half to the top. Equivalently right and left

        Parameters
        ----------
        data : 2D numpy array
            Data matrix to be extended

        Returns
        -------
        d : 2D numpy array extended in both directions
            (ny1,nx1): Tuple with starting indices of the original data in matrix d
            (ny2,nx2): Tuple with final indices of the original data in matrix d plus one
            The original data may thus be retrieved as
            data = d[ny1:ny2,nx1:nx2]

        """
        ny, nx = data.shape
        d = np.zeros((2 * (ny - 1), 2 * (nx - 1)))
        nx_add_left = int((nx - 2) / 2)
        nx_add_right = int((nx - 1) / 2)
        ix_add_left = 1 + nx_add_left
        ix_add_right = nx - nx_add_right - 1

        ny_add_down = int((ny - 2) / 2)
        ny_add_up = int((ny - 1) / 2)
        iy_add_down = 1 + ny_add_down
        iy_add_up = d.shape[0] - ny

        nx_right = nx_add_left + nx
        ny_up = ny_add_down + ny
        d[ny_add_down:ny_up, nx_add_left:nx_right] = data
        d[ny_add_down:ny_up, 0:nx_add_left] = np.flip(data[:, 1:ix_add_left], axis=1)
        d[ny_add_down:ny_up, nx_right:] = np.flip(
            data[:, ix_add_right : nx - 1], axis=1
        )
        d[:ny_add_down, :] = np.flip(
            d[iy_add_down : iy_add_down + ny_add_down, :], axis=0
        )
        d[ny_up:, :] = np.flip(d[iy_add_up : iy_add_up + ny_add_up, :], axis=0)
        return d, (ny_add_down, nx_add_left), (ny_add_down + ny, nx_add_left + nx)

    def pole_Reduction(self, data, dx, dy, Inc, Dec):
        """
        Calculation of pole-reduced magnetic data supposing only induced magnetization
        Formula from Keating and Zerbo, Geophysics 61, nᵒ 1 (1996): 131‑137.


        Parameters
        ----------
        data : 2D numpy float array
            Original magnetic data interpolated on a regular grid which may have
            different grid width in x (E-W) and y (N-S) direction.
        dx : float
            grid step in x direction.
        dy : float
            grid step in y direction.
        Inc : float
            Inclination of magnetic field [degrees].
        Dec : float
            Declination of magnetic field [degrees].

        Returns
        -------
        d : 2D numpy float array with the same shape as data
            Reduced to the pole magnetic data

        """
        fac = np.pi / 180.0
        i = Inc * fac
        d = Dec * fac
        cz = np.sin(i)
        cI = np.cos(i)
        sD = np.sin(d)
        cD = np.cos(d)
        cy = cI * cD
        cx = cI * sD
        d, corner1, corner2 = self.matrixExtension(data)
        ny1 = corner1[0]
        nx1 = corner1[1]
        ny2 = corner2[0]
        nx2 = corner2[1]
        dF = np.fft.fft2(d)
        ny, nx = d.shape
        kx = np.fft.fftfreq(nx, dx) * 2 * np.pi
        ky = np.fft.fftfreq(ny, dy) * 2 * np.pi
        u = np.outer(np.ones_like(ky), kx)
        v = np.outer(ky, np.ones_like(kx))
        s = np.sqrt(u**2 + v**2)
        s[0, 0] = 1.0
        fac = (1j * (cx * u + cy * v) / s + cz) ** 2
        fac[0, 0] = 1.0
        dF /= fac
        d = np.fft.ifft2(dF)
        return np.real(d[ny1:ny2, nx1:nx2])

    def continuation(self, data, dx, dy, dz):
        """
        Vertical continuation of potential field data using Fourier transform

        Parameters
        ----------
        data : 2D numpy float array
            Data interpolated onto a regular grid
        dx, dy : float
            Grid spacing in x and y direction [m]
        dz : float
            Distance to continue data [m], positive upwards

        Returns
        -------
        2D numpy float array, same shape as data
            Prolongated data

        """
        d, corner1, corner2 = self.matrixExtension(data)
        ny1 = corner1[0]
        nx1 = corner1[1]
        ny2 = corner2[0]
        nx2 = corner2[1]
        dF = np.fft.fft2(d)
        ny, nx = d.shape
        kx = np.fft.fftfreq(nx, dx) * 2 * np.pi
        ky = np.fft.fftfreq(ny, dy) * 2 * np.pi
        u = np.outer(np.ones_like(ky), kx)
        v = np.outer(ky, np.ones_like(kx))
        s = np.sqrt(u**2 + v**2)
        dF *= np.exp(-s * dz)
        d = np.fft.ifft2(dF)
        return np.real(d[ny1:ny2, nx1:nx2])

    def analyticSignal(self, data, dx, dy):
        """
        Calculation of analytical signal of potential field data via
        vertical and horizontal derivatives

        Parameters
        ----------
        data : 2D numpy float array
               Data interpolated onto a regular grid
        dx, dy : float
               Grid spacing in x and y direction [m]

        Returns
        -------
        2D numpy float array, same shape as data
            Analytic signal
        """
        gh = self.horizontalDerivative(data, dx, dy)
        gv = self.verticalDerivative(data, dx, dy)
        return np.sqrt(gh**2 + gv**2)

    def horizontalDerivative2(self, data, dx, dy):
        """
        Second horizontal derivative of potential field data using Fourier transform

        Parameters
        ----------
        data : 2D numpy float array
               Data interpolated onto a regular grid
        dx, dy : float
               Grid spacing in x and y direction [m]

        Returns
        -------
        2D numpy float array, same shape as data
            2nd horizontal derivative of data
        """
        gx = np.zeros_like(data)
        gx[:, 1:-1] = (data[:, 2:] + data[:, :-2] - 2 * data[:, 1:-1]) / dx**2
        gx[:, 0] = 2 * gx[:, 1] - gx[:, 2]
        gx[:, -1] = 2 * gx[:, -1] - gx[:, -2]
        gy = np.zeros_like(data)
        gy[1:-1, :] = (data[2:, :] + data[:-2, :] - 2 * data[1:-1, :]) / dy**2
        gy[0, :] = 2 * gy[1, :] - gy[2, :]
        gy[-1, :] = 2 * gy[-1, :] - gy[-2, :]
        return np.sqrt(gx**2 + gy**2)

    def horizontalDerivative(self, data, dx, dy):
        """
        First horizontal derivative of potential field data using Fourier transform

        Parameters
        ----------
        data : 2D numpy float array
               Data interpolated onto a regular grid
        dx, dy : float
               Grid spacing in x and y direction [m]

        Returns
        -------
        2D numpy float array, same shape as data
            First horizontal derivative of data
        """
        gx = np.zeros_like(data)
        gx[:, 1:-1] = (data[:, 2:] - data[:, :-2]) / (dx * 2)
        gx[:, 0] = (data[:, 1] - data[:, 0]) / dx
        gx[:, -1] = (data[:, -1] - data[:, -2]) / dx
        gy = np.zeros_like(data)
        gy[1:-1, :] = (data[2:, :] - data[:-2, :]) / (2 * dy)
        gy[0, :] = (data[1, :] - data[0, :]) / dy
        gy[-1, :] = (data[-1, :] - data[-2, :]) / dy
        return np.sqrt(gx**2 + gy**2)

    def verticalDerivative2(self, data):
        """
        Second vertical derivative of potential field data using finite differences

        Parameters
        ----------
        data : 2D numpy float array
               Data interpolated onto a regular grid

        Returns
        -------
        2D numpy float array, same shape as data
            2nd vertical derivative of data
        """
        gz2 = np.zeros_like(data)
        gz2[:, :] = np.nan
        gz2[1:-1, 1:-1] = (
            data[:-2, 1:-1]
            + data[2:, 1:-1]
            + data[1:1, :-2]
            + data[1:-1, 2:]
            - 4 * data[1:-1, 1:-1]
        )
        return gz2

    def verticalDerivative(self, data, dx, dy):
        """
        First vertical derivative of potential field data using Fourier transform

        Parameters
        ----------
        data : 2D numpy float array
               Data interpolated onto a regular grid
        dx, dy : float
               Grid spacing in x and y direction [m]

        Returns
        -------
        2D numpy float array, same shape as data
            First vertical derivative of data
        """
        d, corner1, corner2 = self.matrixExtension(data)
        dF = np.fft.fft2(d)
        ny1 = corner1[0]
        nx1 = corner1[1]
        ny2 = corner2[0]
        nx2 = corner2[1]
        ny, nx = d.shape
        kx = np.fft.fftfreq(nx, dx) * 2 * np.pi
        ky = np.fft.fftfreq(ny, dy) * 2 * np.pi
        u = np.outer(np.ones_like(ky), kx)
        v = np.outer(ky, np.ones_like(kx))
        s = np.sqrt(u**2 + v**2)
        dF *= s
        return np.real(np.fft.ifft2(dF)[ny1:ny2, nx1:nx2])

    def tilt(self, data, grad, dx, dy):
        """
        Tilt angle of potential field data using Fourier transform

        Parameters
        ----------
        data : 2D numpy float array
               Data interpolated onto a regular grid
        grad : 2D numpy float array
               Vertical derivative of data if it has been measured
        dx, dy : float
               Grid spacing in x and y direction [m]

        Returns
        -------
        2D numpy float array, same shape as data
            Tilt angle of data
        """
        if not self.main.dat[self.main.actual_plotted_file].data["grad_data"]:
            grad = self.verticalDerivative(data, dx, dy)
            self.grad_inter = np.copy(grad)
        grad2 = self.verticalDerivative(grad, dx, dy)
        gh = self.horizontalDerivative(data, dx, dy)
        return np.arctan2(grad, abs(gh)), grad, grad2, gh
