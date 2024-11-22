# -*- coding: utf-8 -*-
"""
Last modified on Nov 08, 2024

@author: Hermann Zeyen University Paris-Saclay, France
"""

import sys
import numpy as np
from PyQt5 import QtWidgets
from ..utilities import utilities as utils


class Geometrics:
    """
    Input and output of Geometrics format files.

    Methods:

    - __init__
    - fill_dict
    - read_stn
    - read_dat
    - write_stn
    - clean_data
    - line_days
    - get_line
    - geometrics_lines
    - get_segment
    - read_base
    - write_base

    """

    def __init__(self):
        """
        Class Geometrics reads and writes Geometrics .stn files and allows
        a number of data treatments

        Parameters
        ----------

        Returns
        -------
        None.

        """
        # Initialize main frame widget
        #        super(Geometrics, self).__init__()
        super().__init__()
        self.n_lines = 0
        self.n_data = 0
        self.grad_data = True
        self.geography_flag = False
        self.grid_flag = True
        self.unit = "nT"
        # \        self.read_stn(file)
        self.lines_per_day = {}
        self.grad = np.array([])
        self.year_base = np.array([])
        self.month_base = np.array([])
        self.day_base = np.array([])
        self.hour_base = np.array([])
        self.minute_base = np.array([])
        self.second_base = np.array([])
        self.time_base = np.array([])
        self.jday_base = np.array([])
        self.base = np.array([])
        self.year = np.array([])
        self.month = np.array([])
        self.day = np.array([])
        self.hour = np.array([])
        self.minute = np.array([])
        self.second = np.array([])
        self.time = np.array([])
        self.x = np.array([])
        self.y = np.array([])
        self.x1 = np.array([])
        self.y1 = np.array([])
        self.x2 = np.array([])
        self.y2 = np.array([])
        self.sensor1 = np.array([])
        self.sensor2 = np.array([])
        # self.sensor1_ori = np.array([])
        # self.sensor2_ori = np.array([])
        # self.grad_ori = np.array([])
        self.n_blocks = 0
        self.d_sensor = 0.0
        self.d_sensor2 = 0.0
        self.h_sensor = 0.0
        self.dispo = 0
        self.infile = ""
        self.segments = {}
        self.grad_flag = False
        self.dx = 0.0
        self.dy = 0.0
        self.direction = 0
        self.d_lines = 0.0
        self.line_pos = np.array([])
        self.line_choice = "all"

    def fill_dict(self, ntracks, x1, y1, mag1, x2, y2, mag2, mark_samples,
                  dx_seg, dy_seg, d_seg):
        """
        Fill in values of dictionary self.segments for a full line.

        Parameters
        ----------
        ntracks : int
            Actual line number (in horizontal configuration, dictionary
            entrancies ntracks and ntarcks+1 are filled).
        x1 : list of floats
            X-coordinates for sensor 1 data.
        y1 : list of floats
            Y-coordinates for sensor 1 data.
        mag1 : list of floats
            Magnetic data recorded by sensor 1.
        x2 : list of floats
            X-coordinates for sensor 2 data.
        y2 : list of floats
            Y-coordinates for sensor 2 data.
        mag2 : list of floats
            Magnetic data recorded by sensor 1.
        mark_samples : list of int
            Number of the sample just after a mark.
        dx_seg : list of floats
            Step length in x-direction for every segment.
        dy_seg : list of floats
            Step length in y-direction for every segment.
        d_seg : list of floats
            Total step length for every segment.

        Returns
        -------
        None.

        """
        dx = x1[-1] - x1[0]
        dy = y1[-1] - y1[0]
        nt1 = ntracks + 1
        if ntracks == 0:
            nshift = 0
        else:
            nshift = self.segments[ntracks - 1]["mark_samples"][-1]
        if abs(dx) > abs(dy):
            if dx > 0:
                self.segments[ntracks]["direction"] = 90.
            else:
                self.segments[ntracks]["direction"] = -90.
        else:
            if dy > 0:
                self.segments[ntracks]["direction"] = 0.
            else:
                self.segments[ntracks]["direction"] = 180.
        self.segments[ntracks]["median1"] = np.nanmedian(mag1)
        self.segments[ntracks]["mask"] = True
        self.segments[ntracks]["block"] = self.n_blocks
        self.segments[ntracks]["mark_samples"] = (
            np.array(mark_samples, dtype=int) + nshift)
        self.segments[ntracks]["dx"] = np.array(dx_seg)
        self.segments[ntracks]["dy"] = np.array(dy_seg)
        self.segments[ntracks]["d"] = np.array(d_seg)
        self.segments[ntracks]["x"] = np.nanmedian(x1)
        self.segments[ntracks]["y"] = np.nanmedian(y1)
        if self.dispo == 0:
            self.segments[ntracks]["median2"] = np.nanmedian(mag2)
            self.segments[ntracks]["sensor"] = 0
        else:
            self.segments[ntracks]["sensor"] = 1
            self.segments[ntracks]["median2"] = 0.0
            self.segments[nt1]["median1"] = np.nanmedian(mag2)
            self.segments[nt1]["median2"] = 0.0
            self.segments[nt1]["x"] = np.nanmedian(x2)
            self.segments[nt1]["y"] = np.nanmedian(y2)
            self.segments[nt1]["mask"] = True
            self.segments[nt1]["block"] = self.n_blocks
            self.segments[nt1]["sensor"] = 2
            self.segments[nt1]["direction"] =\
                self.segments[ntracks]["direction"]
            self.segments[nt1]["mark_samples"] = (
                np.array(mark_samples, dtype=int)
                + self.segments[ntracks]["mark_samples"][-1])
            self.segments[nt1]["dx"] = np.array(dx_seg)
            self.segments[nt1]["dy"] = np.array(dy_seg)
            self.segments[nt1]["d"] = np.array(d_seg)

    def append_data(self, x, y, d, year, month, day, hour, minute, second,
                    time, mark, line):
        """
        Appends date, time, coordinate information of one segment to arrays

        Parameters
        ----------
        x: list of floats
            x-coordinate of data points
        y: list of floats
            y-coordinate of data points
        d: list of floats
            measured values of sensor 1
        year: list of int
            year of acquisition of data points
        month: list of int
            month of acquisition of data points
        day: list of int
            day of acquisition of data points
        hour: list of int
            hour of acquisition of data points
        minute: list of int
            minute of acquisition of data points
        second: list of floats
            second of acquisition of data points
        time: list of int
            julian day of data points
        mark: list of int
            mark numbers
        line: list of int
            mark numbers
        """
        self.x = np.concatenate((self.x, np.round(np.array(x, dtype=float),
                                                  3)))
        self.y = np.concatenate((self.y, np.round(np.array(y, dtype=float),
                                                  3)))
        self.sensor1 = np.concatenate((self.sensor1, np.array(d, dtype=float)))
        self.day = np.concatenate((self.day, np.array(day, dtype=int)))
        self.month = np.concatenate((self.month, np.array(month, dtype=int)))
        self.year = np.concatenate((self.year, np.array(year, dtype=int)))
        self.hour = np.concatenate((self.hour, np.array(hour, dtype=int)))
        self.minute = np.concatenate((self.minute, np.array(minute,
                                                            dtype=int)))
        self.second = np.concatenate((self.second, np.array(second,
                                                            dtype=float)))
        tt = (np.array(time, dtype=int) * 86400.0
              + np.array(hour, dtype=int) * 3600.0
              + np.array(minute, dtype=int) * 60.0
              + np.array(second, dtype=float))
        self.time = np.concatenate((self.time, tt))
        self.mark = np.concatenate((self.mark, np.array(mark, dtype=int)))
        self.line = np.concatenate((self.line, np.array(line, dtype=int)))

    def read_stn(self, infile, n_block, h1_sensor=1.4, h2_sensor=0.4, dispo=0):
        """
        Reads a 2-sensor Geometix .stn file
        If other data have already been read from another file, the new ones
        are added to the vectors with new line numbers, even if effectively,
        the measurements may have been done in the prolongation of existing
        lines.

        Parameters
        ----------
        infile : str
            Name of file to be read.
        n_block : int
            Number of block (file) of this data set within all files to be read
        h1_sensor : float. Optional, default: 1.4 m.
            Height over ground of sensors 1 [m]. Usually, this is the upper
            sensor
        h2_sensor : float, optional, default: 0.4 m.
            Hight over ground of sensor 2 (lower sensor) [m]. Default: 0.4 m
        dispo : int, optional
            If two sensors were used, dispo indicates there relative position:
            - If dispo == 0: vertical disposition, sensor 2 above sensor 1.
            - If dispo == 1: horizontal disposition, sensor 2 to the right of
            sensor 1.

        Data are stored in the following arrays:

        - self.sensor1 : 1D numpy float array contains all data of sensor 1
        - self.sensor2 : 1D numpy float array contains all data of sensor 2
        - self.day     : 1D numpy int array contains day of acquisition for all
                         data
        - self.month   : 1D numpy int array contains month of acquisition for
                         all data
        - self.year    : 1D numpy int array contains year of acquisition for
                         all data
        - self.hour    : 1D numpy int array contains hour of acquisition for
                         all data
        - self.minute  : 1D numpy int array contains minute of acquisition for
                         all data
        - self.second  : 1D numpy float array contains second of acquisition
                         for all data
        - self.time    : 1D numpy float array contains seconds of acquisition
                         for all data counted from the beginning of the
                         acquisition year.
        - self.x       : 1D numpy float array contains x-coordinates of
                         measurement points
        - self.y       : 1D numpy float array contains y-coordinates of
                         measurement points
        - self.segments: Dictionary with information for all lines and segments
                         measured

            Since all data are stored sequentially, we need to know which data
            belong to which line and it may also be necessary to know where
            marks were set, thus where a measured segment starts and ends.
            self.segments.keys() contains line numbers starting at zero
            Every dictionary entrance contains the following lists:

            - "mark_samples" contains for every mark (starting and
              end of line included) the number of the sample in self.sensor1
              or 2 recorded just after the mark
            - "mask" is initially set to True but may be modified in order to
              plot only certain lines. If True, lines are plotted, else they
              are excluded from plot
            - "median1" : median value of sensor 1 for a line
            - "median2" : median value of sensor 2 for a line
              If sensors are in horizontal configuration, both sensors are
              treated as sensor 1 and "median2" has value 0.
            - "sensor" : may be 0 if both sensors are used in vertical
              configuration, 1 or 2 if line has been measured by sensor 1 or 2
              respectively in horizontal configuration.
            - "dx" contains for every segment the sampling step in X direction.
            - "dy" contains for every segment the sampling step in Y direction.
            - "d" contains for every segment the sampling step length.
              dx, dy and d contain one value less than mark_samples
            - "x" median of x-coordinates
            - "y" median of y-coordiantes
              depending on the line direction, "x" or "y" are supposed to be
              the position of the line
            - "pos" coordinate of the line
            - "block" number of block (or file) if several files are read
            - "dir" may be "odd" or "even". First measured line is "odd"
              (odd or even refers to natural counting)
            - "direction" is 0., 180., 90. or -90. depending on the line
              direction

            In addition, the following attributes are accessible:

            - self.n_data   : int, total number of data
            - self.n_lines : int, number of lines read
            - self.direction : int, direction of measurement lines.
              May be 0 (Y-direction) or 1 (X-direction)
            - self.d_lines : float, distance between measurement lines

        Returns
        -------
        data: dictionary with one entry per line, key = number of line.
              Each entry is itself a dictionary containing the following
              entries:

              - "s1": Numpy float array with data of sensor 1
              - "s2": Numpy float array with data of sensor 2
                If only data of one single sensor were measured, "s2" contains
                only one zero.
              - "x":  E-W coordinates of data points
              - "y"   N-S coordinates of data points

        """
        # Read magnetic field data ("stn" file)
        self.n_blocks = n_block
        self.d_sensor = h1_sensor-h2_sensor
        self.d_sensor2 = self.d_sensor / 2.0
        self.h_sensor = h2_sensor
        self.dispo = dispo
        self.infile = infile
        mag1 = []
        mag2 = []
        xpt = []
        ypt = []
        xpt1 = []
        ypt1 = []
        xpt2 = []
        ypt2 = []
        year_m = []
        month_m = []
        day_m = []
        hour_m = []
        min_m = []
        sec_m = []
        time_m = []
        line_nr = []
        mark = []
        if self.n_lines == 0:
            self.sensor1 = np.array([])
            self.sensor2 = np.array([])
            self.x = np.array([])
            self.y = np.array([])
            self.line = np.array([], dtype=int)
            self.mark = np.array([], dtype=int)
            # self.x1 = np.array([])
            # self.y1 = np.array([])
            # self.x2 = np.array([])
            # self.y2 = np.array([])
            self.day = np.array([], dtype=int)
            self.month = np.array([], dtype=int)
            self.year = np.array([], dtype=int)
            self.hour = np.array([], dtype=int)
            self.minute = np.array([], dtype=int)
            self.second = np.array([])
            self.time = np.array([])
            self.segments = {}
            n_start = 0
            first = True
        else:
            n_start = len(self.sensor1)
            first = False

        ntracks = self.n_lines - 1

        # read all lines in magnetic data file
        with open(infile, "r", encoding="utf-8") as fh:
            ll = fh.readlines()
        try:
            _ = int(ll[0].split()[0])
        except Exception as e:
            # If none of the above, stop program
            _ = QtWidgets.QMessageBox.critical(
                None, "Error", f"Error: {e}\n\n"
                + f"File {infile} is not a Geometrics file.\nProgram stops",
                QtWidgets.QMessageBox.Ok,)
            raise IOError("Wrong file type\n") from e
#            sys.exit()
        # store reversed order in list "lines"
        lines = ll[::-1]
        last_mark = 5
        ndat_seg = 0
        xx0 = 0.0
        yy0 = 0.0
        ddx = 0.0
        ddy = 0.0
        nstart = True
        nt1 = ntracks
        mark_samples = []
        dx_seg = []
        dy_seg = []
        d_seg = []

        # read the lines and get data values and corresponding coordinates
        for lin in lines:
            line = lin.split()
# if first number in line is 3, it is a mark (including beginning and end of
#    line) seg_break_data contains number of data point before which mark is
#    located
#    id_seg_break contains the type of mark
#    seg_break_line contains the line in the data file where mark is located
#    line segments contains for every line the position of marks (between
#    marks, data positions are interpolated at regular spacing)
#    dx_seg contains distance between samples for every segment
            if int(line[0]) == 3:
                xx = np.round(float(line[1]), 3)
                yy = np.round(float(line[2]), 3)
                line_number = int(line[6])
                mark_number = int(line[7])
                # code 36 indicates beginning of a new line
                # code 4 indicates a mark, 5 is the Endline mark
                if int(line[8]) in (4, 5):
                    if ndat_seg == 0:
                        continue
                    mark_samples.append(len(mag1) + n_start)
                    n1 = mark_samples[-2]
                    n2 = mark_samples[-1]
                    ndat_segment = n2 - n1
                    if ndat_segment == 0:
                        del mark_samples[-1]
                        continue
                    # print(f"nline: {ntracks}: nseg: "+\
                    #       f"{len(self.segments[ntracks]['mark_samples'])-1},"+\
                    #       f" ndseg: {ndat_segment}")
# If last mark had code 36, a new line was started and it is
#    the first mark after start. The number of measurement steps
#    is 1 less than the number of data points, if not,
#    both are equal. This means that it is supposed that the
#    mark was pressed atthe position of the measurement just
#    before the mark
                    if nstart:
                        dx = (xx - xx0) / (ndat_segment - 1)
                        dy = (yy - yy0) / (ndat_segment - 1)
                        n2 -= 1
                    else:
                        dx = (xx - xx0) / ndat_segment
                        dy = (yy - yy0) / ndat_segment
                    d = np.sqrt(dx * dx + dy * dy)
                    dx_seg.append(dx)
                    dy_seg.append(dy)
                    d_seg.append(d)
                    if self.dispo > 0:
                        ddx1 = ddx
                        ddy1 = ddy
                        nt1 = ntracks + 1
                        ddx = dy / d * self.d_sensor2
                        ddy = dx / d * self.d_sensor2
                    else:
                        nt1 = ntracks
                        ddx1 = 0.0
                        ddy1 = 0.0
                        ddx = 0.0
                        ddy = 0.0
                    if nstart:
                        xpt1[0] -= ddx
                        ypt1[0] += ddy
                        xpt2[0] += ddx
                        ypt2[0] -= ddy
                    else:
                        xpt1.append(xpt1[-1] + dx - ddx + ddx1)
                        ypt1.append(ypt1[-1] + dy + ddy - ddy1)
                        xpt2.append(xpt2[-1] + dx + ddx - ddx1)
                        ypt2.append(ypt2[-1] + dy - ddy + ddy1)
                        n1 += 1
                    for _ in range(n1, n2):
                        xpt1.append(xpt1[-1] + dx)
                        ypt1.append(ypt1[-1] + dy)
                        xpt2.append(xpt2[-1] + dx)
                        ypt2.append(ypt2[-1] + dy)
                    last_mark = int(line[8])
                    xx0 = xx
                    yy0 = yy
                    nstart = False
                if int(line[8]) == 5 or (int(line[8]) == 36
                                         and last_mark != 5):
                    self.fill_dict(ntracks, xpt1, ypt1, mag1, xpt2, ypt2, mag2,
                                   mark_samples, dx_seg, dy_seg, d_seg)
                    self.append_data(xpt1, ypt1, mag1, year_m, month_m, day_m,
                                     hour_m, min_m, sec_m, time_m, mark,
                                     line_nr)
                    if self.dispo:
                        self.append_data(xpt2, ypt2, mag2, year_m, month_m,
                                         day_m, hour_m, min_m, sec_m, time_m,
                                         mark, line_nr)
                        self.n_lines += 1
                    else:
                        self.sensor2 = np.concatenate(
                            (self.sensor2, np.array(mag2, dtype=float)))
                    last_mark = int(line[8])
                    ndat_seg = 0
# xpt and ypt contain the coordinates of all data points of the actual line
                if int(line[8]) == 36:
                    xx0 = xx
                    yy0 = yy
                    xpt1 = [xx0]
                    ypt1 = [yy0]
                    xpt2 = [xx0]
                    ypt2 = [yy0]
                    dx_seg = []
                    dy_seg = []
                    d_seg = []
                    mag1 = []
                    mag2 = []
                    time_m = []
                    month_m = []
                    day_m = []
                    hour_m = []
                    min_m = []
                    sec_m = []
                    line_nr = []
                    mark = []
                    ndat_segment = int(line[5])
                    ndat_seg = 0
                    nstart = True
                    ntracks = nt1 + 1
                    self.n_lines += 1
                    self.segments[ntracks] = {}
                    mark_samples = [0]
                    if self.dispo:
                        self.segments[ntracks + 1] = {}
                    continue
# Don't take into account other mark codes (mainly Pause)
                continue
# if first number in line is 0, it is a data point
# mag1 contains data of first sensor, mag 2 of second sensor
# In addition, times are stored
            if int(line[0]) == 0:
                mag1.append(float(line[1]))
                mag2.append(float(line[2]))
                t = line[3].split(":")
                hour_m.append(int(t[0]))
                min_m.append(int(t[1]))
                sec_m.append(float(t[2]))
                t = line[4].split("/")
                day_m.append(int(t[1]))
                month_m.append(int(t[0]))
                year_m.append(int(t[2]))
                time_m.append(utils.date2julian(day_m[-1], month_m[-1],
                                                year_m[-1]))
                line_nr.append(line_number)
                mark.append(mark_number)
                ndat_seg += 1

        if self.n_lines > 0 and last_mark != 5:
            self.fill_dict(ntracks, xpt1, ypt1, mag1, xpt2, ypt2, mag2,
                           mark_samples, dx_seg, dy_seg, d_seg)
            self.append_data(xpt1, ypt1, mag1, year_m, month_m, day_m, hour_m,
                             min_m, sec_m, time_m, mark, line_nr)
            if self.dispo > 0:
                self.append_data(xpt2, ypt2, mag2, year_m, month_m, day_m,
                                 hour_m, min_m, sec_m, time_m, mark, line_nr)
            else:
                self.sensor2 = np.concatenate(
                    (self.sensor2, np.array(mag2, dtype=float)))
        self.n_data += len(mag1)
        if len(self.sensor1) == len(self.sensor2):
            self.grad = (self.sensor1 - self.sensor2) / self.d_sensor
            self.grad_flag = True
        else:
            self.grad = np.array([0])
            self.grad_flag = False
        del mag1, mag2, xpt, ypt, day_m, month_m, year_m, hour_m, min_m, sec_m
        self.dx = self.segments[0]["dx"][0]
        self.dy = self.segments[0]["dy"][0]
        self.z = np.zeros_like(self.x)
        if first:
            if abs(self.x[0]-self.x[self.segments[0]["mark_samples"][-1]]) > \
                    abs(self.y[0]-self.y[self.segments[0]["mark_samples"][-1]]
                        ):
                self.direction = 1
                self.d_lines = np.round(
                    (self.y.max() - self.y.min()) / (self.n_lines - 1), 1)
            else:
                self.direction = 0
                self.d_lines = np.round(
                    (self.x.max() - self.x.min()) / (self.n_lines - 1), 1)
# Pass certain dictionary data into arrays
        self.mark_samples = []
        for key, val in self.segments.items():
            self.mark_samples.append(val["mark_samples"])
# Store original data to arrays xxx_ori
        # self.sensor1_ori = np.copy(self.sensor1)
        # self.sensor2_ori = np.copy(self.sensor2)
        # self.grad_ori = np.copy(self.grad)
        self.line_pos = np.zeros(len(self.segments.keys()))
        if self.direction == 1:
            for key, entries in self.segments.items():
                self.line_pos[key] = entries["y"]
                self.segments[key]["pos"] = entries["y"]
        else:
            for key, entries in self.segments.items():
                self.line_pos[key] = entries["x"]
                self.segments[key]["pos"] = entries["x"]
        self.line_choice = "all"
        if self.direction == 0:
            sens = np.sign(self.y[1] - self.y[0])
            for key, entries in self.segments.items():
                i1 = entries["mark_samples"][0]
                if np.sign(self.y[i1 + 1] - self.y[i1]) == sens:
                    entries["dir"] = "odd"
                else:
                    entries["dir"] = "even"
        else:
            sens = np.sign(self.x[1] - self.x[0])
            for key, entries in self.segments.items():
                i1 = entries["mark_samples"][0]
                if np.sign(self.x[i1 + 1] - self.x[i1]) == sens:
                    entries["dir"] = "odd"
                else:
                    entries["dir"] = "even"

    def read_dat(self, file, n_block, h1_sensor, h2_sensor, dispo):
        """
        Reads a 1 or 2-sensor Geometix .dat file ("Surfer" format in MagMap).
        To simplify coding, the function writes a temporary file "temp.stn" and
        calls read_stn to import the data.

        Parameters
        ----------
        file : str
            Name of file to be read.
        n_block : int
            Number of block (file) of this data set within all files to be read
        h1_sensor : float
            Hight of sensor 1 (usually upper sensor) above ground [m].
        h2_sensor : float
            Hight of sensor 2 (usually lower sensor) above ground [m].
        dispo : int, optional
            If two sensors were used, dispo indicates there relative position:

            - If dispo == 0: vertical disposition, sensor 2 above sensor 1.
            - If dispo == 1: horizontal disposition, sensor 2 to the right of
              sensor 1.

        """
        with open(file, "r") as fi:
            lines = fi.readlines()
        if "TOP" in lines[0] and "BOTTOM" in lines[0]:
            self.grad_flag = True
        else:
            self.grad_flag = False
        col_t = -1
        col_d = -1
        col_l = -1
        col_x = -1
        col_y = -1
        # col_z = -1
        col_1 = -1
        col_2 = -1
        col_m = -1
        val = lines[0].split()
        if "TIME" in lines[0]:
            for col, l in enumerate(val):
                if "TIME" in l:
                    col_t = col
                    break
        if "DATE" in lines[0]:
            for col, l in enumerate(val):
                if "DATE" in l:
                    col_d = col
                    break
        if "LINE" in lines[0]:
            for col, l in enumerate(val):
                if "LINE" in l:
                    col_l = col
                    break
        if col_t < 0 or col_d < 0 or col_l < 0:
            _ = QtWidgets.QMessageBox.warning(
                None, "Warning",
                f"File {file}:\n\nTrying to read Geometrics .dat file.\n"
                + "The file must contain TIME, DATE and LINE columns, but does"
                + " not\n.As such, the file cannot be read by pymagra.\n"
                + "Go back to MagMap and export data with timing and line\n"
                + "information or open .stn file.\n\nPymagra aborted.",
                QtWidgets.QMessageBox.Close, QtWidgets.QMessageBox.Close)
            sys.exit()
        for col, l in enumerate(val):
            if "X" in l:
                col_x = col
                break
        for col, l in enumerate(val):
            if "Y" in l:
                col_y = col
                break
        if self.grad_flag:
            for col, l in enumerate(val):
                if "TOP" in l:
                    col_1 = col
                    break
            for col, l in enumerate(val):
                if "BOTTOM" in l:
                    col_2 = col
                    break
        else:
            for col_1, l in enumerate(val):
                if "TOP" in l or "BOTTOM" in l:
                    col_1 = col
                    break
        # for col, l in enumerate(val):
        #     if "Z" in l:
        #         col_z = col
        #         break
        for col, l in enumerate(val):
            if "MARK" in l:
                col_m = col
                break
        with open("temp.stn", "w") as fo:
            fo.write("99 0 0\n")
            line0 = int(lines[1].split()[col_l])-1
            izero = 0
            n_lines = 0
            ndat = 0
            x = 0.
            y = 0.
            time = ""
            date = ""
            mark = 0
            for lin in lines[1:]:
                val = lin.split()
                line = int(val[col_l])
                if line == line0:
                    x = float(val[col_x])
                    y = float(val[col_y])
                    time = val[col_t]
                    date = val[col_d]
                    d1 = float(val[col_1])
                    ndat += 1
                    if self.grad_flag:
                        d2 = float(val[col_2])
                        fo.write(f"0{d1:12.3f}{d2:11.3f} {time} {date}   0\n")
                    else:
                        fo.write(f"0{d1:12.3f} {time} {date}   0\n")
                else:
                    if n_lines > 0:
                        fo.write(f"3{x:14.3f}{y:13.3f} {time} {date}"
                                 + f"{ndat:11d}{line:12d}{mark+1:12d}  36\n")
                    x = float(val[col_x])
                    y = float(val[col_y])
                    time = val[col_t]
                    date = val[col_d]
                    line = int(val[col_l])
                    line0 = line
                    n_lines += 1
                    if col_m > -1:
                        mark = val[col_m]
                    else:
                        mark = 0
                    fo.write(f"3{x:14.3f}{y:13.3f} {time} {date}"
                             + f"{mark:11d}{line:12d}{izero:12d}   5\n")
                    ndat = 1
            fo.write(f"3{x:14.3f}{y:13.3f} {time} {date}"
                     + f"{ndat:11d}{line:12d}{mark+1:12d}  36\n")
        self.read_stn("temp.stn", n_block, h1_sensor, h2_sensor, dispo)

    def write_stn(self, file):
        """
        Writes Geometrics magnetic gradiometer file in stn format.

        Parameters
        ----------
        file : str
            File where to store the data.
        """
        counter = np.arange(len(self.x))
        line0 = -1000
        mark0 = -1000
        ncount = 0
        izero = 0
        with open(file, "w") as fh:
            fh.write("99 0 0\n")
            for i in counter[::-1]:
                if self.line[i] != line0:
                    line0 = self.line[i]
                    mark0 = self.mark[i]
                    if ncount > 0:
                        fh.write(f"3{self.x[i+1]:14.3f}{self.y[i+1]:13.3f} "
                                 + f"{self.hour[i+1]:02d}:"
                                 + f"{self.minute[i+1]:02d}:"
                                 + f"{self.second[i+1]:05.2f} "
                                 + f"{self.month[i+1]:02d}/"
                                 + f"{self.day[i+1]:02d}/{self.year[i+1]:02d}"
                                 + f"{ncount:11d}{self.line[i+1]:12d}"
                                 + f"{self.mark[i+1]:12d}  36\n")
                    fh.write(f"3{self.x[i]:14.3f}{self.y[i]:13.3f} "
                             + f"{self.hour[i]:02d}:"
                             + f"{self.minute[i]:02d}:"
                             + f"{self.second[i]:05.2f} "
                             + f"{self.month[i]:02d}/"
                             + f"{self.day[i]:02d}/{self.year[i]:02d}"
                             + f"{izero:11d}{self.line[i]:12d}"
                             + f"{self.mark[i]+1:12d}   5\n")
                    ncount = 0
                elif self.mark[i] != mark0:
                    mark0 = self.mark[i]
                    fh.write(f"3{self.x[i]:14.3f}{self.y[i]:13.3f} "
                             + f"{self.hour[i]:02d}:{self.minute[i]:02d}:"
                             + f"{self.second[i]:05.2f} "
                             + f"{self.month[i]:02d}/"
                             + f"{self.day[i]:02d}/{self.year[i]:02d}"
                             + f"{ncount:11d}{self.line[i+1]:12d}"
                             + f"{self.mark[i+1]:12d}   4\n")
                    ncount = 0
                fh.write(f"0 {self.sensor1[i]:11.3f}"
                         + f"{self.sensor2[i]:11.3f} "
                         + f"{self.hour[i]:02d}:{self.minute[i]:02d}:"
                         + f"{self.second[i]:05.2f} {self.month[i]:02d}/"
                         + f"{self.day[i]:02d}/{self.year[i]:02d}{izero:4d}\n")
                ncount += 1
            fh.write(f"3{self.x[0]:14.3f}{self.y[0]:13.3f} "
                     + f"{self.hour[0]:02d}:{self.minute[0]:02d}:"
                     + f"{self.second[0]:05.2f} {self.month[0]:02d}/"
                     + f"{self.day[0]:02d}/{self.year[0]:02d}"
                     + f"{ncount:11d}{self.line[0]:12d}"
                     + f"{self.mark[0]:12d}  36\n")
        print(f"\nfile {file} written")

    def write_dat(self, file):
        """
        Writes Geometrics magnetic gradiometer file in *.dat (Surfer) format.

        Parameters
        ----------
        file : str
            File where to store the data.
        """
        counter = np.arange(len(self.x))
        ye = self.year[0]
        ye1000 = int(ye/100.)*100
        ye -= ye1000
        mo = self.month[0]
        d = self.day[0]
        datum = f"{mo:02d}/{d:02d}/{ye:02d}"
        with open(file, "w", encoding="utf-8") as fo:
            if np.isclose(np.std(self.z), 0.):
                fo.write(
                    "           X           Y     TOP_RDG  "
                    + "BOTTOM_RDG        TIME      DATE         LINE"
                    + "        MARK\n")
                for i in counter[::-1]:
                    fo.write(f"{self.x[i]:12.3f}{self.y[i]:12.3f}"
                             + f"{self.sensor1[i]:12.3f}"
                             + f"{self.sensor2[i]:12.3f} {self.hour[i]:02d}:"
                             + f"{self.minute[i]:02d}:{self.second[i]:05.2f}  "
                             + f"{datum}{self.line[i]:13d}"
                             + f"{self.mark[i]:12d}\n")
            else:
                fo.write(
                    "           X           Y           Z     TOP_RDG "
                    + " BOTTOM_RDG        TIME      DATE         LINE"
                    + "        MARK\n")
                for i in counter[::-1]:
                    fo.write(f"{self.x[i]:12.3f}{self.y[i]:12.3f}"
                             + f"{self.z[i]:12.3f}{self.sensor1[1]:12.3f} "
                             + f"{self.sensor2[i]:12.3f}"
                             + f"{self.hour[i]:02d}:{self.minute[i]:02d}:"
                             + f"{self.second[i]:05.2f}  {datum}"
                             + f"{self.line[i]:13d}{self.mark[i]:12d}\n")
        print(f"\nfile {file} written")

    def clean_data(self, min_fix=None, max_fix=None, percent_down=None,
                   percent_up=None):
        """
        Set data to np.nan under certain conditions which may be:

        Parameters
        ----------
        min_fix : float. Optional (default: None)
            All data below this value are set to nan
        max_fix : float. Optional (default: None)
            All data above this value are set to nan
        percent_down : float. Optional (default: None)
            The lowermost percentile values are set to nan. A value of 0.01
            means that all values lower than the 1% quantile are set to None
        percent_up : float. Optional (default: None)
            The uppermost percentile values are set to nan. A value of 0.01
            means that all values higher than the 99% quantile are set to None

        The operation is done in situ.
        """
        print("\n")
        if min_fix:
            self.sensor1[self.sensor1 < min_fix] = np.nan
            if self.grad_data:
                self.sensor2[self.sensor2 < min_fix] = np.nan
            print(f"Clip below {np.round(min_fix,1)}")
        if max_fix:
            self.sensor1[self.sensor1 > max_fix] = np.nan
            if self.grad_data:
                self.sensor2[self.sensor2 > max_fix] = np.nan
            print(f"Clip above {np.round(max_fix,1)}")
        if percent_down:
            vmin1 = np.nanquantile(self.sensor1, percent_down)
            self.sensor1[self.sensor1 < vmin1] = np.nan
            if self.grad_data:
                vmin2 = np.nanquantile(self.sensor2, percent_down)
                self.sensor2[self.sensor2 < vmin2] = np.nan
                print(
                    f"Clip below {np.round(vmin1,1)} for sensor 1 and "
                    + f"{np.round(vmin2,1)} for sensor 2")
            else:
                print(f"Clip below {np.round(vmin1,1)}")
        if percent_up:
            vmax1 = np.nanquantile(self.sensor1, 1 - percent_up)
            self.sensor1[self.sensor1 > vmax1] = np.nan
            if self.grad_data:
                vmax2 = np.nanquantile(self.sensor2, 1 - percent_up)
                self.sensor2[self.sensor2 > vmax2] = np.nan
                print(
                    f"Clip above {np.round(vmax1,1)} for sensor 1 and "
                    + f"{np.round(vmax2,1)} for sensor 2")
            else:
                print(f"Clip above {np.round(vmax1,1)}")
        if self.grad_data:
            self.grad = (self.sensor1 - self.sensor2) / self.d_sensor

    def line_days(self):
        """
        Create a dictionary containing for every day of data acquisition the
        numbers of the lines having been measured.

        Returns
        -------
        self.lines_per_day : dict
            Dictionary has as key the number of the day. It is implicitely
            supposed that the days are unique, i.e. measurements were not done
            over a period longer than a month. For every entrance, a list
            contains the line numbers

        """
        days = np.unique(self.day)
        for d in days:
            self.lines_per_day[d] = []
            for key, entries in self.segments.items():
                if self.day[entries["mark_samples"][0]] == d:
                    self.lines_per_day[d].append(key)

    def get_line(self, i_line=0):
        """
        Get data of one single line

        Parameters
        ----------
        i_line : int, optional. Default is 0.
            Number of line to be extracted (counting starts at 0)

        Returns
        -------
        sensor1: numpy float array
            Data of sensor 1.
        sensor2: numpy float array
            Data of sensor 2.
        x: numpy float array
            X-coordinate of all data points extracted.
        y: numpy float array
            Y-coordinate of all data points extracted.
        year: numpy int array
            year of acquisition of all data points extracted.
        month: numpy int array
            month of acquisition of all data points extracted.
        day: numpy int array
            day of acquisition of all data points extracted.
        hour: numpy int array
            hour of acquisition of all data points extracted.
        minute: numpy int array
            minute of acquisition of all data points extracted.
        second: numpy float array
            second of acquisition of all data points extracted.
        time: numpy int array
            second of day of acquisition of all data points extracted.

        """
        n1 = self.segments[i_line]["mark_samples"][0]
        n2 = self.segments[i_line]["mark_samples"][-1]
        if self.grad_data:
            return (self.sensor1[n1:n2], self.sensor2[n1:n2], self.x[n1:n2],
                    self.y[n1:n2], self.year[n1:n2], self.month[n1:n2],
                    self.day[n1:n2], self.hour[n1:n2], self.minute[n1:n2],
                    self.second[n1:n2], self.time[n1:n2])
        dum = np.array([0])
        return (self.sensor1[n1:n2], dum, self.x[n1:n2], self.y[n1:n2],
                self.year[n1:n2], self.month[n1:n2], self.day[n1:n2],
                self.hour[n1:n2], self.minute[n1:n2], self.second[n1:n2],
                self.time[n1:n2])

    def geometrics_lines(self):
        """
        Put all data into a simplified dictionary, one entry per line

        No input parameters

        Returns
        -------
        data: dictionary with one entry per line, key = number of line.
              Each entry is itself a dictionary containing the following
              entries:

              - "s1": Numpy float array with data of sensor 1
              - "s2": Numpy float array with data of sensor 2
                If only data of one single sensor were measured,
                "s2" contains only one zero.
              - "x":  E-W coordinates of data points
              - "y"   N-S coordinates of data points

        """
        data = {}
        for i in range(self.n_lines):
            s1, s2, x, y, _, _, _, _, _, _, _ = self.get_line(i)
            data[i] = {}
            data[i]["s1"] = s1
            data[i]["s2"] = s2
            data[i]["x"] = x
            data[i]["y"] = y
            return data

    def get_segment(self, i_line=0, i_seg=0):
        """
        Get data of one segment (data between two marks) of a line

        Parameters
        ----------
        i_line : int, optional. Default is 0.
            Number of line to be extracted (counting starts at 0).
        i_seg : int, optional. TDefault is 0.
            Number of segment within line to be extracted
            (counting starts at 0).

        Returns
        -------
        sensor1: numpy float array
            Data of sensor 1.
        sensor2: numpy float array
            Data of sensor 2.
        x: numpy float array
            X-coordinate of all data points extracted.
        y: numpy float array
            Y-coordinate of all data points extracted.
        year: numpy int array
            year of acquisition of all data points extracted.
        month: numpy int array
            month of acquisition of all data points extracted.
        day: numpy int array
            day of acquisition of all data points extracted.
        hour: numpy int array
            hour of acquisition of all data points extracted.
        minute: numpy int array
            minute of acquisition of all data points extracted.
        second: numpy float array
            second of acquisition of all data points extracted.
        time: numpy int array
            second of day of acquisition of all data points extracted.

        """

        n1 = self.segments[i_line]["mark_samples"][i_seg]
        n2 = self.segments[i_line]["mark_samples"][i_seg + 1]
        if self.grad_data:
            return (self.sensor1[n1:n2], self.sensor2[n1:n2], self.x[n1:n2],
                    self.y[n1:n2], self.year[n1:n2], self.month[n1:n2],
                    self.day[n1:n2], self.hour[n1:n2], self.minute[n1:n2],
                    self.second[n1:n2], self.time[n1:n2])
        dum = np.array([0])
        return (self.sensor1[n1:n2], dum, self.x[n1:n2], self.y[n1:n2],
                self.year[n1:n2], self.month[n1:n2], self.day[n1:n2],
                self.hour[n1:n2], self.minute[n1:n2], self.second[n1:n2],
                self.time[n1:n2])

    def read_base(self, infile, year=None):
        """
        Read data from a base station. It is supposed the base station is a
        Geometrics PPM station. The function recognizes automatically whether
        it is a G-856 (old version) or a G-857 (newer version) instrument

        If other data have already been read in, the new ones are concatenated
        to the existing data. Before leaving the subroutine, data are sorted by
        measurement time. In this way, the order of data storage within a file
        and the order of reading of different files are without importance.

        Parameters
        ----------
        infile : str
            Name of file containing base station data.
        year : int, optional
            If base station data were recorded with a G-856 PPM, the date is
            coded as Julian day without the year of acquisition. In this case,
            the year should be given. If Geometrics G-858 (Cs magnetometer)
            data have been input already, the year of those data may be used
            and the variable may be left "None". The default is None.

        Data are stored in the following arrays (usually, they are base station
                                                 data):

        - self.base : 1D numpy float array contains all base station data
        - self.day_base   : 1D numpy int array contains day of acquisition for
          all data
        - self.month_base : 1D numpy int array contains month of acquisition
          for all data
        - self.year_base  : 1D numpy int array contains year of acquisition for
          all data
        - self.hour_base  : 1D numpy int array contains hour of acquisition for
          all data
        - self.minute_base: 1D numpy int array contains minute of acquisition
          for all data
        - self.second_base: 1D numpy float array contains second of acquisition
          for all data
        - self.time_base  : 1D numpy float array contains second within year
          for all data

        """

        if not isinstance(year, (int, np.int32)):
            year = self.year[0]

        with open(infile, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
        nlines = len(lines)
        self.year_base = list(self.year_base)
        self.month_base = list(self.month_base)
        self.day_base = list(self.day_base)
        self.jday_base = list(self.jday_base)
        self.hour_base = list(self.hour_base)
        self.minute_base = list(self.minute_base)
        self.second_base = list(self.second_base)
        self.base = list(self.base)
        self.time_base = list(self.time_base)
        if lines[0][0] == "*":
            for k in range(nlines):
                line = lines[k].split()
                try:
                    j_day = int(line[2])
                except IndexError:
                    break
                d, m = utils.julian2date(j_day, year)
                self.jday_base.append(j_day)
                self.year_base.append(year)
                self.month_base.append(m)
                self.day_base.append(d)
                self.hour_base.append(int(line[3][0:2]))
                self.minute_base.append(int(line[3][2:4]))
                self.second_base.append(float(line[3][4:6]))
                self.time_base.append(j_day)
                if len(line) == 5:
                    self.base.append(float(line[-1].split("?")[1]) / 10.0)
                else:
                    self.base.append(float(line[-1]) / 10.0)
        elif lines[0][0] == "9":
            for lin in lines[1:]:
                line = lin.split()
                self.base.append(float(line[1]))
                nums = line[3].split(":")
                self.hour_base.append(int(nums[0]))
                self.minute_base.append(int(nums[1]))
                self.second_base.append(float(nums[2]))
                nums = line[4].split("/")
                self.month_base.append(int(nums[0]))
                self.day_base.append(int(nums[1]))
                self.year_base.append(int(nums[2]))
                self.jday_base.append(
                    utils.date2julian(self.day_base[-1], self.month_base[-1],
                                      self.year_base[-1]))
                self.time_base.append(self.jday_base[-1])
        else:
            for k in range(2, nlines):
                line = lines[k].split(",")
                try:
                    self.month_base.append(int(line[2][0:2]))
                except IndexError:
                    break
                self.day_base.append(int(line[2][3:5]))
                self.year_base.append(int(line[2][6:8]))
                self.hour_base.append(int(line[3][0:2]))
                self.jday_base.append(utils.date2julian(
                    self.day_base[-1], self.month_base[-1],
                    self.year_base[-1]))
                self.minute_base.append(int(line[3][3:5]))
                self.second_base.append(float(line[3][6:8]))
                self.time_base.append(self.jday_base[-1])
                self.base.append(float(line[6]))
        self.month_base = np.array(self.month_base, dtype=int)
        self.day_base = np.array(self.day_base, dtype=int)
        self.year_base = np.array(self.year_base, dtype=int)
        self.hour_base = np.array(self.hour_base, dtype=int)
        self.minute_base = np.array(self.minute_base, dtype=int)
        self.second_base = np.array(self.second_base, dtype=float)
        self.base = np.array(self.base, dtype=float)
        self.time_base = (
            np.array(self.time_base) * 86400.0
            + (self.hour_base * 60.0 + self.minute_base) * 60.0
            + self.second_base)
        index = np.argsort(self.time_base)
        self.month_base = self.month_base[index]
        self.day_base = self.day_base[index]
        self.year_base = self.year_base[index]
        self.hour_base = self.hour_base[index]
        self.minute_base = self.minute_base[index]
        self.second_base = self.second_base[index]
        self.base = self.base[index]
        self.time_base = self.time_base[index]

    def write_base(self, outfile):
        """
        Write base station data in G-856 format.

        Parameters
        ----------
        outfile : str
            Name of file where to write the base station data.

        """
        with open(outfile, "w", encoding="utf-8") as fo:
            for i, b in enumerate(self.base):
                if np.isnan(b):
                    continue
                fo.write(
                    f"*  0 {self.jday_base[i]:3d} {self.hour_base[i]:02d}"
                    + f"{self.minute_base[i]:02d}"
                    + f"{int(self.second_base[i]):02d}"
                    + f"{i:5d}{int(b*10):7d}\n")