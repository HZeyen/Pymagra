# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 2023
last modified on Aug 14, 2024

@author: Hermann Zeyen <hermann.zeyen@universite-paris-saclay.fr>
         University Paris-Saclay, France

Needed Python packages:
    PyQt5
    matplotlib
    numpy
    os
    sys
    signal
    sklearn
    scipy

Needed private files:

    Geometrics.py
    Magnetics.ui


Contains the following class:
    Main : Main class, controlling inputs, data treatment and outputs

"""

# 7import sys
import os
from copy import deepcopy
from datetime import datetime, date
from signal import signal, SIGINT
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib import colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.linear_model import LinearRegression as LR
# The next two lines may have to be modified:
#  sys_path is the folder where all python program files are located
#  dir0 is the folder where the data are located

# Paths on HZ computer
# sys_path = r"E:/Sources_2010/Python_programs/"
# dir0 = r"D:/Tadjikistan/ZolIZard/"
# dir0 = r"E:/Daten/Magnetics/Test_Campus_haut"
# dir0 = r"E:/Daten/Fontaines_Salées_BRGM"
# dir0 = r"E:/Documents/Cours/Stages/M1/2022-23/Magnetics/2022-09-20_day2_field5"
# dir0 = r"D:/Documents/Cours/Stages/M1/2023-24/Magnetics/2023-09-26"
# dir0 = r"E:/Seg2Dat/Garchy sismique 2024/Magnetisme"

# Example of paths for programs and data on department desktop
# sys_path = r"C:/Users/Utilisateur/Desktop/Geophysique/Sismique"
# dir0 = r"C:/Users/Utilisateur/Desktop/Geophysique/magnetique"

# Example of paths for Linux
# sys_path = r"/home/zeyen/src/Python"
# dir0 = r"/home/zeyen/magnetics"

# if sys_path not in sys.path:
#    sys.path.append(sys_path)

import numpy as np

# import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, QtCore

# from PyQt5.QtCore import Qt
# from PyQt5.QtWidgets import (QRadioButton, QButtonGroup)
# s_path = r"E:/Sources_2010/Python_programs/Pymagra"
from .in_out import io
from .plotting import plot
from .plotting import newWindow

# from .in_out.geometrics import Geometrics as geom
from .utilities import Utilities
from .in_out import dialog

# from .utilities import Utilities
# from .plotting import plot


class Main(QtWidgets.QWidget):
    """
    Main class for PyMaGra Program

    Parameters
    ----------
    dir0 : str (default = None)
        name of working folder

    Methods
    -------
    __init__ : Initialization
    readdata : Input data sets
    getGeography : Get geography information (towns, geological borders...)
    oneOfTwo : Extract one line out of two (for lines in alternating directions)
    readBaseData : Input base station data
    writeStn : Write treated data in Geometrics .stn format
    writeDat : Write treated data in Geometrics Surfer format
    saveBaseData : Write base station data into Geometrics .str file
    original : Reset actual data to the original ones
    plotOriginal : Plot original data set (but keep treated data as actual ones)
    plotActual : Plot actual, treated data to screen
    plot_gradient : Toggle on/off plotting of gradient data if there are
    plot_geography : Toggle on/off plotting of geography data
    plot_lineaments : Toggle on/off plotting of measured lineaments
    plot_Line : Plot a lineament onto the data map
    get_mouse_click (with method onClick); Wait for mouse click return position and type
    setHelp : Set help text written at the base of the screen
    changeColorScale : Change parameters for Color scale
    plotBase : Plot base station data
    plotMedian : Plot medians of all lines
    zooming : (TODO)
    zoomOut : (TODO)
    zoomIn : (TODO)
    zoomInitial : (TODO)
    savePlot : Save actual plot into .png file
    diurnal : Estimate diurnal variations from variation of line medians
    clean : Eliminate outliers from data
    justify_median : Justify medians of lines measured in opposite directions
    justify_gauss : Justify gaussian distribution of lines in opposite directions
    interpol : Interpolate data onto a regular grid
    nanFill : Interpolate data at positions have nan-value
    reducePole : Reduce magnetic data to the pole
    log_spect : Calculate lograrithmic spectrum of a data series
    fit2lines : Fit two slopes to a data series
    spector_line : Fit lines to lograithmic spectrum and calculate source depth
    spector : Calculate 1D source depths from data spectrum
    spector2D : Calculate 1D source depths from data spectrum on a 2D grid
    tilt : Calculate tilt angle of 2D data set
    continuation : Continue potential field data set upwards
    analyticSignal : Calculate analytic signal and source depths
    followLine : Follow the cursor and plot a straight line
    eventFilter : (TODO) Event filter for keyboard strokes
    Handler : Should handle exeptions - not really working
    dialog : Wrapper for general dialog interface (calls class Dialog)
    save_lineaments : Saves measured data lineaments into file
    closeApp : Close application in an ordered way
    """

    def __init__(self, dir0=None):
        super().__init__()

        self.function = "main"
        self.data_read = False
        self.percent = 0.01
        self.mincol1 = 0.0
        self.maxcol1 = 0.0
        self.mincol2 = 0.0
        self.maxcol2 = 0.0
        self.mincolg = 0.0
        self.maxcolg = 0.0
        self.d_sensor = 0.9
        self.h_sensor = 0.4
        self.height = 1.3
        self.gradient_flag = True
        self.inter_flag = False
        self.n_blocks = 0
        self.grid_flag = True
        self.nlineaments = 0
        self.color = "rainbow"
        self.lineaments = {}
        self.treatments = {}
        self.treatments["diurnal"] = False
        self.treatments["clip"] = False
        self.treatments["justify_median"] = False
        self.treatments["justify_Gauss"] = False
        self.treatments["interpol"] = False
        self.treatments["nan_fill"] = False
        self.treatments["pole"] = False
        self.treatments["odd lines"] = False
        self.treatments["even lines"] = False
        self.treatments["up"] = False
        self.actual_plotted_file = 0
        self.u = Utilities(self)
        self.line_choice = "all"
        self.dir0 = dir0
        self.fig_base = None
        self.inclination = 62.0
        self.declination = 0.0
        self.geography = {}
        self.base = None
        self.fig_line = None
        self.ax_line = None
        self.ax_base = None
        self.fig_median = None
        self.ax_median = None
        self.histo = None
        self.ax_histo = None
        self.fig_spector = None
        self.ax_spector = None
        self.fig_FFT = None
        self.ax_FFT = None
        self.fig_spect2 = None
        self.ax_spect2 = None
        self.fig_FFT2 = None
        self.ax_FFT2 = None
        self.fig_grad = None
        self.ax_grad = None
        self.fig_tilt = None
        self.ax_tilt = None
        self.fig_ana = None
        self.ax_ana = None
        self.fig_q = None
        self.ax_q = None
        self.fig_sig = None
        self.ax_sig = None
        self.wait = True
        self.click = False
        self.press = False
        self.event = None
        self.x_event = None
        self.y_event = None
        self.coor_x = 0.0
        self.coor_y = 0.0
        self.line = None
        self.released = False
        self.line_click = None
        self.x_mouse = 0.0
        self.y_mouse = 0.0
        self.mouse = None
        self.xmin = 0.0
        self.ymin = 0.0
        self.xwin = 0.0
        self.ywin = 0.0
        self.day_joint_flag = False
        self.diff_weight = 1.0
        self.dx = 0.0
        self.dy = 0.0
        self.dz = 0.0
        self.sensor1_inter = []
        self.sensor2_inter = []
        self.grad_inter = []
        self.x_inter = []
        self.y_inter = []
        self.z_inter = []
        self.sensor1_fill = []
        self.sensor2_fill = []
        self.grad_fill = []
        self.mask1 = []
        self.mask2 = []
        self.start = [0.0, 0.0]
        self.side = 0
        self.background = None
        self.canvas = None
        self.axl = None
        self.cidmotion = None
        self.cidrelease = None
        self.cidpress = None
        # string_keys gives the number of keys in data that are not numbers
        self.string_keys = 0
        #
        self.w = plot(self)

        # Input data
        self.readData(dir0)
        x = self.dat[0].data[0]["x"]
        y = self.dat[0].data[0]["y"]
        dx = x.max() - x.min()
        dy = y.max() - y.min()
        if dx > dy:
            self.direction = 1
        else:
            self.direction = 0
        if "m" not in self.data_types[0]:
            self.w.save_base.setEnabled(False)
            self.w.readBase.setEnabled(False)
            self.w.basePlot.setEnabled(False)
            self.w.plotGradient.setEnabled(False)
            self.w.diurnalCorrection.setEnabled(False)
            self.w.medianJustify.setEnabled(False)
            self.w.poleReduction.setEnabled(False)

        self.help = QtWidgets.QLabel(self)
        QtWidgets.qApp.installEventFilter(self)

        # Define actions for Menu buttons
        # Actions for menu File
        self.w.addData.triggered.connect(self.readData)
        self.w.saveSTN.triggered.connect(self.writeStn)
        self.w.saveDat.triggered.connect(self.writeDat)
        self.w.readBase.triggered.connect(self.readBaseData)
        self.w.save_base.triggered.connect(self.saveBaseData)
        self.w.Save_plot.triggered.connect(self.savePlot)
        self.w.geography.triggered.connect(self.getGeography)
        self.w.quitAction.triggered.connect(self.close_app)
        # Actions for menu Display
        self.w.originalPlot.triggered.connect(self.plotOriginal)
        self.w.actualPlot.triggered.connect(self.plotActual)
        self.w.plotLine.triggered.connect(self.plot_Line)
        self.w.basePlot.triggered.connect(self.plotBase)
        self.w.medianPlot.triggered.connect(self.plotMedian)
        self.w.plotGradient.triggered.connect(self.plot_gradient)
        self.w.plotGeo.triggered.connect(self.plot_geography)
        self.w.plotGrid.triggered.connect(self.plot_grid)
        self.w.plotLineaments.triggered.connect(self.plot_lineaments)
        # self.w.zoom.triggered.connect(self.zooming)
        # self.w.zoom_Out.triggered.connect(self.zoomOut)
        # self.w.zoom_In.triggered.connect(self.zoomIn)
        # self.w.zoom_Initial.triggered.connect(self.zoomInitial)
        self.w.changeQuantile.triggered.connect(self.changeColorScale)
        self.w.secondLine.triggered.connect(self.oneOfTwo)
        # Actions for menu Utilities
        self.w.originalData.triggered.connect(self.original)
        self.w.diurnalCorrection.triggered.connect(self.diurnal)
        self.w.cleanData.triggered.connect(self.clean)
        self.w.medianJustify.triggered.connect(self.justify_median)
        self.w.gaussJustify.triggered.connect(self.justify_gauss)
        self.w.interpolate.triggered.connect(self.interpol)
        self.w.fill.triggered.connect(self.nanFill)
        self.w.poleReduction.triggered.connect(self.reducePole)
        self.w.lineFFT.triggered.connect(self.spector)
        self.w.Spector_2D.triggered.connect(self.spector2D)
        self.w.tiltAngle.triggered.connect(self.tilt)
        self.w.prolongation.triggered.connect(self.continuation)
        self.w.analytic.triggered.connect(self.analyticSignal)

        # Check whether file "lineaments.dat" exists containing magnetic or gravity
        # lineaments picked from tilt angle maps. If so, create dictionary xith all
        # lineament information.
        if os.path.isfile("lineaments.dat"):
            with open("lineaments.dat", "r", encoding="utf-8") as fi:
                lines = fi.readlines()
            il = 0
            self.nlineaments = 0
            x = []
            y = []
            while True:
                if lines[il][0] == "#":
                    if len(x) > 0:
                        self.lineaments[self.nlineaments]["x"] = np.array(x)
                        self.lineaments[self.nlineaments]["y"] = np.array(y)
                        x = []
                        y = []
                    if lines[il][:4] == "#END":
                        break
                    self.nlineaments += 1
                    self.lineaments[self.nlineaments] = {}
                    self.lineaments[self.nlineaments]["type"] = lines[il][1:-1]
                    il += 1
                else:
                    nums = lines[il].split()
                    x.append(float(nums[0]))
                    y.append(float(nums[1]))
                    il += 1
            self.w.plotLineaments.setEnabled(True)
            self.w.plotLin_flag = True
            self.w.lineaments = self.lineaments
        # Intercept CTL-C to exit in a controlled way
        signal(SIGINT, self.Handler)
        self.w.grad_flag = False
        self.grad_flag = False
        self.gradient_flag = False
        if self.data_types[self.actual_plotted_file] == "magnetic":
            self.unit = "nT"
        else:
            self.unit = "mGal"
        # Plot data
        self.fig, self.ax = self.w.plot_triang(
            self.dat[self.actual_plotted_file].data,
            title=f"Original measured {self.data_types[0]} data",
            percent=self.percent,
            mincol1=self.mincol1,
            maxcol1=self.maxcol1,
            mincol2=self.mincol2,
            maxcol2=self.maxcol2,
            mincolg=self.mincolg,
            maxcolg=self.maxcolg,
            grad_flag=self.gradient_flag,
        )

    def readData(self, dir0):
        """
        Reads additional data files to the ones read during init of Geometrics
        normally, Geometrics stn files should be read, but another format is
        also possible, considered to be output of program mgwin.
        See Geometrics.read_txt for more information

        Returns
        -------
        None.

        """
        self.data_files, self.file_types, self.data_types, self.dir0 = io.get_files(
            dir0
        )
        self.base_files = []
        self.dat = []
        for i, f in enumerate(self.data_files):
            self.dat.append(io.Data())
            if self.file_types[i] == "GEOMETRICS":
                self.dat[-1].read_geometrics(f)
                self.w.saveSTN.setEnabled(True)
                self.w.fill.setEnabled(True)
                self.w.poleReduction.setEnabled(True)
            elif self.file_types[i] == "GXF":
                self.dat[-1].read_gxf(f)
                self.gradient_flag = False
                self.w.fill.setEnabled(True)
                self.w.plotGradient.setEnabled(False)
                if self.data_types[i] == "magnetic":
                    self.w.poleReduction.setEnabled(True)
            elif self.file_types[i] == "BRGM":
                self.dat[-1].read_BRGM_flight(f)
                self.gradient_flag = False
                self.w.fill.setEnabled(True)
                self.w.plotGradient.setEnabled(False)
                self.w.poleReduction.setEnabled(True)
            self.n_blocks += 1
            self.dat[-1].data["type"] = self.data_types[i]
            if "magnetic" in self.data_types:
                self.inclination, self.declination = io.get_mag_field()
                self.inclination_ori = self.inclination
                self.declination_ori = self.declination
        self.dat_ori = deepcopy(self.dat)
        data_keys = list(self.dat[0].data.keys())
        self.height = self.dat[0].data["height"]
        self.string_keys = 0
        for i, k in enumerate(data_keys):
            if isinstance(k, (str)):
                break
        self.string_keys = len(data_keys) - i

    def getGeography(self):
        """
        Ask for file containing geography data, read data and plot them onto
        data map.
        
        Geography file has the following form:

        #Keyword (may be "#LINE", "#POINT" or "#END")
        If "#POINT", one line follows with x y text \
        If "#LINE", one line follows for every point defining the line x y

        The file must be finished with a line containing #END in the first
        four columns.
        Usually, points are towns and text the name of the town
        x and y must be given in the same coordinate system as the data.

        Returns
        -------
        None.

        """
        files = list(
            QtWidgets.QFileDialog.getOpenFileNames(
                None,
                "Select geography data file",
                "",
                filter="txt (*.txt) ;; all (*.*)",
            )
        )
        if len(files) == 0:
            print("No file chosen, try again")
        if len(files[0]) < 1:
            print("getGeograpgy: No files read")
            return
        self.geography = io.read_geography_file(files[0][0])
        self.w.set_geography_flag(True)
        self.w.plotGeo.setEnabled(True)
        self.plotActual(self.dat[self.actual_plotted_file].data)

    def oneOfTwo(self):
        """
        Extract every second line starting with line 1 (odd lines) or line 2 (even lines)
        May be useful is there is a strong directional effect in magnetic data

        Returns
        -------
        None.

        """
        if self.treatments["odd lines"]:
            choice = ["even", "all"]
        elif self.treatments["even lines"]:
            choice = ["odd", "all"]
        else:
            choice = ["odd", "even"]
        results, okButton = dialog(
            ["Choose data of", choice], ["l", "r"], [None, 0], "Extract lines"
        )
        if okButton:
            c = int(results[1])
            if self.treatments["odd lines"]:
                self.treatments["odd lines"] = False
                self.u.extract(choice[c])
                if c == 0:
                    self.treatments["even lines"] = True
                else:
                    self.treatments["even lines"] = False
            elif self.treatments["even lines"]:
                self.treatments["even lines"] = False
                if c == 0:
                    self.treatments["odd lines"] = True
                else:
                    self.treatments["odd lines"] = False
            else:
                if c == 0:
                    self.treatments["odd lines"] = True
                else:
                    self.treatments["even lines"] = True
            self.u.extract(choice[c])
        else:
            return
        self.plotActual(self.dat[self.actual_plotted_file].data)

    def readBaseData(self):
        """
        Read base station data. These data must be Geometrics PPM files

        Returns
        -------
        None.

        """
        data_files, _, _, _ = io.get_files(ftype="base")
        if len(data_files) > 0:
            for f in data_files:
                self.base_files.append(f)
                self.base = io.Data()
                self.base.read_base(f, self.dat[self.actual_plotted_file].data["year"])
            self.w.basePlot.setEnabled(True)
            self.plotBase()

    def getDateString(self):
        """
        Gets date and time of execution of the function

        Returns
        -------
        date and time: str
            Date and time are stored as YYYY-MM-DD_hh-mm-ss

        Is used to define output names where the text is standard but time of
        creation allows distinguishing.

        """
        now = datetime.now()
        c_time = now.strftime("%H-%M-%S")
        today = date.today()
        d = today.strftime("%Y-%m-%d")
        return f"{d}_{c_time}"

    def writeStn(self):
        """
        Writes actual (last treated interpolated data) to Geometrics stn file.
        The file name is "data-type"followed by date and hour of file production

        Returns
        -------
        None.

        """
        fname = (
            f"{self.dat[self.actual_plotted_file].data['type']}_"
            + f"{self.getDateString()}.stn"
        )
        try:
            isinstance(self.sensor1_inter, np.ndarray)
            if self.dat[self.actual_plotted_file].data["grad_data"]:
                self.dat[self.actual_plotted_file].write_geometrics(
                    fname,
                    self.sensor1_inter,
                    self.x_inter,
                    self.y_inter,
                    data2=self.sensor2_inter,
                )
            else:
                self.dat[self.actual_plotted_file].write_geometrics(
                    fname, self.sensor1_inter, self.x_inter, self.y_inter
                )
        except AttributeError:
            self.dat[self.actual_plotted_file].write_geometrics(
                fname, self.dat[self.actual_plotted_file].data, [None], [None]
            )

    def writeDat(self):
        """
        Writes actual (last treated not necessarily interpolated data) to
        Geometrics Surfer (.dat) format file.
        The file name is "data-type"followed by date and hour of file production

        Returns
        -------
        None.

        """
        fname = (
            f"{self.dat[self.actual_plotted_file].data['type']}_"
            + f"{self.getDateString()}.dat"
        )
        try:
            isinstance(self.sensor1_inter, np.ndarray)
            if self.dat[self.actual_plotted_file].data["grad_data"]:
                self.dat[self.actual_plotted_file].write_dat(
                    fname,
                    self.sensor1_inter,
                    self.x_inter,
                    self.y_inter,
                    data2=self.sensor2_inter,
                    z=self.z_inter,
                )
            else:
                self.dat[self.actual_plotted_file].write_dat(
                    fname,
                    self.sensor1_inter,
                    self.x_inter,
                    self.y_inter,
                    z=self.z_inter,
                )
        except AttributeError:
            self.dat[self.actual_plotted_file].write_dat(
                fname, self.dat[self.actual_plotted_file].data, [None], [None]
            )

    def saveBaseData(self):
        """
        Writes base station data to Geometrics G-856 .stn format file.
        The file name is "base"followed by date and hour of file production

        Returns
        -------
        None.

        """
        fname = f"base_{self.getDateString()}.stn"
        self.base.write_base(fname)

    def original(self):
        """
        Recovers original (not interpolated) data to restart data treatment
        procedure.
        If not base station data had been read, but they were produced through
        spline fit of median values, those data are deleted.

        Returns
        -------
        None.

        """
        self.dat[self.actual_plotted_file] = deepcopy(
            self.dat_ori[self.actual_plotted_file]
        )
        self.treatments["odd lines"] = False
        self.treatments["even lines"] = False
        if len(self.base_files) == 0:
            try:
                del self.w.base
                del self.w.time_base
            except AttributeError:
                pass
        try:
            del self.sensor1_inter
            del self.sensor2_inter
            del self.grad_inter
        except AttributeError:
            pass
        self.w.saveSTN.setEnabled(False)
        self.w.fill.setEnabled(False)
        self.w.poleReduction.setEnabled(False)
        self.w.tiltAngle.setEnabled(False)
        self.w.lineFFT.setEnabled(False)
        self.w.Spector_2D.setEnabled(False)
        self.w.prolongation.setEnabled(False)
        self.w.analytic.setEnabled(False)
        self.w.gaussJustify.setEnabled(False)
        self.treatments["diurnal"] = False
        self.treatments["clip"] = False
        self.treatments["justify_median"] = False
        self.treatments["justify_Gauss"] = False
        self.treatments["interpol"] = False
        self.treatments["nan_fill"] = False
        self.treatments["pole"] = False
        self.treatments["odd lines"] = False
        self.treatments["even lines"] = False
        self.treatments["up"] = False
        self.inclination = self.inclination_ori
        self.declination = self.declination_ori
        self.height = self.h_sensor + self.d_sensor
        self.inter_flag = False
        self.plotOriginal()

    def plotOriginal(self):
        """
        Plot original data set again

        Returns
        -------
        None.

        """
        data = self.dat_ori[self.actual_plotted_file].data
        self.fig, self.ax = self.w.plot_triang(
            data,
            title=f"Original measured {data['type']} data",
            percent=self.percent,
            mincol1=self.mincol1,
            maxcol1=self.maxcol1,
            mincol2=self.mincol2,
            maxcol2=self.maxcol2,
            mincolg=self.mincolg,
            maxcolg=self.maxcolg,
            grad_flag=self.gradient_flag,
            c=self.color,
        )

    def plotActual(self, data=None):
        """
        Plot actual data set modified and interpolated or not

        Returns
        -------
        None.

        """
        title = f"{self.dat[self.actual_plotted_file].data['type']} data;\nTreatments:"
        if self.inter_flag:
            for t in self.treatments.items():
                if t[1]:
                    title += " " + t[0] + ","
            if title[-1] == ":":
                title += " None"
            if title[-1] == ",":
                title = title[:-1]

            self.fig, self.ax = self.w.plot_image(
                title=title,
                percent=self.percent,
                mincol1=self.mincol1,
                maxcol1=self.maxcol1,
                mincol2=self.mincol2,
                maxcol2=self.maxcol2,
                mincolg=self.mincolg,
                maxcolg=self.maxcolg,
                grad_flag=self.gradient_flag,
                c=self.color,
                dec=data["line_declination"],
            )
        else:
            if not data:
                data = self.dat[self.actual_plotted_file].data
            for t in self.treatments.items():
                if t[0] in ("interpol", "nan_fill", "pole", "up"):
                    continue
                if t[1]:
                    title += " " + t[0] + ","
            if title[-1] == ":":
                title += " None"
            if title[-1] == ",":
                title = title[:-1]
            self.fig, self.ax = self.w.plot_triang(
                data,
                title=title,
                percent=self.percent,
                mincol1=self.mincol1,
                maxcol1=self.maxcol1,
                mincol2=self.mincol2,
                maxcol2=self.maxcol2,
                mincolg=self.mincolg,
                maxcolg=self.maxcolg,
                grad_flag=self.gradient_flag,
                c=self.color,
            )

    def plot_gradient(self):
        """
        Activate or desactivate plot of vertical gradient maps

        Returns
        -------
        None.

        """
        self.gradient_flag = not self.gradient_flag
        if self.gradient_flag:
            self.w.plotGradient.setChecked(True)
        else:
            self.w.plotGradient.setChecked(False)
        self.plotActual(self.dat[self.actual_plotted_file].data)

    def plot_geography(self):
        """
        Activate or deactivate plot of geographical information

        Returns
        -------
        None.

        """
        self.w.toggle_geography_flag()
        self.plotActual(self.dat[self.actual_plotted_file].data)

    def plot_lineaments(self):
        """
        Activate or deactivate plot of measured lineaments

        Returns
        -------
        None.

        """
        self.w.plotLin_flag = not self.w.plotLin_flag
        self.plotActual(self.dat[self.actual_plotted_file].data)

    def plot_grid(self):
        """
        Activate or deactivate plot of grid

        Returns
        -------
        None.

        """
        self.w.grid_flag = not self.w.grid_flag
        self.grid_flag = not self.grid_flag
        self.plotActual(self.dat[self.actual_plotted_file].data)

    def plot_Line(self):
        """
        Plot one line out of the data set.
        The user is first asked to choose one line by mouse click. This line
        is shown in a floating window.
        Erroneous data may be erased by clicking the mouse wheel at the beginning
        and at the end of the zone to be erased. The program recognizes
        automatically whether first or second sensor has been clicked.
        Left mouse click changes line to the left/below, right mouse click goes
        to the line to the right/above.
        A click outside the two coordinate systems stops the module and deletes
        the flowting window.

        Returns
        -------
        None.

        """
        direction = ""
        self.setHelp("Click on line to be plotted", self.w.mplvl)
        # Wait for mouse click to choose line to be plotted first
        self.get_mouse_click(self.fig)
        xm = self.x_mouse
        ym = self.y_mouse
        ms = self.mouse
        read_flag = True
        line_positions = np.array([])
        # If interpolated data exist, use their coordinates. If not, the coordinates
        #    are defined as all unique values of x and y
        try:
            _ = self.x_inter
            inter = True
        except AttributeError:
            inter = False
        # Create figure for line plot in floating window
        self.fig_line = newWindow("Single line", 800, 500)
        while True:
            # If read_flag is True, define new line set to be plotted
            if read_flag:
                # If data set exists that has been interpolated onto a regular grid, use its
                #    data, if not original ones
                if inter:
                    # If left mouse click, plot line in Y direction
                    if ms == 1:
                        lin = np.argmin(np.abs(self.x_inter - xm))
                        s1 = self.sensor1_inter[:, lin]
                        pos = self.y_inter
                        line_positions = self.x_inter
                        index = np.argsort(pos)
                        s1 = s1[index]
                        if self.w.grad_data:
                            s2 = self.sensor2_inter[:, lin]
                            s2 = s2[index]
                        pos = pos[index]
                        pos_line = self.x_inter[lin]
                        direction = "N"
                    # IIf right mous click or wheel, plot line in X direction
                    else:
                        lin = np.argmin(np.abs(self.y_inter - ym))
                        s1 = self.sensor1_inter[lin, :]
                        pos = self.x_inter
                        line_positions = self.y_inter
                        index = np.argsort(pos)
                        s1 = s1[index]
                        if self.w.grad_data:
                            s2 = self.sensor2_inter[lin, :]
                            s2 = s2[index]
                        pos = pos[index]
                        pos_line = self.y_inter[lin]
                        direction = "E"
                # Do the same if original data are ro be used
                else:
                    data = self.dat[self.actual_plotted_file].data
                    direction = data[0]["direction"]
                    keys = list(data.keys())[: -self.string_keys]
                    pos_l = []
                    for k in keys:
                        if data[k]["direction"] in ("N", "S"):
                            pos_l.append(np.median(data[k]["x"]))
                        else:
                            pos_l.append(np.median(data[k]["y"]))
                    line_positions = np.unique(np.array(pos_l))
                    pos_l = np.array(pos_l)
                    if data[keys[0]]["direction"] in ("N", "S"):
                        lin = np.argmin(abs(line_positions - xm))
                    else:
                        lin = np.argmin(abs(line_positions - ym))
                    pos_line = line_positions[lin]
                    line_keys = []
                    pos = []
                    s1 = []
                    if self.dat[self.actual_plotted_file].data["grad_data"]:
                        s2 = []
                    for k in keys:
                        if data[k]["direction"] in ("N", "S"):
                            if np.isclose(np.median(data[k]["x"]), pos_line):
                                pos += list(data[k]["y"])
                                s1 += list(data[k]["s1"])
                                if data["grad_data"]:
                                    s2 += list(data[k]["s2"])
                                line_keys.append(k)
                        else:
                            if np.isclose(np.median(data[k]["y"]), pos_line):
                                pos += list(data[k]["x"])
                                s1 += list(data[k]["s1"])
                                if data["grad_data"]:
                                    s2 += list(data[k]["s2"])
                                line_keys.append(k)
                    index = np.argsort(pos)
                    s1 = np.array(s1)
                    s1 = s1[index]
                    if data["grad_data"]:
                        s2 = np.array(s2)
                        s2 = s2[index]
                    pos = np.array(pos)[index]
            read_flag = True
            lmax = len(line_positions) - 1
            self.function = "line"
            self.fig_line = newWindow("Single line", 800, 500)
            # If 2 sensors have been read in, create two subplots, if not, only one.
            if self.dat[self.actual_plotted_file].data["grad_data"]:
                self.ax_line = self.fig_line.fig.subplots(2, 1)
            else:
                ax = self.fig_line.fig.subplots(1, 1)
                # For simpler programming in the following part, copy single axis into a list
                self.ax_line = [ax]
            # Plot line values into first axis
            self.ax_line[0].plot(pos, s1)
            self.ax_line[0].set_title(f"Line at {pos_line:0.2f}")
            if self.data_types[self.actual_plotted_file] == "magnetic":
                if self.dat[self.actual_plotted_file].data["grad_data"]:
                    self.ax_line[0].set_ylabel("Magnetic field [nT]\nsensor 1")
                else:
                    self.ax_line[0].set_ylabel("Magnetic field [nT]")
            else:
                if self.dat[self.actual_plotted_file].data["grad_data"]:
                    self.ax_line[0].set_ylabel("Gravity field [mGal]\nsensor 1")
                else:
                    self.ax_line[0].set_ylabel("Gravity field [mGal]")
            # if two sensors were used, plot values of second sensor into second axis
            if self.dat[self.actual_plotted_file].data["grad_data"]:
                self.ax_line[1].plot(pos, s2)
                self.ax_line[1].set_xlabel("Distance [m]")
                if self.data_types[self.actual_plotted_file] == "magnetic":
                    self.ax_line[1].set_ylabel("Magnetic field [nT]\nsensor 2")
                else:
                    self.ax_line[1].set_ylabel("Gravity field [mGal]\nsensor 2")
            self.setHelp(
                "Click left/right for next line to left/right, "
                + "wheel inside axis to erase data (2 points, start&end), "
                + "click outside axis to stop",
                self.fig_line,
            )
            self.fig_line.show()
            # Wait for mouse click in order to determine the next step:
            #      Right click: plot next line to the East or North
            #      Left click: plot next line to the West or South
            #      Wheel allows eliminating manually erroneous data (click at left and
            #            right of zone to be muted)
            #      If the click was done outside of the axes, finish line plotting
            self.get_mouse_click(self.fig_line.fig)
            # Click was outside axes
            if not self.event.inaxes:
                self.fig_line.close()
                self.function = "main"
                self.plotActual(self.dat[self.actual_plotted_file].data)
                return
            # Left click inside an axis
            if self.mouse == 1:
                lin = max(0, lin - 1)
                if direction in ("N", "S"):
                    xm = line_positions[lin]
                else:
                    ym = line_positions[lin]
            # Right click inside an axis
            elif self.mouse == 3:
                lin = min(lmax, lin + 1)
                if direction in ("N", "S"):
                    xm = line_positions[lin]
                else:
                    ym = line_positions[lin]
            # Wheel was clicked. The click is interpreted as one of the limits of the
            #       zone to be muted
            else:
                print("click1:", self.mouse, self.x_mouse)
                x0 = self.x_mouse
                # Wait for next mouse click.
                self.get_mouse_click(self.fig_line.fig)
                print("click2:", self.mouse, self.x_mouse)
                # x0 = self.x_mouse
                x1 = self.x_mouse
                # Data between the two mouse clicks are muted.
                # If the mouse click was detected at y > 200, mouse was in the upper subplot and
                #    data of sensor 1 are muted. Else data of sensor 2 are muted.
                index = np.where((pos > min(x0, x1)) & (pos < max(x0, x1)))
                if self.ywin > 200 or not data["grad_data"]:
                    s1[index] = np.nan
                    if inter:
                        if direction in ("N", "S"):
                            self.sensor1_inter[:, lin] = s1
                        else:
                            self.sensor1_inter[lin, :] = s1
                    else:
                        i1 = 0
                        for lin in line_keys:
                            i2 = i1 + len(data[lin]["x"])
                            data[lin]["s1"] = s1[i1:i2]
                            i1 = i2
                else:
                    s2[index] = np.nan
                    if inter:
                        if direction in ("N", "S"):
                            self.sensor2_inter[:, lin] = s2
                        else:
                            self.sensor2_inter[lin, :] = s2
                    else:
                        i1 = 0
                        for lin in line_keys:
                            i2 = i1 + len(data[lin]["x"])
                            data[lin]["s2"] = s2[i1:i2]
                            i1 = i2
                read_flag = False

    def get_mouse_click(self, fig):
        """
        wait for a mous click within figure "fig".

        Parameters
        ----------
        fig : matplotlib figure
            for main window, it must be self.w.mplvl
            for floating window, it is the name that has been given to the
            window at creation time

        Returns
        -------
        None.
        Mouse position, button pressed and the general event parameters are
        accessible through self.x_mouse, self.y_mouse, self.mouse and self.event

        """
        self.wait = True
        self.click = False
        self.press = False

        def onClick(event):
            self.wait = False
            self.click = True
            self.event = event
            self.x_mouse = event.xdata
            self.y_mouse = event.ydata
            self.mouse = event.button
            self.xwin = event.x
            self.ywin = event.y

        # def onPress(event):
        #     self.wait = False
        #     self.press = True
        #     self.key = event.key
        fig.canvas.mpl_connect("button_press_event", onClick)
        # fig.canvas.mpl_connect("key_press_event",onPress)
        while self.wait:
            QtCore.QCoreApplication.processEvents()

    # def get_key(self,event):
    #     self.wait = False
    #     print(f"key: {event}")
    #     if event.key == 16777234:
    #         self.key = "left"
    #     elif event.key == 16777236:
    #         self.key = "right"
    #     else:
    #         self.key = event.key

    def setHelp(self, text, fig):
        """
        Set help text at bottom of screen.

        Parameters
        ----------
        text : str
            Text to be printed (defined in __init__)
            Text is written in QLabel widget. In order to keep the widget at the
            bottom of the screen, the existing one is first erased and then reopened.
            This call has to be done after any modification of the graphics window.

        fig : Matplotlib Figure
        Figure where to write the help text.

        Returns
        -------
        None
        """
        try:
            self.help.close()
        except (AttributeError, RuntimeError):
            pass
        self.help = QtWidgets.QLabel(self)
        self.help.setMaximumHeight(15)
        if fig == self.w.mplvl:
            fig.addWidget(self.help)
        else:
            fig.layout.addWidget(self.help)
        self.help.show()
        self.help.setText(text)

    def changeColorScale(self):
        """
        Change the limiting quantiles for creating of color scales.

        The values below self.percent and above (1-self.percent) are clipped

        Returns
        -------
        None.

        """
        cols = ["rainbow", "seismic", "viridis"]
        if self.w.grad_data:
            results, okButton = dialog(
                [
                    "If min==max, quantile is used\n"
                    + "If quantile is also 0, min and max values of arrays\n"
                    + "If min!=max, these values set the color scale limits",
                    "Cliping quantile for plotting",
                    f"Minimum of color scale sensor1 [{self.unit}]",
                    f"Maximum of color scale sensor1 [{self.unit}]",
                    "_______________________________",
                    f"Minimum of color scale sensor2 [{self.unit}]",
                    f"Maximum of color scale sensor2 [{self.unit}]",
                    "_______________________________",
                    f"Minimum of color scale gradient [{self.unit}/m]",
                    f"Maximum of color scale gradient [{self.unit}/m]",
                    "Color map",
                    cols,
                ],
                ["l", "e", "e", "e", "l", "e", "e", "l", "e", "e", "l", "b"],
                [
                    None,
                    self.percent,
                    self.mincol1,
                    self.maxcol1,
                    None,
                    self.mincol2,
                    self.maxcol2,
                    None,
                    self.mincolg,
                    self.maxcolg,
                    0,
                ],
                "Color scale limits",
            )
        else:
            results, okButton = dialog(
                [
                    "If min==max, quantile is used\n"
                    + "If quantile is also 0, min and max values of arrays\n"
                    + "If min!=max, these values set the color scale limits",
                    "Cliping quantile for plotting",
                    f"Minimum of color scale [{self.unit}]",
                    f"Maximum of color scale [{self.unit}]",
                    "Color map",
                    cols,
                ],
                ["l", "e", "e", "e", "l", "b"],
                [None, self.percent, self.mincol1, self.maxcol1, 0],
                "Color scale limits",
            )
        if okButton:
            self.percent = float(results[1])
            self.mincol1 = float(results[2])
            self.maxcol1 = float(results[3])
            if self.w.grad_data:
                self.mincol2 = float(results[5])
                self.maxcol2 = float(results[6])
                self.mincolg = float(results[8])
                self.maxcolg = float(results[9])
                self.color = cols[int(results[11])]
            else:
                self.color = cols[int(results[5])]
            self.plotActual(self.dat[self.actual_plotted_file].data)
        else:
            print(
                f"\nClipping quantile left at {self.percent:0.3f}\n"
                + f"    minimum color: {self.mincol1:0.1f} {self.unit}"
                + f"    maximum color: {self.maxcol1:0.1f} {self.unit}"
            )

    def plotBase(self):
        """
        Plot base station data as function of time (seconds in year).
        The user may erase erroneous data by clicking the mouse wheel at the
        beginning and at the end of the zone to be erased.
        A click with any other mouse button finishes the module and closes the
        floating window.

        Returns
        -------
        None.

        """
        self.function = "base"
        while True:
            try:
                self.fig_base.clf()
            except (NameError, AttributeError):
                pass
            base = self.base.base
            self.fig_base = newWindow("Diurnal variations", 800, 500)
            self.ax_base = self.fig_base.fig.add_subplot()
            self.ax_base.plot(base.time_base / 86400.0, base.base)
            self.ax_base.set_title("Diurnal variations")
            self.ax_base.set_xlabel("Time of acquisition [day in year]")
            if self.data_types[self.actual_plotted_file] == "magnetic":
                self.ax_base.set_ylabel("Magnetic field [nT]")
            else:
                self.ax_base.set_ylabel("Gravity field [mGal]")
            self.setHelp(
                "Click left mouse button to close, wheel at two points "
                + "to cut out data",
                self.fig_base,
            )
            self.fig_base.show()
            self.get_mouse_click(self.fig_base.fig)
            if self.mouse == 2:
                x0 = self.x_mouse * 86400.0
                self.get_mouse_click(self.fig_base.fig)
                x1 = self.x_mouse * 86400.0
                index = np.where(
                    (base.time_base > min(x0, x1)) & (base.time_base < max(x0, x1))
                )
                self.base.base.base[index] = np.nan
            else:
                self.function = "main"
                self.fig_base.close()
                self.plotActual(self.dat[self.actual_plotted_file].data)
                return

    def plotMedian(self):
        """
        Plot medians of every measured line (from non-interpolated data). Odd
        lines and even lines are plotted with different colours. Also sensor1
        and sensor 2 data are plotted in to the same axis and distinuished by
        different colors.

        """
        median1_even = []
        median1_odd = []
        if self.w.grad_data:
            median2_even = []
            median2_odd = []
        nline_even = []
        nline_odd = []
        data = deepcopy(self.dat[self.actual_plotted_file].data)
        for key, val in data.items():
            if isinstance(key, (str)):
                break
            if key % 2 == 0:
                median1_even.append(val["median1"])
                if self.w.grad_data:
                    median2_even.append(val["median2"])
                nline_even.append(key + 1)
            else:
                median1_odd.append(val["median1"])
                if self.w.grad_data:
                    median2_odd.append(val["median2"])
                nline_odd.append(key + 1)
        self.fig_median = newWindow("Medians", 800, 500)
        self.function = "median"
        self.ax_median = self.fig_median.fig.add_subplot()
        self.ax_median.plot(
            nline_even, median1_even, color="b", marker="*", label="sensor1 even"
        )
        self.ax_median.plot(
            nline_odd, median1_odd, color="c", marker="*", label="sensor1 odd"
        )
        if self.w.grad_data:
            self.ax_median.plot(
                nline_even, median2_even, color="r", marker="*", label="sensor2 even"
            )
            self.ax_median.plot(
                nline_odd, median2_odd, color="orange", marker="*", label="sensor2 odd"
            )
        self.ax_median.set_title("Median values of all profiles")
        self.ax_median.set_xlabel("Number of line")
        if self.data_types[self.actual_plotted_file] == "magnetic":
            self.ax_median.set_ylabel("Magnetic field [nT]")
        else:
            self.ax_median.set_ylabel("Gravity field [mGal]")
        self.ax_median.legend(bbox_to_anchor=(1, 0), loc="lower right")
        self.setHelp("Click any mouse button to close", self.fig_median)
        self.fig_median.show()
        self.get_mouse_click(self.fig_median.fig)
        self.function = "main"
        self.fig_median.close()

    # def zooming(self):
    #     pass

    # def zoomOut(self):
    #     pass

    # def zoomIn(self):
    #     pass

    # def zoomInitial(self):
    #     pass

    def savePlot(self):
        """
        Function saves plot inside actual window into a png file
        The file name is prefix_date_time.png
        Prefix depends on the actual image type.

        Returns
        -------
        None.

        """
        now = datetime.now()
        c_time = now.strftime("%H-%M-%S")
        today = date.today()
        d1 = today.strftime("%Y-%m-%d")
        fname = f"{self.dat[self.actual_plotted_file].data['type']}_{d1}_{c_time}.png"
        self.fig.savefig(fname)

    def diurnal(self):
        """
        If no base station data have been read create diurnal variation
        by adjusting a spline of degree deg to the median values as function
        of time.

        """
        if len(self.base_files) > 0:
            deg = 3
            self.day_joint_flag = False
            self.diff_weight = 1.0
        else:
            if self.n_blocks > 1:
                results, okButton = dialog(
                    [
                        "Degree of polynom",
                        "Treat every block separately",
                        "Weight for differences at edges",
                    ],
                    ["e", "c", "e"],
                    [3, None, 5],
                    "Diurnal variation by curve fit",
                )
                if not okButton:
                    print("\nDiurnal correction not applied")
                deg = int(results[0])
                if int(results[1]) > 0:
                    self.day_joint_flag = True
                else:
                    self.day_joint_flag = False
                self.diff_weight = float(results[2])
            else:
                results, okButton = dialog(
                    ["Degree of polynom"], ["e"], [3], "Diurnal variation by curve fit"
                )
                if not okButton:
                    print("\nDiurnal correction not applied")
                deg = int(results[0])
                self.day_joint_flag = False
                self.diff_weight = 1.0
        self.treatments["diurnal"] = True
        self.u.diurnal_correction(degree=deg, diff_weight=self.diff_weight)
        self.w.basePlot.setEnabled(True)
        self.plotActual(self.dat[self.actual_plotted_file].data)

    def clean(self):
        """
        Delete erroneous data
        Opions are
        * giving fixed upper and/or lower bounds (in or mGal)
        * eliminating all data below and/or above a certain quantile
        * choose limits by mouse click an a histogram (same value for both
        sensors). First click: cut below, second: cut above. a click outside
        an axis means not clipping in this direction.
        Data outside the chosen zone are set to Nan.

        Returns
        -------
        None.

        """
        # Get extreme valuess where to clip values
        results, okButton = dialog(
            [
                "Lower fixed clip value",
                "Upper fixed clip value",
                "Lower percentile",
                "upper percentile",
                "histogram",
            ],
            ["e", "e", "e", "e", "c"],
            [None, None, 0.01, None, None],
            "Clipping parameters",
        )
        if okButton:
            min_fix = None
            max_fix = None
            percent_down = None
            percent_up = None
            if results[0] != "None":
                min_fix = float(results[0])
            if results[1] != "None":
                max_fix = float(results[1])
            if results[2] != "None":
                percent_down = float(results[2])
            if results[3] != "None":
                percent_up = float(results[3])
            if min_fix:
                percent_down = None
            if max_fix:
                percent_up = None
            # If extreme values should be chosen manually in histogram, do this now
            if results[4] == 0:
                data = self.dat[self.actual_plotted_file].data
                sensor1 = []
                sensor2 = []
                for key, val in data.items():
                    if isinstance(key, (str)):
                        break
                    sensor1 += list(val["s1"])
                    if data["grad_data"]:
                        sensor2 += list(val["s2"])
                sensor1 = np.array(sensor1)
                if data["grad_data"]:
                    sensor2 = np.array(sensor2)
                # Plot 1 or 2 histograms depending on the number of sensors used
                self.histo = newWindow("Clip_histogram", 800, 500)
                if data["grad_data"]:
                    self.ax_histo = self.histo.fig.subplots(1, 2)
                else:
                    ax = self.histo.fig.subplots(1, 1)
                    self.ax_histo = [ax]
                rmin1 = np.nanquantile(sensor1, 0.001)
                rmax1 = np.nanmax(sensor1)
                counts1, _, _ = self.ax_histo[0].hist(sensor1, 20, (rmin1, rmax1))
                if data["grad_data"]:
                    self.ax_histo[0].set_title("Sensor 1")
                else:
                    if data["type"] == "magnetic":
                        self.ax_histo[0].set_title("Magnetic field")
                    else:
                        self.ax_histo[0].set_title("Gravity field")
                if data["grad_data"]:
                    rmin2 = np.nanquantile(sensor2, 0.001)
                    rmax2 = np.nanmax(sensor2)
                    self.ax_histo[1].hist(self.w.sensor2, 20, (rmin2, rmax2))
                    self.ax_histo[1].set_title("Sensor 2")
                self.setHelp(
                    "click within one of the histograms for minimum value, "
                    + "then for maximum value. Click outside = None",
                    self.histo,
                )
                self.histo.show()
                # Wait for mous click to define lower data limit
                #                self.get_mouse_click(self.histo.fig, self.ax_histo)
                self.get_mouse_click(self.histo.fig)
                if self.x_mouse:
                    if data["grad_data"]:
                        if self.x_mouse > min(sensor1.min(), sensor2.min()):
                            self.ax_histo[0].plot(
                                [self.x_mouse, self.x_mouse], [0, max(counts1)], "r"
                            )
                            min_fix = self.x_mouse
                            percent_down = None
                    else:
                        if self.x_mouse > sensor1.min():
                            self.ax_histo[0].plot(
                                [self.x_mouse, self.x_mouse], [0, max(counts1)], "r"
                            )
                            min_fix = self.x_mouse
                            percent_down = None
                # Wait for mous click to define upper data limit
                #                self.get_mouse_click(self.histo.fig, self.ax_histo)
                self.get_mouse_click(self.histo.fig)
                if self.x_mouse:
                    if data["grad_data"]:
                        if self.x_mouse < max(sensor1.max(), sensor2.max()):
                            self.ax_histo[0].plot(
                                [self.x_mouse, self.x_mouse], [0, max(counts1)], "r"
                            )
                            max_fix = self.x_mouse
                            percent_up = None
                    else:
                        if self.x_mouse < sensor1.max():
                            self.ax_histo[0].plot(
                                [self.x_mouse, self.x_mouse], [0, max(counts1)], "r"
                            )
                            max_fix = self.x_mouse
                            percent_up = None
                self.histo.close()
        else:
            print("\nclipping cancelled")
            return
        self.treatments["clip"] = True
        # Delet data outside limits
        self.u.clean_data(
            min_fix=min_fix,
            max_fix=max_fix,
            percent_down=percent_down,
            percent_up=percent_up,
        )
        self.plotActual(self.dat[self.actual_plotted_file].data)

    def justify_median(self):
        """
        Adjust medians of every second line to the average of the medians of
        the neighboring lines in order to attenuate the directional effects
        of measurements.
        The lines to be modified may be the even ones or the odd ones (does not
        always give the same result). Medians of lines at the edges are set to
        the same value as the neighboring line.

        """
        results, okButton = dialog(
            ["Modify median of", ["Odd lines", "Even lines"]],
            ["l", "r"],
            [None, 0],
            "Justification type",
        )
        if okButton:
            justify = int(results[1])
        else:
            print("\nJustification cancelled")
            return
        self.treatments["justify_median"] = True
        self.u.justify_lines_median(justify)
        self.plotActual(self.dat[self.actual_plotted_file].data)

    def justify_gauss(self):
        """
        see Masoudi et al., J. Geophys. Eng., 2023

        Returns
        -------
        None.

        """
        results, okButton = dialog(
            [
                "Modify median of",
                ["Odd lines", "Even lines"],
                ["Global adjustment", "Line by line"],
            ],
            ["l", "r", "r"],
            [None, 0, 1],
            "Justification type",
        )
        if okButton:
            justify = int(results[1])
            local = int(results[2])
        else:
            print("\nJustification cancelled")
            return
        self.treatments["justify_Gauss"] = True
        self.w.justify_lines_gaussian(justify, local)
        self.plotActual(self.dat[self.actual_plotted_file].data)

    def interpol(self):
        """
        Interpolate data within the measurement lines onto a regular grid.
        No data are extrapolated, i.e. if a line starts later or finishes earlier
        than a regular grid, missing grid points are set to nan

        Returns
        -------
        None.

        """
        data = self.dat[self.actual_plotted_file].data
        if data[0]["direction"] in ("N", "S"):
            ddy = np.round(
                abs(data[0]["y"][-1] - data[0]["y"][0]) / len(data[0]["y"]), 2
            )
            ddx = np.round(abs(data[0]["x"][0] - data[1]["x"][0]), 2)
        else:
            ddx = np.round(
                abs(data[0]["x"][-1] - data[0]["x"][0]) / len(data[0]["x"]), 2
            )
            ddy = np.round(abs(data[0]["y"][0] - data[1]["y"][0]), 2)
        while True:
            results, okButton = dialog(
                ["dx [m]", "dy [m]"], ["e", "e"], [ddx, ddy], "Interpolation parameters"
            )
            if okButton:
                self.dx = float(results[0])
                self.dy = float(results[1])
            else:
                print("\nInterpolation cancelled")
                return
            xmin = 1.0e10
            xmax = -1.0e10
            ymin = 1.0e10
            ymax = -1.0e10
            for key, val in data.items():
                if isinstance(key, (str)):
                    break
                xmin = min(xmin, val["x"].min())
                xmax = max(xmax, val["x"].max())
                ymin = min(ymin, val["y"].min())
                ymax = max(ymax, val["y"].max())
            n_inter_x = (xmax - xmin) / self.dx
            n_inter_y = (ymax - ymin) / self.dy
            # if data[0]["direction"] in ("N", "S"):
            #     n_inter = int(np.round((self.w.x.max()-self.w.x.min()),1)/self.dx)
            # else:
            #     n_inter = int(np.round((self.w.y.max()-self.w.y.min()),1)/self.dx)
            if n_inter_x > 1000 or n_inter_y > 1000:
                answer = QtWidgets.QMessageBox.warning(
                    None,
                    "Warning",
                    f"dx/dy={self.dx}/{self.dy} produces huge grid "
                    + f"({n_inter_x}x{n_inter_y} points)\n "
                    + "Ignore and continue or\n  close and try again\n",
                    QtWidgets.QMessageBox.Ignore | QtWidgets.QMessageBox.Close,
                    QtWidgets.QMessageBox.Close,
                )
                if answer == QtWidgets.QMessageBox.Close:
                    continue
                break
            break
        self.inter_flag = True
        if data["grad_data"]:
            (
                self.sensor1_inter,
                self.sensor2_inter,
                self.grad_inter,
                self.x_inter,
                self.y_inter,
                self.z_inter,
            ) = self.u.interpol_2D(dx=self.dx, dy=self.dy)
        else:
            self.sensor1_inter, _, _, self.x_inter, self.y_inter, self.z_inter = (
                self.u.interpol_2D(dx=self.dx, dy=self.dy)
            )
        self.mask1 = np.isnan(self.sensor1_inter)
        self.dx = self.x_inter[1] - self.x_inter[0]
        self.dy = self.y_inter[1] - self.y_inter[0]
        self.sensor1_fill = self.u.extrapolate(
            self.sensor1_inter, self.x_inter, self.y_inter
        )
        if data["grad_data"]:
            self.mask2 = np.isnan(self.sensor2_inter)
            self.sensor2_fill = self.u.extrapolate(
                self.sensor2_inter, self.x_inter, self.y_inter
            )
            self.grad_fill = (self.sensor1_fill - self.sensor2_fill) / data["d_sensor"]
        self.w.saveSTN.setEnabled(True)
        self.w.fill.setEnabled(True)
        if data["type"] == "magnetic":
            self.w.poleReduction.setEnabled(True)
        self.w.tiltAngle.setEnabled(True)
        self.w.lineFFT.setEnabled(True)
        self.w.Spector_2D.setEnabled(True)
        self.w.prolongation.setEnabled(True)
        self.w.analytic.setEnabled(True)
        self.w.gaussJustify.setEnabled(True)
        self.treatments["interpol"] = True
        self.plotActual(self.dat[self.actual_plotted_file].data)

    def nanFill(self):
        """
        Fill nan values by interpolation of data in the direction perpendicular
        to the measurement direction (it is supposed that if a line is not
        complete, nearby ones will be). Extrapolation will be done (one wants
        to create a complete grid) and different possibilities exist (mainly
        spline or constant). Spline is often very risky.

        Returns
        -------
        None.

        """
        self.treatments["nan_fill"] = True
        self.sensor1_inter = self.u.extrapolate(
            self.sensor1_inter, self.x_inter, self.y_inter
        )
        if self.dat[self.actual_plotted_file].data["grad_data"]:
            self.sensor2_inter = self.u.extrapolate(
                self.sensor2_inter, self.x_inter, self.y_inter
            )
            self.grad_inter = (self.sensor1_inter - self.sensor2_inter) / self.dat[
                self.actual_plotted_file
            ].data["d_sensor"]
        self.plotActual(self.dat[self.actual_plotted_file].data)

    def reducePole(self):
        """
        Pole reduction is done, only for the external field, eventual remanent
        magnetization is not taken into account.

        Returns
        -------
        None.

        """
        self.treatments["pole"] = True

        self.sensor1_inter = self.u.pole_Reduction(
            self.sensor1_fill, self.dx, self.dy, self.inclination, self.declination
        )
        self.sensor1_fill = np.copy(self.sensor1_inter)
        self.sensor1_inter[self.mask1] = np.nan
        if self.dat[self.actual_plotted_file].data["grad_data"]:
            d_sensor = self.dat[self.actual_plotted_file].data["d_sensor"]
            self.sensor2_inter = self.u.pole_Reduction(
                self.sensor2_fill, self.dx, self.dy, self.inclination, self.declination
            )
            self.sensor2_fill = np.copy(self.sensor2_inter)
            self.sensor2_inter[self.mask2] = np.nan
            self.grad_inter = (self.sensor1_inter - self.sensor2_inter) / d_sensor
            self.grad_fill = (self.sensor1_fill - self.sensor2_fill) / d_sensor
        self.inclination = 90.0
        self.declination = 0.0
        self.plotActual(self.dat[self.actual_plotted_file].data)

    def log_spect(self, data, d, n_coef):
        """
        Calculate logarithmic power spectrum of a series of data

        Parameters
        ----------
        data : numpy 1D array, float
            Data to be analyzed.
        d : float
            Distance between data points (sampling distance).
        n_coef : int
            Number of coefficients of the spectrum to be returned.

        Returns
        -------
        numpy 1D array
            logarithm of normalized power spectrum.
        list
            Wavenumbers of power spectrum.

        """
        index = np.isfinite(data)
        data = data[index]
        if len(data) < 3:
            return [None], [None]
        # Calculate Fourier transform
        FT = np.fft.fft(data)
        FT *= 2 / len(data)
        FT[0] /= 2.0
        Fabs = abs(FT)
        k = np.fft.fftfreq(len(data), d=d) * 2.0 * np.pi
        # Plot data only up to coefficient (n_coef)
        eps = 1e-10
        # Return log of power spectrum (add epsilon to avoid log(0)), wave numbers
        return np.log(Fabs[1:n_coef] ** 2 + eps), k[1:n_coef]

    def fit2lines(self, x, y, n0, n1, n2, n0_flag):
        """
        Fit two regression lines to a data set.
        Find the point where a break in slope gives the best fit between the
        n1th point and the n2th point

        Parameters
        ----------
        x : numpy 1D array, float
            x-coordinates of the data series.
        y : numpy 1D array, float
            y-coordinates of the data series.
        n0 : int
            first point of the series to be considered.
        n1 : int
            First point in series considered for possible slope break.
        n2 : int
            Last point in series considered for possible slope break.
        n0_flag : bool
            if True : first line must pass through point n0

        Returns
        -------
        regression coefficients for first slope
        regression coefficients for second slope
        int: position of slope beak
        float: misfit

        """
        r1_best = None
        r2_best = None
        qual = -1.0e20
        n3 = len(x)
        isplit = n1
        #        weight = np.exp(y)
        #        weight = np.arange(len(y))**2
        # Fit two regression lines to data. For this, search breaking point between
        #     third and 11th data point for which the fit is best
        for i in range(n1, n2):
            k1 = x[n0 : i + 1].reshape(-1, 1)
            k2 = x[i:].reshape(-1, 1)
            if n0_flag:
                reg1 = LR(fit_intercept=False).fit(k1 - k1[0], y[n0 : i + 1] - y[n0])
                reg1.intercept_ = y[n0] - k1[0] * reg1.coef_
            else:
                reg1 = LR(fit_intercept=True).fit(k1, x[n0 : i + 1])
            reg2 = LR(fit_intercept=True).fit(k2, y[i:])
            fit = reg1.score(k1, y[n0 : i + 1]) / (i - n0) + reg2.score(k2, y[i:]) / (
                n3 - i
            )
            # yp1 = reg1.predict(k1)
            # nn = len(yp1)
            # fit1 = np.sum((yp1-y[:nn])**2)
            # yp2 = reg2.predict(k2[1:])
            # fit2 = np.sum((yp2-y[nn:])**2)
            # fit = fit1+fit2
            # yp = np.concatenate((yp1,yp2))
            # fit = np.sum((yp - y)**2*weight)
            # If fit is better than earlier ones, calculate depths from both slopes
            if np.isfinite(fit) and fit > qual:
                qual = fit
                r1_best = reg1
                r2_best = reg2
                isplit = i
        return r1_best, r2_best, isplit, qual

    #     def fit3lines(self,x,y,n0,n1,n2):
    #         from sklearn.linear_model import LinearRegression as LR
    #         qual = np.inf
    #         n3 = len(x)
    # # Fit two regression lines to data. For this, search breaking point between
    # #     third and 11th data point for which the fit is best
    #         for i in range(n1,n2):
    #             for j in
    #             k1 = y[n0:i+1].reshape(-1,1)
    #             k2 = y[i:].reshape(-1,1)
    #             reg1 = LR().fit(k1,x[n0:i+1])
    #             reg2 = LR().fit(k2,x[i:])
    #             fit = reg1.score(k1,x[n0:i+1])*(i-n0)+reg2.score(k2,x[i:])*(n3-i)
    # # If fit is better than earlier ones, calculate depths from both slopes
    #             if fit < qual:
    #                 qual = fit
    #                 r1 = reg1
    #                 r2 = reg2
    #                 isplit = i
    #         return r1,r2,isplit

    def min_max(self, data, half_width=3):
        """
        Find all relative minima and maxima in a data vector.
        
        A maximum is found if a value at position i of the vector is larger
        than or equal to all other values in a range [i-half_width:i+half_with]
        and at the same time strictly larger than all values of one side.
        Sometimes, e.g. if seismograms are saturated, a slope exists on one side,
        but the values are constant on the other side. The one-sided test is to
        avoid that a flat curve with 2*half_width constant values is also
        considered as local maximum. Equivalent scheme for definition of a minimum.
        In addition, the function reduces possible local maxima and minima such that
        a maximum is always followed by a minimum and vice versa. If several local
        maxima follow each other (i.e. no wide enough local minimum exists between
        them), the program searches the strongest one of the subsequent maxima or, if
        several equal maximum values exists, takes as position of the maximum the
        center point between those multiple maxima (again for saturated curves).

        Parameters
        ----------
        data, half_width
        data: data vector to be analysed (Defult value: 3)
        half_width: integer value giving the number of samples analyzed to both \
                    sides of every data sample.

        Returns
        -------
        max_pos, max_val, min_pos, min_val
        max_pos: integer vector with all position numbers where there is a \
                 relative maximum in vector "data"
        max_val: values at these positions
        min_pos: integer vector with all position numbers where there is a \
                 relative minimum in vector "data"
        min_val: values at these positions
        """
        N = len(data)
        NN = np.arange(N, dtype="int")
        # extreme_pos (extreme_neg) will contain the sum of all values <= (>=) the
        #   central value
        # half will contain the maximum of the number of
        #   values < (>) the central value on the left and the right side
        # A maximum (minimum) is found if extreme_xxx[i]==(2*half_width+1) and if
        #   half_extreme_xxx[i]==half_width.
        extreme_pos = np.zeros(N, dtype=bool)
        extreme_neg = np.zeros(N, dtype=bool)
        # # Define vectors with ones have lengths of the full or half sliding window
        #     test = np.ones(2*half_width+1)
        #     half_test = np.ones(half_width)
        # Start loop over data points
        for k in range(N):
            # Sum of neigbouring points for which value[i] <= (>=) value[test_point]
            dn0 = min(half_width, k)
            dn1 = min(half_width, N - k - 1)
            width = dn0 + dn1 + 1
            ext_pos = sum(data[k] - data[k - dn0 : k + dn1 + 1] >= 0) == width
            # Sum of neighbouring values to the left (half1) and right (half2) < value [test_point]
            half1 = sum(data[k] - data[k - dn0 : k] > 0)
            half2 = sum(data[k] - data[k + 1 : k + dn1 + 1] > 0)
            half = max(half1, half2) == half_width
            extreme_pos[k] = ext_pos and half

            # Sum of neighbouring values to the left (half1) and right (half2) > value [test_point]
            ext_neg = sum(data[k] - data[k - dn0 : k + dn1 + 1] <= 0) == width
            half1 = sum(data[k] - data[k - dn0 : k] < 0)
            half2 = sum(data[k] - data[k + 1 : k + dn1 + 1] < 0)
            half = max(half1, half2) == half_width
            extreme_neg[k] = ext_neg and half
        # Search all points that fulfill the criteria for local maximum and minimum
        #        max_pos = NN[(extreme_pos==width) & (half_extreme_pos==half_width)]
        max_pos = NN[extreme_pos]
        max_val = data[max_pos]
        #        min_pos = NN[(extreme_neg==width) & (half_extreme_neg==half_width)]
        min_pos = NN[extreme_neg]
        min_val = data[min_pos]
        del extreme_pos, extreme_neg
        # mx_sig is a vector with length equal to number of found maxima with +1
        # mn_sig is a vector with length equal to number of found maxima with -1
        #   These vectors will be used to know which position is a maximum, which one
        #   a minimum, once all extrema are concatenated in a single vector in order to
        #   intercalate maxima and minima and to find places where multiple maxima or
        #   minima follow each other
        mx_sig = np.ones(len(max_pos))
        mn_sig = -np.ones(len(min_pos))
        # Concatenate positions, values and signs of maxima and minima into a single vector
        #   for each of them
        signs = np.concatenate((mx_sig, mn_sig))
        positions = np.concatenate((max_pos, min_pos))
        values = np.concatenate((max_val, min_val))
        # Order the concatenated vectors by positions
        iord = np.argsort(positions)
        pord = positions[iord]
        vord = values[iord]
        sord = signs[iord]
        ls = len(sord)
        # Prepare lists that will contain positions, values and signs of alternating
        #   extreme values (avoiding having several maxima (minima) following each other
        #   without a minumum (maximum) between them).
        pos = []
        val = []
        sig = []
        i = 1
        # Start loop over concatenated extreme positions
        while i < ls:
            # If sign of position [i] is different from position [i-1] accept position [i-1]
            #   into a new list
            if sord[i] != sord[i - 1]:
                pos.append(pord[i - 1])
                val.append(vord[i - 1])
                sig.append(sord[i - 1])
            if i == ls - 1:
                if sord[i] != sord[i - 1]:
                    pos.append(pord[i])
                    val.append(vord[i])
                    sig.append(sord[i])
                    i += 1
                break
            i += 1
            #            continue
            # if sign of position i is the same as the one of position i-1 search for next
            #   position where sign changes
            i1 = i + 1
            for i1 in range(i + 1, ls):
                if sord[i] != sord[i1]:
                    break
            if i1 < i:
                break
            # Search maximum values of the positions having the same sign
            #   the chosen position is the average position of all equal maximum (minimum)
            #   values. If one of the relative maxima (minima) has the strongest value, its
            #   position and value will be copied into the new list.
            if sord[i] > 0:
                mx = np.where(vord[i:i1] == max(vord[i:i1]))
                mpos = int(np.mean(pord[i:i1][mx]))
                pos.append(mpos)
                val.append(max(vord[i:i1]))
                sig.append(sord[i])
            else:
                mx = np.where(vord[i:i1] == min(vord[i:i1]))
                mpos = int(np.mean(pord[i:i1][mx]))
                pos.append(mpos)
                val.append(min(vord[i:i1]))
                sig.append(sord[i])
            i = i1 + 1
        del max_pos, max_val, min_pos, min_val, iord, pord, vord, sord
        # Transform lists to numpy arrays
        pos = np.array(pos)
        val = np.array(val)
        sig = np.array(sig)
        # Separate again relative maxima from relative minima
        max_val = val[sig > 0]
        max_pos = pos[sig > 0]
        min_val = val[sig < 0]
        min_pos = pos[sig < 0]
        del pos, val, sig, positions, signs, values
        return max_pos, max_val, min_pos, min_val

    def spector_line(self, data, d, n_coef, half_width):
        """
        Calculate depth of random sources with formula of (Spector and Grant,
        Geophysics, 1970) for one line.
        Depths are calculated by fitting two lines to logarithmic spectrum. The
        break point between the two lines is searched between the 4th spectral
        coefficient and the one at position n_Ny-4.

        Parameters
        ----------
        data : numpy 1D array, float
            Data to be analyzed
        d : float
            distance between data points [m].
        n_coef : int
            Number of coefficients to be used for line fitting.
        half_width : int
            Used for determination of local maxima: if the value of point # i
            is larger than all values between i-helf_width a,d i+half_width, the
            point i is considered a local maximum.

        Returns
        -------
        float
            Depth calculated with slope of small wave numbers.
        float
            Depth calculated with slope of large wave numbers.
        int
            Number of spectral coefficient where the slope break is located.
        float
            Intercept of first line (small wave numbers).
        float
            Intercept of second line (large wave numbers).
        float
            Misfit of data adjustment
        numpy 1D array, float
            Logarithmic power spectral values.
        numpy 1D array, float
            Wave numbers of power spectrum.

        """
        dd, kk = self.log_spect(data, d, n_coef)
        if not dd[0]:
            return None, None, -1, None, None, None, [None], [None]
        if kk[-1] < 0:
            index = kk > 0
            dd = dd[index]
            kk = kk[index]
            dd = dd[: int(len(dd) / 10)]
            kk = kk[: int(len(dd) / 10)]
        # In order to avoid negative depths, the analysis is started at the coefficient
        #    having the maximum amplitude (excluding the first coefficient which is the
        #    average value)
        max_pos, d, _, _ = self.min_max(dd, half_width=half_width)
        if len(max_pos) < 8:
            kkk = kk
            d = dd
        else:
            kkk = kk[max_pos]
        n0 = np.argmax(d)
        n1 = n0 + 3
        n2 = len(d) - 3
        if n2 <= n1:
            return None, None, -1, None, None, None, [None], [None]
        # Fit two regression lines to data. For this, search breaking point between
        #     third and 11th data point for which the fit is best
        reg1, reg2, isp, fit = self.fit2lines(kkk, d, n0, n1, n2, True)
        isplit = np.argmin(abs(kk - kkk[isp]))
        depth1 = -reg1.coef_[0] / 2
        depth2 = -reg2.coef_[0] / 2
        return depth1, depth2, isplit, reg1.intercept_, reg2.intercept_, fit, dd, kk

    # Store best fitting depths in list

    def spector(self):
        """
        Calculate depth of random sources with formula of (Spector and Grant,
        Geophysics, 1970) for all lines (N-S or E-W direction).
        Depths are calculated by fitting two lines to logarithmic spectrum. The
        break point between the two lines is searched between the 4th and the 10th
        spectral coefficient.
        Results of all lines are saved in file spector.dat.

        """
        # Check whether data are interpolated
        if not self.inter_flag:
            _ = QtWidgets.QMessageBox.warning(
                None,
                "Warning",
                "Data are not yet interpolated.\nYou should do this "
                + "before calling FFT.",
                QtWidgets.QMessageBox.Close,
                QtWidgets.QMessageBox.Close,
            )
            return
        # Wait for mouse click to choose line for which the result will be shown on screen
        self.function = "spector"
        xx = self.x_inter
        yy = self.y_inter
        max_len_x = self.x_inter.max() - self.x_inter.min()
        max_len_y = self.y_inter.max() - self.y_inter.min()
        max_len = [max_len_y, max_len_x]
        ndat = np.array([len(self.y_inter), len(self.x_inter)])
        Ny = np.array(ndat / 2, dtype=int)
        # Set parameters depending on measurement direction
        d = self.dat[self.actual_plotted_file].data[0]["direction"]
        if d in ("N", "S"):
            direction = 0
        else:
            direction = 1
        results, okButton = dialog(
            [
                "Direction of analysis",
                ["N-S", "E-W"],
                "Half width for maxima determination",
                f"Window length (not yet used)\nmax in X: {max_len_x} in Y: {max_len_y}",
            ],
            ["l", "r", "e", "e"],
            [None, direction + 1, 1, max_len[direction]],
            "FFT analysis",
        )
        if okButton:
            direction = int(results[1])
            half_width = int(results[2])
        else:
            print("No FFT analysis done")
            return
        dx = xx[1] - xx[0]
        dy = yy[1] - yy[0]
        if direction == 0:
            nlines = len(self.x_inter)
            dsamp = dy
        else:
            nlines = len(self.y_inter)
            dsamp = dx
        n_Ny = Ny[direction]
        # Prepare lists where to store the calculation results
        depths1 = []
        depths2 = []
        intercepts1 = []
        intercepts2 = []
        isplits = []
        lpos = []
        fits = []
        # Loop over all lines, extract data and define coordinates
        for il in range(nlines):
            if direction:
                data = self.sensor1_inter[il, :]
                pos_line = self.y_inter[il]
            else:
                data = self.sensor1_inter[:, il]
                pos_line = self.x_inter[il]
            depth1, depth2, isplit, intercept1, intercept2, fit, _, _ = (
                self.spector_line(data, dsamp, n_Ny, half_width)
            )
            if not depth1:
                continue
            # Store best fitting depths in list
            depths1.append(max(depth1, depth2))
            depths2.append(min(depth1, depth2))
            intercepts1.append(intercept1)
            intercepts2.append(intercept2)
            isplits.append(isplit)
            lpos.append(pos_line)
            fits.append(fit)
        # Store all calculated depths into file "spector.dat
        while True:
            self.fig_spector = newWindow("Spector", 1900, 1100)
            self.ax_spector = self.fig_spector.fig.add_subplot()
            self.ax_spector.plot(lpos, depths1)
            self.ax_spector.invert_yaxis()
            self.ax_spector.set_title("Average depths from spectral analysis")
            self.ax_spector.xaxis.set_minor_locator(AutoMinorLocator())
            if direction:
                self.ax_spector.set_xlabel("Northing of line [m]")
            else:
                self.ax_spector.set_xlabel("Easting of line [m]")
            self.ax_spector.set_ylabel("Average depth [m]")
            self.ax_spector.grid(visible=True, which="both")
            self.setHelp(
                "Click left mouse button to see spectrum and modify slope;"
                + " right mouse button to finish and close window",
                self.fig_spector,
            )
            self.fig_spector.show()
            self.get_mouse_click(self.fig_spector.fig)
            if self.mouse != 1:
                break
            if not self.x_mouse:
                continue
            il = np.argmin(abs(lpos - self.x_mouse))
            if direction:
                data = self.sensor1_inter[il, :]
                pos_line = self.y_inter[il]
            else:
                data = self.sensor1_inter[:, il]
                pos_line = self.x_inter[il]
            depth1, depth2, isplit, intercept1, intercept2, fit, dd, kk = (
                self.spector_line(data, dsamp, n_Ny, half_width)
            )
            dk = kk[1] - kk[0]
            max_pos, d, _, _ = self.min_max(dd, half_width=half_width)
            try:
                kkk = kk[max_pos]
            except IndexError:
                print(f"IndexError in Spector: max_pos={max_pos}")
                break
            n0 = 0
            y = np.zeros(n_Ny - 1)
            y[:] = np.nan
            y[n0 : isplit + 1] = intercept1 - kk[n0 : isplit + 1] * depth1 * 2
            y[isplit + 2 :] = intercept2 - kk[isplit + 2 :] * depth2 * 2
            y[isplit + 1] = np.nan
            self.fig_FFT = newWindow("FFT", 800, 500)
            self.ax_FFT = self.fig_FFT.fig.add_subplot()
            self.ax_FFT.plot(kk, dd, "k")
            self.ax_FFT.plot(kk, y, "r")
            self.ax_FFT.plot(kkk, d, "r*")
            self.ax_FFT.text(
                np.mean(kk[n0 : isplit + 1]) + dk,
                np.mean(y[n0 : isplit + 1]),
                f"{depth1:0.2f}",
                verticalalignment="bottom",
                horizontalalignment="left",
            )
            self.ax_FFT.text(
                np.mean(kk[isplit + 2 :]) + dk,
                np.mean(y[isplit + 2 :]),
                f"{depth2:0.2f}",
                verticalalignment="bottom",
                horizontalalignment="left",
            )
            self.ax_FFT.set_title(f"Line at {pos_line:0.1f}")
            if self.dat[self.actual_plotted_file].data["grad_data"]:
                self.ax_FFT.set_ylabel("ln(Amplitude**2) sensor 1")
                self.ax_FFT.set_xlabel("Wavenumber [rad/m] sensor 1")
            else:
                self.ax_FFT.set_ylabel("ln(Amplitude**2)")
                self.ax_FFT.set_xlabel("Wavenumber [rad/m]")
            self.setHelp(
                "Draw line with left mouse button to get new slope. "
                + "Single click to close without change",
                self.fig_FFT,
            )
            self.fig_FFT.show()
            self.followLine(self.fig_FFT.fig, self.ax_FFT, release_flag=True)
            if len(self.coor_x) > 0 and self.coor_x[0] != self.coor_x[1]:
                depths1[il] = (
                    -(
                        (self.coor_y[1] - self.coor_y[0])
                        / (self.coor_x[1] - self.coor_x[0])
                    )
                    / 2
                )
                intercepts1[il] = self.coor_y[0] + depths1[il] * 2 * self.coor_x[0]
                self.ax_FFT.text(
                    self.coor_x[0],
                    self.coor_y[0],
                    f"  {depths1[il]:0.2f}",
                    verticalalignment="bottom",
                    horizontalalignment="left",
                )
                self.fig_FFT.canvas.draw()
                self.get_mouse_click(self.fig_FFT.fig)
            self.fig_FFT.close()
            self.fig_spector.close()
        self.fig_spector.close()
        self.function = "main"
        with open(
            f"spector.dat_{self.dat[self.actual_plotted_file].data['type']}",
            "w",
            encoding="utf-8",
        ) as fo:
            fo.write("Line nr  position [m]    depth1 [m]    depth2 [m]    misfit\n")
            for i, lp in enumerate(lpos):
                fo.write(
                    f"{i:7d}{lp:10.2f}{depths1[i]:10.2f}"
                    + f"{depths2[i]:10.2f}    {fits[i]:0.5f}\n"
                )

    def spector2D(self):
        """
        Calculate depth of random sources with formula of (Spector and Grant,
        Geophysics, 1970) in 2D.

        Choose first a window length. Spectral coefficients are averaged over
        radial equidistant coefficients. Depths are calculated by fitting two
        lines to logarithmic spectrum. The break point between the two lines is
        searched between the 4th and the 10th spectral coefficient.
        Results of all lines are saved in file spector.dat.

        The window length should be defined such that the number of Fourier
        coefficients is at least 8:
        (n = window_length/(2*max(dx,dy)), dx, dy, step sizes defined during
        interpolation)

        Results are stored in file spector2D_<data_type>.dat; Data type may be
        "magnetic" or "gravity".

        """
        # from sklearn.linear_model import LinearRegression as LR
        self.function = "spector2D"
        try:
            x_min = self.x_inter.min()
            x_max = self.x_inter.max()
            dx = self.x_inter[1] - self.x_inter[0]
            y_min = self.y_inter.min()
            y_max = self.y_inter.max()
            dy = self.y_inter[1] - self.y_inter[0]
            d = max(dx, dy)
        except AttributeError:
            _ = QtWidgets.QMessageBox.warning(
                None,
                "Warning",
                "Data are not yet interpolated.\nYou should do this "
                + "before calling FFT.",
                QtWidgets.QMessageBox.Close,
                QtWidgets.QMessageBox.Close,
            )
            return
        max_len_x = x_max - x_min
        max_len_y = y_max - y_min
        window_len = np.round(min(max_len_x, max_len_y) / 4, 0)
        xstart = x_min + window_len / 2.0
        xend = x_max - window_len / 2.0
        nx = int((xend - xstart) / d)
        ystart = y_min + window_len / 2.0
        yend = y_max - window_len / 2.0
        ny = int((yend - ystart) / d)
        ntot = nx * ny
        nfac = max(int(np.ceil(np.sqrt(ntot / 500))), 1)
        step = d * nfac
        n_Nys = [int(window_len / (2 * dx)), int(window_len / (2 * dy))]
        n_Ny = np.min(n_Nys)
        # Set parameters depending on measurement direction
        results, okButton = dialog(
            [
                "Window length [m]",
                "  Attention: there must be at least 16 points per window length\n"
                + "             see below: Nr of FFT coefficients >= 8!",
                "Step size [m]",
                "Half width for maxima determination",
                f"Number of FFT coefficients\nmax in X: {n_Nys[1]}, Y: {n_Nys[0]}",
            ],
            ["e", "l", "e", "e", "e"],
            [window_len, None, step, 1, n_Ny],
            "2D FFT parameters",
        )
        if okButton:
            window_len = float(results[0])
            step = float(results[2])
            half_width = int(results[3])
            n_Ny = int(results[4])
        else:
            print("No FFT analysis done")
            return
        n_Nys = [int(window_len / (2 * dx)), int(window_len / (2 * dy))]
        if n_Ny < 8:
            _ = QtWidgets.QMessageBox.warning(
                None,
                "Warning",
                "For automatic depth determination the number of\n"
                + "FFT coefficients must be >= 8\n"
                + f"Actual value: N_coef: {n_Ny}\n\nSpector2D not calculated\n",
                QtWidgets.QMessageBox.Close,
                QtWidgets.QMessageBox.Close,
            )
            return
        if n_Ny > n_Nys[0]:
            _ = QtWidgets.QMessageBox.warning(
                None,
                "Warning",
                "For automatic depth determination the number of\n"
                + "FFT coefficients used for depth determination\n"
                + "must be <= nr of Nyquist coefficient\n"
                + f"Actual values: N_coef: {n_Ny}, Nyquist_y: {n_Nys[0]}\n\n"
                + "Retry increasing window length or interpolate with smaller dy\n\n"
                + "Spector2D not calculated\n",
                QtWidgets.QMessageBox.Close,
                QtWidgets.QMessageBox.Close,
            )
            return
        if n_Ny > n_Nys[1]:
            _ = QtWidgets.QMessageBox.warning(
                None,
                "Warning",
                "For automatic depth determination the number of\n"
                + "FFT coefficients used for depth determination\n"
                + "must be <= nr of Nyquist coefficient\n"
                + f"Actual values: N_coef: {n_Ny}, Nyquist_x: {n_Nys[1]}\n\n"
                + "Retry increasing window length or interpolate with smaller dx\n\n"
                + "Spector2D not calculated\n",
                QtWidgets.QMessageBox.Close,
                QtWidgets.QMessageBox.Close,
            )
            return
        nr, nc = self.sensor1_inter.shape
        # Prepare lists where to store the calculation results
        nstep_x = int(step / dx)
        nwin_x = int(window_len / dx)
        if nwin_x % 2:
            nwin_x -= 1
        window_len_x = nwin_x * dx
        nwinx2 = int(window_len_x / (2 * dx))
        xstart = x_min + window_len_x / 2.0
        n_xstart = int((xstart - self.x_inter[0]) / dx)
        xend = x_max - window_len_x / 2.0
        n_xend = int((xend - self.x_inter[0]) / dx) + 1
        xcalc_pos = np.arange(xstart, xend + step / 2.0, step)
        nstep_y = int(step / dy)
        nwin_y = int(window_len / dy)
        if nwin_y % 2:
            nwin_y -= 1
        window_len_y = nwin_y * dy
        nwiny2 = int(window_len_y / (2 * dy))
        ystart = y_min + window_len_y / 2.0
        n_ystart = int((ystart - self.y_inter[0]) / dy)
        yend = y_max - window_len_y / 2.0
        n_yend = int((yend - self.y_inter[0]) / dy) + 1
        ycalc_pos = np.arange(ystart, yend + step / 2.0, step)
        nx_calc = len(xcalc_pos)
        ny_calc = len(ycalc_pos)
        depths_1 = np.zeros((ny_calc, nx_calc))
        depths_2 = np.zeros((ny_calc, nx_calc))
        depths1 = np.zeros((ny_calc, nx_calc))
        depths2 = np.zeros((ny_calc, nx_calc))
        depths3 = np.zeros((ny_calc, nx_calc))
        depths4 = np.zeros((ny_calc, nx_calc))
        intercepts1 = np.zeros((ny_calc, nx_calc))
        intercepts2 = np.zeros((ny_calc, nx_calc))
        intercepts3 = np.zeros((ny_calc, nx_calc))
        intercepts4 = np.zeros((ny_calc, nx_calc))
        isplits1 = np.zeros((ny_calc, nx_calc), dtype=int)
        isplits2 = np.zeros((ny_calc, nx_calc), dtype=int)
        fits = np.zeros((ny_calc, nx_calc))
        ii = -1
        xpos = np.zeros(nx_calc)
        ypos = np.zeros(ny_calc)
        for i in range(n_xstart, n_xend, nstep_x):
            n1x = i - nwinx2
            n2x = i + nwinx2
            ii += 1
            jj = -1
            xpos[ii] = self.x_inter[i]
            for j in range(n_ystart, n_yend, nstep_y):
                jj += 1
                n1y = j - nwiny2
                n2y = j + nwiny2
                ypos[jj] = self.y_inter[j]
                data = self.sensor1_inter[n1y:n2y, i]
                depth1, depth2, isplit1, intercept1, intercept2, fit1, dd, kk = (
                    self.spector_line(data, dy, n_Ny, half_width)
                )
                depths1[jj, ii] = depth1
                depths2[jj, ii] = depth2
                intercepts1[jj, ii] = intercept1
                intercepts2[jj, ii] = intercept2
                isplits1[jj, ii] = isplit1
                data = self.sensor1_inter[j, n1x:n2x]
                depth3, depth4, isplit2, intercept3, intercept4, fit2, dd, kk = (
                    self.spector_line(data, dx, n_Ny, half_width)
                )
                depths3[jj, ii] = depth3
                depths4[jj, ii] = depth4
                intercepts3[jj, ii] = intercept3
                intercepts4[jj, ii] = intercept4
                isplits2[jj, ii] = isplit2
                if depth1:
                    if depth3:
                        depths_1[jj, ii] = (depth1 + depth3) * 0.5
                        depths_2[jj, ii] = (depth2 + depth4) * 0.5
                        fits[jj, ii] = np.sqrt((fit1**2 + fit2**2) * 0.5)
                    else:
                        depths_1[jj, ii] = depth1
                        depths_2[jj, ii] = depth2
                        fits[jj, ii] = fit1
                elif depth3:
                    depths_1[jj, ii] = depth3
                    depths_2[jj, ii] = depth4
                    fits[jj, ii] = fit2
        while True:
            self.fig_spect2, self.ax_spect2 = self.w.plotFloating(
                depths_1,
                xpos,
                ypos,
                wtitle="Depths from Spector&Grant",
                sizeh=1200,
                sizev=900,
                ptitle=f"Spectral depth solutions, window length: {window_len} m",
                xlabel="Easting [m]",
                ylabel="Northing [m]",
                clabel="Depth [m]",
                percent=0.005,
                c=self.color,
            )
            if self.w.geography_flag:
                self.w.plot_geography(self.ax_spect2)
            if self.w.plotLin_flag:
                self.w.plot_lineaments(self.ax_spect2)
                self.w.plot_lineaments(self.ax_spect2)
            self.ax_spect2.set_xlim([np.nanmin(self.x_inter), np.nanmax(self.x_inter)])
            self.ax_spect2.set_ylim([np.nanmin(self.y_inter), np.nanmax(self.y_inter)])
            self.setHelp(
                "Left click on map: show results of nearest point; "
                + "Right click on map close window and return to main window",
                self.fig_spect2,
            )
            self.get_mouse_click(self.fig_spect2.fig)
            if not self.x_mouse:
                continue
            if self.mouse > 1:
                self.fig_spect2.close()
                break
            self.fig_FFT2 = newWindow("FFT", 800, 500)
            self.ax_FFT2 = self.fig_FFT2.fig.subplots(2, 1)
            ixclick = np.argmin(abs(xpos - self.x_mouse))
            ixpos = np.argmin(abs(self.x_inter - xpos[ixclick]))
            iyclick = np.argmin(abs(ypos - self.y_mouse))
            iypos = np.argmin(abs(self.y_inter - ypos[iyclick]))
            data = self.sensor1_inter[iypos - nwiny2 : iypos + nwiny2, ixpos]
            dd, kk = self.log_spect(data, dy, n_Ny)
            self.ax_FFT2[0].plot(kk, dd, "k")
            if depths1[iyclick, ixclick]:
                depth1 = depths1[iyclick, ixclick]
                intercept1 = intercepts1[iyclick, ixclick]
                xxx = kk[0 : isplits1[iyclick, ixclick]]
                yyy = intercept1 - xxx * depth1 * 2
                self.ax_FFT2[0].plot(xxx, yyy, "r")
                self.ax_FFT2[0].text(
                    np.mean(xxx),
                    np.mean(yyy),
                    f"  {depth1:0.2f}",
                    verticalalignment="bottom",
                    horizontalalignment="left",
                )
                depth2 = depths2[iyclick, ixclick]
                intercept2 = intercepts2[iyclick, ixclick]
                xxx = kk[isplits1[iyclick, ixclick] :]
                yyy = intercept2 - xxx * depth2 * 2
                self.ax_FFT2[0].plot(xxx, yyy, "r")
                self.ax_FFT2[0].text(
                    np.mean(xxx),
                    np.mean(yyy),
                    f"{  depth2:0.2f}",
                    verticalalignment="bottom",
                    horizontalalignment="left",
                )
            self.ax_FFT2[0].set_title(f"N-S line at X = {xpos[ixclick]:0.1f}")
            if self.dat[self.actual_plotted_file].data["grad_data"]:
                self.ax_FFT2[0].set_ylabel("ln(Amplitude**2) sensor 1")
                self.ax_FFT2[0].set_xlabel("Wavenumber [rad/m] sensor 1")
            else:
                self.ax_FFT2[0].set_ylabel("ln(Amplitude**2)")
                self.ax_FFT2[0].set_xlabel("Wavenumber [rad/m]")
            data = self.sensor1_inter[iypos, ixpos - nwinx2 : ixpos + nwinx2]
            dd, kk = self.log_spect(data, dx, n_Ny)
            self.ax_FFT2[1].plot(kk, dd, "k")
            if depths3[iyclick, ixclick]:
                depth3 = depths3[iyclick, ixclick]
                intercept3 = intercepts3[iyclick, ixclick]
                xxx = kk[0 : isplits2[iyclick, ixclick]]
                yyy = intercept3 - xxx * depth3 * 2
                self.ax_FFT2[1].plot(xxx, yyy, "r")
                self.ax_FFT2[1].text(
                    np.mean(xxx),
                    np.mean(yyy),
                    f"  {depth3:0.2f}",
                    verticalalignment="bottom",
                    horizontalalignment="left",
                )
                depth4 = depths4[iyclick, ixclick]
                intercept4 = intercepts4[iyclick, ixclick]
                xxx = kk[isplits2[iyclick, ixclick] :]
                yyy = intercept4 - xxx * depth4 * 2
                self.ax_FFT2[1].plot(xxx, yyy, "r")
                self.ax_FFT2[1].text(
                    np.mean(xxx),
                    np.mean(yyy),
                    f"{  depth4:0.2f}",
                    verticalalignment="bottom",
                    horizontalalignment="left",
                )
            self.ax_FFT2[1].set_title(f"E-W line at Y = {ypos[iyclick]:0.1f}")
            if self.dat[self.actual_plotted_file].data["grad_data"]:
                self.ax_FFT2[1].set_ylabel("ln(Amplitude**2) sensor 1")
                self.ax_FFT2[1].set_xlabel("Wavenumber [rad/m] sensor 1")
            else:
                self.ax_FFT2[1].set_ylabel("ln(Amplitude**2)")
                self.ax_FFT2[1].set_xlabel("Wavenumber [rad/m]")
            # dk = kk[1]-kk[0]
            # max_pos,d,_,_ = self.min_max(dd,half_width=half_width)
            # kkk = kk[max_pos]
            # n0 = 0
            # self.ax_FFT.plot(kkk,d,"r*")
            self.setHelp(
                "Draw line with left mouse button in upper window to get new slope. "
                + "Single click to close without change",
                self.fig_FFT2,
            )
            self.fig_FFT2.show()
            self.followLine(self.fig_FFT2.fig, self.ax_FFT2[0], release_flag=True)
            if len(self.coor_x) > 0 and self.coor_x[0] != self.coor_x[1]:
                depths1[iyclick, ixclick] = (
                    -(
                        (self.coor_y[1] - self.coor_y[0])
                        / (self.coor_x[1] - self.coor_x[0])
                    )
                    / 2
                )
                intercepts1[iyclick, ixclick] = (
                    self.coor_y[0] + depths1[iyclick, ixclick] * 2 * self.coor_x[0]
                )
                self.ax_FFT2[0].text(
                    self.coor_x[0],
                    self.coor_y[0],
                    f"  {depths1[iyclick,ixclick]:0.2f}",
                    verticalalignment="bottom",
                    horizontalalignment="left",
                )
                self.fig_FFT2.canvas.draw()
            #                self.get_mouse_click(self.fig_FFT2.fig)
            self.setHelp(
                "Draw line with left mouse button in lower window to get new slope. "
                + "Double click to close without change",
                self.fig_FFT2,
            )
            self.fig_FFT2.canvas.draw()
            self.followLine(self.fig_FFT2.fig, self.ax_FFT2[1], release_flag=True)
            if len(self.coor_x) > 0 and self.coor_x[0] != self.coor_x[1]:
                depths3[iyclick, ixclick] = (
                    -(
                        (self.coor_y[1] - self.coor_y[0])
                        / (self.coor_x[1] - self.coor_x[0])
                    )
                    / 2
                )
                intercepts1[iyclick, ixclick] = (
                    self.coor_y[0] + depths3[iyclick, ixclick] * 2 * self.coor_x[0]
                )
                self.ax_FFT2[1].text(
                    self.coor_x[0],
                    self.coor_y[0],
                    f"  {depths3[iyclick,ixclick]:0.2f}",
                    verticalalignment="bottom",
                    horizontalalignment="left",
                )
                self.fig_FFT2.canvas.draw()
                self.setHelp("Click to finish", self.fig_FFT2)
                self.get_mouse_click(self.fig_FFT2.fig)
            self.fig_FFT2.close()
            self.fig_spect2.close()

        self.function = "main"
        with open(
            f"spector2D_{self.data_types[self.actual_plotted_file]}.dat",
            "w",
            encoding="utf-8",
        ) as fo:
            fo.write("Line nr  X [m]      Y[m]    depth1 [m]    depth2 [m]    misfit\n")
            nr, nc = depths_1.shape
            for i in range(nc):
                for j in range(nr):
                    if np.isfinite(depths_1[j, i]):
                        fo.write(
                            f"{i:7d}{xpos[i]:10.1f}{ypos[j]:10.1f}"
                            + f"{depths_1[j,i]:10.2f}{depths_2[j,i]:10.2f}"
                            + f"    {fits[j,i]:0.5f}\n"
                        )

    def tilt(self):
        """
        Calculate tilt angle (Miller & Singh, JAG, 1994)

        Returns
        -------
        None.

        """
        if self.dat[self.actual_plotted_file].data["grad_data"]:
            tilt_ang, vgrad, vgrad2, hgrad = self.u.tilt(
                self.sensor1_fill, self.grad_fill, self.dx, self.dy
            )
        else:
            tilt_ang, vgrad, vgrad2, hgrad = self.u.tilt(
                self.sensor1_fill, None, self.dx, self.dy
            )
        tilt_ang[self.mask1] = np.nan
        vgrad[self.mask1] = np.nan
        vgrad2[self.mask1] = np.nan
        hgrad[self.mask1] = np.nan
        nr, nc = tilt_ang.shape
        data = vgrad.reshape(nr, nc, 1)
        data = np.concatenate((data, vgrad2.reshape(nr, nc, 1)), axis=2)
        data = np.concatenate((data, hgrad.reshape(nr, nc, 1)), axis=2)
        self.fig_grad, self.ax_grad = self.w.plotFloating(
            data,
            self.x_inter,
            self.y_inter,
            wtitle="Gradients",
            sizeh=1200,
            sizev=900,
            ptitle=[
                "Vertical gradient",
                "Second vertical derivative",
                "Horizontal gradient",
            ],
            percent=0.005,
            xlabel=["Easting [m]", "Easting [m]", "Easting [m]"],
            ylabel=["Northing [m]", "Northing [m]", "Northing [m]"],
            clabel=[
                f"Vert. grad. [{self.unit}/m]",
                f"2nd vert. grad. [{self.unit}/m2]",
                f"Hor. grad. [{self.unit}/m]",
            ],
            c=self.color,
        )
        if self.w.plotLin_flag:
            self.w.plot_lineaments(self.ax_grad[0])
            self.w.plot_lineaments(self.ax_grad[1])
            self.w.plot_lineaments(self.ax_grad[2])
        xmin = np.nanmin(self.x_inter)
        xmax = np.nanmax(self.x_inter)
        ymin = np.nanmin(self.y_inter)
        ymax = np.nanmax(self.y_inter)
        self.ax_grad[0].set_xlim([xmin, xmax])
        self.ax_grad[0].set_ylim([ymin, ymax])
        self.ax_grad[1].set_xlim([xmin, xmax])
        self.ax_grad[1].set_ylim([ymin, ymax])
        self.ax_grad[2].set_xlim([xmin, xmax])
        self.ax_grad[2].set_ylim([ymin, ymax])

        self.fig_tilt, self.ax_tilt = self.w.plotFloating(
            tilt_ang,
            self.x_inter,
            self.y_inter,
            wtitle="Tilt angle",
            sizeh=1200,
            sizev=900,
            ptitle="Tilt angle",
            xlabel="Easting [m]",
            ylabel="Northing [m]",
            clabel="Tilt angle [rad]",
            c=self.color,
        )
        if self.w.geography_flag:
            self.w.plot_geography(self.ax_tilt)
        if self.w.plotLin_flag:
            self.w.plot_lineaments(self.ax_tilt)
        self.ax_tilt.set_xlim([np.nanmin(self.x_inter), np.nanmax(self.x_inter)])
        self.ax_tilt.set_ylim([np.nanmin(self.y_inter), np.nanmax(self.y_inter)])
        self.setHelp(
            "Left click add point, right click finish line. "
            + "Immediate right click without chosen point: stop tilt routine",
            self.fig_tilt,
        )
        while True:
            self.followLine(self.fig_tilt.fig, self.ax_tilt)
            # If followLine was exited with the wheel button, this means that no new line
            # has been defined. It is then considered that the line nearest to the mouse click
            # should be erased
            if self.mouse == 2:
                if len(list(self.lineaments.keys())) == 0:
                    continue
                xc = self.x_event
                yc = self.y_event
                min_lin = list(self.lineaments.keys())[0]
                min_dist = (xc - self.lineaments[min_lin]["x"][0]) ** 2 + (
                    yc - self.lineaments[min_lin]["y"][0]
                ) ** 2
                for ll, lin in enumerate(self.lineaments.items()):
                    for i in range(len(lin["x"])):
                        dist = (xc - lin["x"][i]) ** 2 + (yc - lin["y"][i]) ** 2
                        if dist < min_dist:
                            min_lin = ll
                            min_dist = dist
                print(f"Erase lineament {min_lin}")
                del self.lineaments[min_lin]
                self.fig_tilt.close()
                self.fig_tilt, self.ax_tilt = self.w.plotFloating(
                    tilt_ang,
                    self.w.x_inter,
                    self.w.y_inter,
                    wtitle="Tilt angle",
                    sizeh=1200,
                    sizev=900,
                    ptitle="Tilt angle",
                    xlabel="Easting [m]",
                    ylabel="Northing [m]",
                    clabel="Tilt angle [rad]",
                    c=self.color,
                )
                if self.w.plotLin_flag:
                    self.w.plot_lineaments(self.ax_tilt)
                self.ax_tilt.set_xlim(
                    [np.nanmin(self.x_inter), np.nanmax(self.x_inter)]
                )
                self.ax_tilt.set_ylim(
                    [np.nanmin(self.y_inter), np.nanmax(self.y_inter)]
                )
                self.setHelp(
                    "Left click add point, right click finish line (not integrated). "
                    + "Immediate right click without chosen point: stop tilt routine. "
                    + "Click with mouse wheel eliminates nearest lineament",
                    self.fig_tilt,
                )
                continue
            if len(self.coor_x) == 0:
                self.fig_tilt.close()
                self.fig_grad.close()
                break
            self.nlineaments += 1
            self.lineaments[self.nlineaments] = {}
            self.lineaments[self.nlineaments]["x"] = np.array(self.coor_x[:-1])
            self.lineaments[self.nlineaments]["y"] = np.array(self.coor_y[:-1])
            self.lineaments[self.nlineaments]["type"] = (
                f"{self.dat[self.actual_plotted_file].data['type']}"
            )
            if self.w.plotLin_flag:
                self.w.plot_lineaments(self.ax_tilt)
            self.fig_tilt.canvas.draw()
        if self.nlineaments > 0:
            self.w.plotLineaments.setEnabled(True)
            self.save_lineaments()
        with open("tilt.dat", "w", encoding="utf-8") as fo:
            fo.write("    X       Y     angle\n")
            for i, x in enumerate(self.x_inter):
                for j, y in enumerate(self.y_inter):
                    fo.write(f"{x:0.2f} {y:0.2f} {tilt_ang[j,i]:0.3f}\n")
        self.plotActual(self.dat[self.actual_plotted_file].data)

    def continuation(self):
        """
        Calculate field at higher or lower altitude (Nabighian, Geophysics, 1972)
        """
        results, okButton = dialog(
            ["Continuation_height [m] (>0: upward)"],
            ["e"],
            [10.0],
            "Field continuation",
        )
        if okButton:
            self.dz = float(results[0])
        else:
            print("\nUpward continuation cancelled")
            return
        pro_data1 = self.u.continuation(self.sensor1_fill, self.dx, self.dy, self.dz)
        self.sensor1_fill = np.copy(pro_data1)
        pro_data1[self.mask1] = np.nan
        self.sensor1_inter = np.copy(pro_data1)
        if self.dat[self.actual_plotted_file].data["grad_data"]:
            pro_data2 = self.u.continuation(
                self.sensor2_fill, self.dx, self.dy, self.dz
            )
            self.sensor2_fill = np.copy(pro_data2)
            pro_data2[self.mask1] = np.nan
            pro_datag = (pro_data1 - pro_data2) / self.d_sensor
            self.grad_fill = (self.sensor1_fill - self.sensor2_fill) / self.d_sensor
            self.sensor2_inter = np.copy(pro_data2)
            self.grad_inter = np.copy(pro_datag)
        self.treatments["up"] = True
        self.height += self.dz
        self.plotActual(self.dat[self.actual_plotted_file].data)

    def analyticSignal(self):
        """
        Calculate analytic signal (Nabighian, Geophysics, 1972)

        Returns
        -------
        None.

        """
        # Calculate analytic signal of first sensor
        d = self.dat[self.actual_plotted_file].data[0]["direction"]
        if d in ("N", "S"):
            direction = 0
        else:
            direction = 1
        analytic_signal1 = self.u.analyticSignal(self.sensor1_fill, self.dx, self.dy)
        # Calculate instataneous phase and frequency
        inst_phase1 = np.unwrap(np.angle(analytic_signal1))
        if direction:
            add = inst_phase1[-1, :].reshape(1, -1)
            inst_freq1 = np.diff(inst_phase1, append=add, axis=0)
        else:
            add = inst_phase1[:, -1].reshape(-1, 1)
            inst_freq1 = np.diff(inst_phase1, append=add)
        # Set areas without measured data to nan
        analytic_signal1[self.mask1] = np.nan
        inst_phase1[self.mask1] = np.nan
        inst_freq1[self.mask1] = np.nan
        xmin = np.nanmin(self.x_inter)
        xmax = np.nanmax(self.x_inter)
        ymin = np.nanmin(self.y_inter)
        ymax = np.nanmax(self.y_inter)
        # The following commented lines allow computation of instantaneous phases and frequencies
        #     of the analytic signal. Since the results were not convincing, I commented the
        #     plotting of these data, but maybe with other data it would be worthwhile
        #     looking at them again. In this case, indent the first line following the
        #    commented "else"
        # if self.w.grad_data:
        #     analytic_signal2 = self.w.analyticSignal(self.sensor2_fill,self.dx,self.dy)
        #     inst_phase2 = np.unwrap(np.angle(analytic_signal2))
        #     if self.w.direction:
        #         add = inst_phase2[-1,:].reshape(1,-1)
        #         inst_freq2 = np.diff(inst_phase2, append=add)
        #     else:
        #         add = inst_phase2[:,-1].reshape(-1,1)
        #         inst_freq2 = np.diff(inst_phase2, append=add)
        #     analytic_signal2[self.mask2==True] = np.nan
        #     inst_phase2[self.mask2==True] = np.nan
        #     inst_freq2[self.mask2==True] = np.nan
        # nr = analytic_signal1.shape[0]
        # nc = analytic_signal1.shape[1]
        # if self.w.grad_data:
        #     data = abs(analytic_signal1).reshape(nr,nc,1)
        #     data = np.concatenate((data,abs(analytic_signal2).reshape(nr,nc,1)),axis=2)
        #     self.fig_ana,self.ax_ana = self.w.plotFloating(data,self.w.x_inter,\
        #                       self.w.y_inter,wtitle="Analytic signal", sizeh=1200,\
        #                       sizev=900,ptitle=["Analytic signal of sensor 1",\
        #                                         "Analytic signal of sensor 2"],\
        #                       xlabel=["Easting [m]","Easting [m]"], ylabel=["Northing [m]",
        #                       "Northing [m]"],\
        #                       clabel=[f"Analytic signal [{self.unit}/m]",\
        #                               f"Analytic signal [{self.unit}/m]"],percent=0.005)
        #     if self.w.plotLin_flag:
        #         self.w.plot_lineaments(self.ax_ana[0])
        #         self.w.plot_lineaments(self.ax_ana[1])
        #     self.ax_ana[0].set_xlim([xmin,xmax])
        #     self.ax_ana[0].set_ylim([ymin,ymax])
        #     self.ax_ana[1].set_xlim([xmin,xmax])
        #     self.ax_ana[1].set_ylim([ymin,ymax])
        # else:
        data = abs(analytic_signal1)
        # Plot analytic signal
        self.fig_ana, self.ax_ana = self.w.plotFloating(
            data,
            self.x_inter,
            self.y_inter,
            wtitle="Analytic signal",
            sizeh=1200,
            sizev=900,
            ptitle="Analytic signal",
            c=self.color,
            xlabel="Easting [m]",
            ylabel="Northing [m]",
            clabel=f"Analytic signal [{self.unit}/m]",
            percent=0.005,
        )
        # If lineaments have been measured, plot them now
        if self.w.plotLin_flag:
            self.w.plot_lineaments(self.ax_ana)
        self.ax_ana.set_xlim([xmin, xmax])
        self.ax_ana.set_ylim([ymin, ymax])
        if self.dat[self.actual_plotted_file].data["grad_data"]:
            self.ax_ana.grid(visible=True, which="both")
        self.ax_ana.xaxis.set_minor_locator(AutoMinorLocator())
        self.ax_ana.yaxis.set_minor_locator(AutoMinorLocator())
        self.setHelp(
            "Click left mouse button to see analytic signal of one line in "
            + "Y, right: in X; Wheel to finish and close window",
            self.fig_ana,
        )
        while True:
            # Wait for mouse click. Left mouse click will trigger inversion of analytic
            #      signal for depths along N-S lines, right mouse click along E-W lines.
            #      Clicking on wheel finishes this module.
            # The line neares to the mouse click will be shown in an own window with
            #     inversion results
            self.get_mouse_click(self.fig_ana.fig)
            if not self.x_mouse or not self.y_mouse:
                continue
            if self.mouse == 2:
                self.fig_ana.close()
                self.function = "main"
                return
            if self.mouse == 1:
                nline = np.argmin(abs(self.x_inter - self.x_mouse))
                pos = self.y_inter
                pos_line = self.x_inter[nline]
                text = f"{pos_line} East"
                text_x = "Northing [m]"
                nl = len(self.x_inter)
                direct = "y"
            else:
                nline = np.argmin(abs(self.y_inter - self.y_mouse))
                pos = self.x_inter
                pos_line = self.y_inter[nline]
                text = f"{pos_line} North"
                text_x = "Easting [m]"
                nl = len(self.y_inter)
                direct = "x"
            dx = pos[1] - pos[0]
            slope = []
            intercept = []
            depth = []
            alpha = []
            x_center = []
            y_center = []
            fit = []
            pmax = []
            # Interpretation is based on the squared analytic signal
            d2 = data**2
            index = np.isnan(d2)
            d2[index] = 0.0
            #            dmax = d2.max()
            # Mask valaues smaller than 1/1000 of maximum values by setting them to negative
            #      values
            dd = np.copy(d2)
            hist, edges = np.histogram(d2.flatten(), bins=1000, density=True)
            dedge = edges[1] - edges[0]
            cum = np.cumsum(hist) * dedge
            nlim = int(len(cum) * 0.85)
            self.fig_q = newWindow("Analytic signal quantiles", 800, 500)
            self.ax_q = self.fig_q.fig.add_subplot()
            self.ax_q.plot(edges[1:-nlim], cum[:-nlim])
            self.ax_q.set_title("Cumulative sum analytic signal")
            self.ax_q.set_xlabel("value")
            self.ax_q.set_ylabel("quantile")
            self.setHelp("Click left mouse button to define threshold", self.fig_q)
            self.fig_q.show()
            self.get_mouse_click(self.fig_q)
            dmin = self.x_mouse
            self.fig_q.close()
            results, okButton = dialog(
                ["Threshold", "Minimum depth", "Maximum depth"],
                ["e", "e", "e"],
                [dmin, 0, 2000],
                "Analytic signal parameters",
            )
            if okButton:
                dmin = float(results[0])
                depth_min = float(results[1])
                depth_max = float(results[2])
            dd[d2 < dmin] = -1.0
            # prepare window with inversion results of clicked line
            self.fig_sig = newWindow("Analytic signal inversion", 800, 500)
            # Start loop over lines
            s_best = 1.0e10
            i_best = 0.0
            x_best = 0.0
            depth_min = 0.0
            depth_max = 0.0
            for ll in range(nl):
                if direct == "y":
                    d = np.copy(d2[:, ll])
                    d_interest = np.copy(d2[:, ll])
                    d_interest[d < dmin] = -1.0
                    xline = self.x_inter[ll]
                else:
                    d = np.copy(d2[ll, :])
                    d_interest = np.copy(dd[ll, :])
                    yline = self.y_inter[ll]
                # If no data larger than the threshold exist on the line, continue with next line
                if d_interest.max() < 0.0:
                    continue
                # Il line corresponds to clicked one, plot analytic signal
                if ll == nline:
                    self.ax_sig = self.fig_sig.fig.add_subplot()
                    self.ax_sig.plot(pos, d, "k*", label="Analytic signal")
                    self.ax_sig.plot(
                        pos[d_interest > 0],
                        d_interest[d_interest > 0],
                        "g*",
                        label="Unused above threshold",
                    )
                    self.ax_sig.set_title(f"Squared analytic signal at {text}")
                    self.ax_sig.set_xlabel(text_x)
                    self.ax_sig.set_ylabel(f"Amplitude [{self.unit}/m]**2")
                # Find maxim and minima along line
                max_pos, _, min_pos, _ = self.min_max(d_interest, half_width=2)
                lab_flag = False
                # Start loop over all found maxima
                for p in max_pos:
                    # regression.append(None)
                    ff = np.inf
                    # Search points belonging to the actual maximum. Limit is found when either
                    #    a negative value is found (too small analytic signal to be interpreted) or
                    #    when the next relative minimum is found
                    for k in range(p, -1, -1):
                        if d_interest[k] < 0.0 or k in min_pos:
                            break
                    n1 = k + 1
                    for k in range(p, len(d_interest)):
                        if d_interest[k] < 0.0 or k in min_pos:
                            break
                    n2 = k
                    # If less than 4 points have been found belonging to the maximum, dont threat
                    #    this maximum
                    if n2 - n1 < 3:
                        continue
                    # Do inversion
                    # For this the inverse of squared analytic signal should have a linear
                    #     relation with the squared distance from the maximum.
                    #     Since the exact position of the maximum is not know, suppose that it is
                    #     located in the range measured maximum +/- 1/2*data spacing
                    #     The regression line is calculated using the squared amplitude as weight.
                    #     In this way, the high amplitudes have a stonger weight for the inversion
                    #     than the small ones.
                    #     The obtaines slope is the inverse of the parameter alpha, the intercept
                    #     corresponds to (depth/alpha)**2

                    # Calculate inverse of squared amplitude (add a small value to avoid
                    # division by 0)
                    y = 1 / (d_interest[n1:n2] + d_interest[p] * 1e-20)
                    # Test positions around measured maximum with a step of dx/10 for best solution
                    for x0 in np.arange(pos[p] - dx / 2.0, pos[p] + dx / 2.0, dx / 10):
                        # x contains the squared distance from maximum
                        x = (pos[n1:n2] - x0) ** 2
                        # G is the Frechet matrix
                        G = np.ones((len(x), 2))
                        G[:, 0] = x
                        # C contains the weights
                        C = np.diag(1.0 / y**2)
                        # Invert for regression line coefficients
                        mat1 = np.matmul(np.transpose(G), C)
                        coefs = np.matmul(
                            np.matmul(np.linalg.inv(np.matmul(mat1, G)), mat1), y
                        )
                        y_fit = np.matmul(G, coefs)
                        f = np.sum((y - y_fit) ** 2)
                        # If fit is better than earlier values, store corresponding
                        # parameters, but only
                        #    if slope and intercept are positive
                        if f < ff and coefs[0] > 0.0 and coefs[1] > 0.0:
                            s_best = coefs[0]
                            i_best = coefs[1]
                            x_best = x0
                            ff = f
                    # If a best fit was found, store the parameters
                    if np.isfinite(ff):
                        dep = (
                            np.sqrt(i_best / s_best)
                            - self.dat[self.actual_plotted_file].data["height"]
                        )
                        if depth_min <= dep <= depth_max:
                            slope.append(s_best)
                            intercept.append(i_best)
                            alpha.append(np.sqrt(1 / s_best))
                            depth.append(dep)
                            if direct == "y":
                                y_center.append(x_best)
                                x_center.append(xline)
                            else:
                                y_center.append(yline)
                                x_center.append(x_best)
                            fit.append(np.sqrt(ff))
                            pmax.append(p)
                            # If line corresponds to picked one, plot results into floating window
                            if ll == nline:
                                xx = (pos[n1:n2] - x_best) ** 2
                                yy = xx * s_best + i_best
                                yy = 1 / yy
                                if lab_flag:
                                    self.ax_sig.plot(
                                        pos[n1:n2], d_interest[n1:n2], "r*"
                                    )
                                    self.ax_sig.plot(pos[p], d_interest[p], "b*")
                                    self.ax_sig.plot(pos[n1:n2], yy, "r")
                                else:
                                    self.ax_sig.plot(
                                        pos[n1:n2],
                                        d_interest[n1:n2],
                                        "r*",
                                        label="Fitted points",
                                    )
                                    self.ax_sig.plot(
                                        pos[p],
                                        d_interest[p],
                                        "b*",
                                        label="Fitted maximum",
                                    )
                                    self.ax_sig.plot(
                                        pos[n1:n2], yy, "r", label="Calculated signal"
                                    )
                                lab_flag = True
                                self.ax_sig.text(
                                    x_center[-1],
                                    yy.max(),
                                    f"h:{depth[-1]:0.1f}, al:{alpha[-1]:0.2f}",
                                )
                if ll == nline:
                    self.ax_sig.legend(bbox_to_anchor=(1, 1), loc="upper right")
            slope = np.array(slope)
            intercept = np.array(intercept)
            alpha = np.array(alpha)
            depth = np.array(depth)
            x_center = np.array(x_center)
            y_center = np.array(y_center)
            fit = np.array(fit)
            # Store results into file, name depends on the directions used for calculation
            if direct == "y":
                file = "Analytic-signal-solutions_N-S-lines.dat"
            else:
                file = "Analytic-signal-solutions_E-W-lines.dat"
            with open(file, "w", encoding="utf-8") as fo:
                fo.write("     X        Y    Depth[m]   Alpha    Fit\n")
                for i, d in enumerate(depth):
                    fo.write(
                        f"{x_center[i]:0.1f} {y_center[i]:0.1f} "
                        + f"{d:0.2f} {alpha[i]:0.3f} {fit[i]:0.6f}\n"
                    )
            self.setHelp("Click any mouse button to close window", self.fig_sig)
            self.fig_sig.show()
            # Wait for click within results window to close it
            self.get_mouse_click(self.fig_sig)
            self.fig_sig.close()
            # Plot the obtained depths onto map of analytic signal
            #            vmin = np.quantile(depth,0.01)
            vmin = depth.min()
            vmax = np.quantile(depth, 0.99)
            nc = -int(np.log10(vmax)) + 2
            dv = round((vmax - vmin) / 10, nc)
            vmin = np.ceil(vmin / dv) * dv
            v = list(np.arange(vmin, vmin + 10.5 * dv, dv))
            cmap = plt.get_cmap("hot_r")
            norm = colors.BoundaryNorm(v, cmap.N)
            gna = self.ax_ana.scatter(x_center, y_center, c=depth, cmap=cmap, norm=norm)
            # Plot colorbar for depths into the axis
            cbbox = inset_axes(self.ax_ana, "10%", "30%", loc="upper left")
            [cbbox.spines[k].set_visible(False) for k in cbbox.spines]
            cbbox.xaxis.set_visible(False)
            cbbox.yaxis.set_visible(False)
            cbbox.set_facecolor([1, 1, 1, 0.7])
            cax = inset_axes(cbbox, "25%", "90%", loc="center left")
            dbar = self.fig_ana.fig.figure.colorbar(
                gna,
                cax=cax,
                extend="max",
                orientation="vertical",
                anchor=(0, 0),
                shrink=0.5,
            )
            dbar.ax.set_ylabel("depth [m]")
            ncd = -int(np.log10(dmin)) + 2
            self.ax_ana.set_title(
                f"{self.dat[self.actual_plotted_file].data['type']}"
                + " analytic signal and depth "
                + f"solutions in {direct} direction, "
                + f"threshold = {np.round(dmin,ncd)}"
            )
            self.setHelp(
                "Click left mouse button to see analytic signal of one "
                + "line in Y, right: in X; Wheel to finish "
                + "and close window",
                self.fig_ana,
            )
            self.fig_ana.canvas.draw()

    def followLine(self, fig, ax, release_flag=False, nleft=1, nright=1):
        """
        Pull line across plot
        Parameters
        ----------
        fig, ax : names of figures and axis where to follow the line
        release_flag (boolean): If True, end of line when left button released \
                                if False, end of line triggered by pressing right button
        nleft (int): if 0 start line for negative direction at origin \
                     if not, start line at the position of first click
        nright (int): if 0 start line for positive direction at origin \
                     if not, start line at the position of first click
                     
        Returns
        -------
        event.button

        Mouse button pressed to exit sunction (may be 2 for wheel ar 3
        for right button)
        
        Left button adds new point. right button finishes. Wheel erases last
        clicked point or, if no points is available, returns
        """
        global figure

        def onPress(event):
            global figure
            self.line_click = False
            if event.button == 1:  # left mouse button is pressed
                if event.xdata is None or event.ydata is None:
                    self.mouse = 1
                    self.x_event = event.xdata
                    self.y_event = event.ydata
                    return
                if len(self.coor_x) == 0:
                    if (event.xdata < 0 and nleft == 0) or (
                        event.xdata >= 0 and nright == 0
                    ):
                        self.start = [0, 0]
                        self.coor_x.append(0)
                        self.coor_y.append(0)
                    else:
                        self.start = [event.xdata, event.ydata]
                        self.coor_x.append(event.xdata)
                        self.coor_y.append(event.ydata)
                    if event.xdata < 0:
                        self.side = -1
                    else:
                        self.side = 1
                    self.background = figure.canvas.copy_from_bbox(figure.bbox)
                self.coor_x.append(
                    event.xdata
                )  # set starting point initially also as end point
                self.coor_y.append(event.ydata)
                self.canvas = self.line.figure.canvas
                self.axl = self.line.axes
                self.line.set_data(self.coor_x, self.coor_y)
                self.axl.draw_artist(self.line)
                self.cidmotion = self.line.figure.canvas.mpl_connect(
                    "motion_notify_event", onMotion
                )  # set action on mouse motion
                if release_flag:
                    self.cidrelease = self.line.figure.canvas.mpl_connect(
                        "button_release_event", onRelease
                    )  # set action on mouse release
            elif event.button == 3:  # if right button is pressed, finish
                self.mouse = 3
                self.x_event = event.xdata
                self.y_event = event.ydata
                self.line.figure.canvas.mpl_disconnect(self.cidpress)
                self.line_click = False
                # try:
                self.line.figure.canvas.mpl_disconnect(self.cidmotion)
                if len(self.coor_x) > 0:
                    self.line_click = True
                # except:
                #     self.line_click = False
                self.line.set_animated(False)
                if self.line_click:
                    figure.canvas.restore_region(self.background)
                self.background = None
                self.released = True
                return
            else:  # Wheel is pressed, erase last point
                if len(self.coor_x) > 0:
                    print(f"Erase point ({self.coor_x[-1]},{self.coor_y[-1]})")
                    del self.coor_x[-1]
                    del self.coor_y[-1]
                    self.canvas = self.line.figure.canvas
                    self.axl = self.line.axes
                    self.line.set_data(self.coor_x, self.coor_y)
                    self.axl.draw_artist(self.line)
                    self.cidmotion = self.line.figure.canvas.mpl_connect(
                        "motion_notify_event", onMotion
                    )  # set action on mouse motion
                else:
                    self.mouse = 2
                    self.x_event = event.xdata
                    self.y_event = event.ydata
                    self.line.figure.canvas.mpl_disconnect(self.cidpress)
                    self.line_click = False
                    # try:
                    self.line.figure.canvas.mpl_disconnect(self.cidmotion)
                    if len(self.coor_x) > 0:
                        self.line_click = True
                    # except:
                    #     self.line_click = False
                    self.line.set_animated(False)
                    if self.line_click:
                        figure.canvas.restore_region(self.background)
                    self.background = None
                    self.released = True
                    return

        def onRelease(event):
            # If line finishes when button is released do this here
            global figure
            self.line.figure.canvas.mpl_disconnect(self.cidpress)
            self.line.figure.canvas.mpl_disconnect(self.cidmotion)
            self.line.figure.canvas.mpl_disconnect(self.cidrelease)
            self.line.set_animated(False)
            if len(self.coor_x) > 0:
                figure.canvas.restore_region(self.background)
            self.background = None
            self.released = True
            return False

        def onMotion(event):
            global figure
            if event.xdata is None or event.ydata is None:
                return False
            # set second point of line as actual mouse position
            self.coor_x[-1] = event.xdata
            self.coor_y[-1] = event.ydata
            self.line.set_data(self.coor_x, self.coor_y)  # Draw new line
            self.line.set_color("k")
            self.canvas = self.line.figure.canvas
            self.axl = self.line.axes
            figure.canvas.restore_region(self.background)
            self.axl.draw_artist(self.line)
            self.canvas.blit(self.axl.bbox)
            return True

        # set flags and initialize coordinates
        self.released = False
        self.start = []
        self.coor_x = []
        self.coor_y = []
        figure = fig
        (self.line,) = ax.plot(self.coor_x, self.coor_y, "k", animated=True)
        self.cidpress = self.line.figure.canvas.mpl_connect(
            "button_press_event", onPress
        )
        # As long as release flag is not set listen to events
        while self.released is not True:
            QtCore.QCoreApplication.processEvents()

    def eventFilter(self, obj, event):
        """
        Function interprets keyboard events. It reacts only on key release !
        TODO

        Parameters
        ----------
        obj : TYPE Object
            The object that contains the window where the cursor should be so
            the keystroke is captured, i.e. Main.window

        event : Qt event
            Any QT event, although only key-release events are treated.

        Returns
        -------
        None.

        """
        #        from PyQt5.QtGui import QWindow
        if event.type() == QtCore.QEvent.KeyRelease and obj is self.window:
            pass
        # In main function, only "+" and "-" keys have a meaning, changing amplitude
        #             if self.function == "main":
        #                 if event.key() == 43:
        #                     self.window.changeAmp(1)
        #                 elif event.key() == 45:
        #                     self.window.changeAmp(-1)
        #                 elif event.key() == 16777234:
        # #     If Left arrow is pressed, choose next block to be plotted to the left
        #                     self.window.nextBlock(-1)
        #                 elif event.key() == 16777236:
        # #     If Right arrow is pressed, choose next block to be plotted to the right
        #                     self.window.nextBlock(1)
        # # in pickMove function, the keyboard arrows are checkes with possible SHFT and CTRL
        #             elif self.function == "pick_move":
        #                 if event.key() == 16777234:
        # #     If Left arrow is pressed, choose next trace to the left
        #                     self.window.shiftPickTrace(-1)
        #                 elif event.key() == 16777236:
        # #     If Right arrow is pressed, choose next trace to the right
        #                     self.window.shiftPickTrace(1)
        # #     If Up key is pressed, move pick upwards
        #                 elif event.key() == 16777235:
        #                     if event.modifiers() & Qt.ShiftModifier:
        # #       With shift by 10 samples
        #                         self.window.movePick(10)
        #                     elif event.modifiers() & Qt.ControlModifier:
        # #       With CTRL by 1 milisecond
        #                         nshift = 0.001/self.data.dt
        #                         self.window.movePick(nshift)
        #                     else:
        # #       Without modifier by 1 sample
        #                         self.window.movePick(1)
        # #     If Down key is pressed, move pick upwards
        #                 elif event.key() == 16777237:
        #                     if event.modifiers() & Qt.ShiftModifier:
        # #       With shift by 10 samples
        #                         self.window.movePick(-10)
        #                     elif event.modifiers() & Qt.ControlModifier:
        # #       With CTRL by 1 milisecond
        #                         nshift = 0.001/self.data.dt
        #                         self.window.movePick(-nshift)
        #                     else:
        # #       Without modifier by 1 sample
        #                         self.window.movePick(-1)
        # # in pickMove function, the keyboard arrows are checkes with possible SHFT and CTRL
        #             elif self.function == "uncertainty":
        #                 if event.key() == 16777234:
        # #     If Left arrow is pressed, choose next trace to the left
        #                     self.window.shiftPickTrace(-1)
        #                 elif event.key() == 16777236:
        # #     If Right arrow is pressed, choose next trace to the right
        #                     self.window.shiftPickTrace(1)
        # #     If Up key is pressed, move pick upwards
        #                 elif event.key() == 16777235:
        #                     if event.modifiers() & Qt.ShiftModifier:
        # #       With shift by 10 samples
        #                         self.window.changeUnc(10)
        #                     elif event.modifiers() & Qt.ControlModifier:
        # #       With CTRL by 1 milisecond
        #                         nshift = 0.001/self.data.dt
        #                         self.window.changeUnc(nshift)
        #                     else:
        # #       Without modifier by 1 sample
        #                         self.window.changeUnc(1)
        # #     If Down key is pressed, move pick upwards
        #                 elif event.key() == 16777237:
        #                     if event.modifiers() & Qt.ShiftModifier:
        # #       With shift by 10 samples
        #                         self.window.changeUnc(-10)
        #                     elif event.modifiers() & Qt.ControlModifier:
        # #       With CTRL by 1 milisecond
        #                         nshift = 0.001/self.data.dt
        #                         self.window.changeUnc(-nshift)
        #                     else:
        # #       Without modifier by 1 sample
        #                         self.window.changeUnc(-1)
        #             elif self.function == "vfilt":
        #                 if self.window.verticalSlider.isVisible() and event.key() == 16777220:
        #                     self.window.verticalSlider.setVisible(False)
        # # Check whether the letter C has been types in tomography result window
        # #       No idea why, but after typing C the first time, the event gets associated to
        # #       2 types of objects, rP.newWindow and PyQt5.QtGui.QWindow and the dialogue
        # #       window appears again after plotting the tomography results with the new
        # #       settings. Only when the object type is QWindow, the event should be
        # #       accepted.
        # # This works as long as only one key is pressed (so, do not press SHFT+C), just "c"
        # elif event.type() == QtCore.QEvent.KeyRelease and self.function == "linePlot":
        #     if type(obj) is QWindow:
        #         if event.key() == 67 or event.key() == 16777248:
        #             self.utilities.invCol()
        return QtWidgets.QWidget.eventFilter(self, obj, event)

    def Handler(self, signal_received, frame):
        """
        Handles CTRL-C key stroke

        Parameters
        ----------
        signal_received : CTRL-C keystroke
        frame : No idea

        Returns
        -------
        None.

        """
        self.close_app()

    def save_lineaments(self):
        """
        If lineaments were picked, save the information into file lineaments.dat

        The file has the format:
        #key (including the character #)
        x y

        Actually, key may be "magnetic tilt" or "gravity tilt", meaning
        lineaments having been traced on tilt angle maps of gravity or
        megnetic data.

        x y line is repeated for every point defining a lineament.
        File is finished with "#END"

        Returns
        -------
        None.

        """
        if self.nlineaments > 0:
            with open("lineaments.dat", "w", encoding="utf-8") as fo:
                for lin in self.lineaments.values():
                    fo.write(f"#{lin['type']}\n")
                    for i in range(len(lin["x"])):
                        fo.write(f"{lin['x'][i]:0.1f}  {lin['y'][i]:0.1f}\n")
                fo.write("#END\n")

    def close_app(self):
        """
        Finishes application:

        Stores picks into file pick.dat
        Stores information concerning possibly modified traces (muted or sign-inversed)
        Deletes unneeded folder if tomography was calculated
        Closes window
        """
        #        import os
        choice = QtWidgets.QMessageBox.question(
            None,
            "Confirm",
            "Are you sure?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if choice == QtWidgets.QMessageBox.Yes:
            # if max(self.traces.amplitudes) != self.window.general_sign:
            #     if os.path.exists("receivers_modify.dat"):
            #         os.remove("receivers_modify.dat")
            #     with open("receivers_modify.dat","w") as fh:
            #         for i in range(self.traces.number_of_traces):
            #             if self.traces.amplitudes[i] != self.window.general_sign:
            #                 a = self.traces.amplitudes[i]*\
            #                     self.window.general_sign
            #                 rec = self.traces.receiver[i]
            #                 fh.write(f"{rec} {a:0.0f}\n")
            print("application finished.\nClose console if you are working with Spyder")
            # if self.utilities.w_tau:
            #     self.utilities.w_tau.close()
            # if self.utilities.w_env:
            #     self.utilities.w_env.close()
            # if self.utilities.w_tomo:
            #     self.utilities.w_tomo.close()
            # if self.utilities.w_fcol:
            #     self.utilities.w_fcol.close()
            # if self.utilities.w_amp:
            #     self.utilities.w_amp.close()
            # if self.window.w_anim:
            #     self.window.w_anim.close()
            # if self.window.w_picks:
            #     self.window.w_picks.close()
            # if np.amax(self.traces.npick) > 0:
            #     self.traces.storePicks()
            # try:
            #     os.rmdir(self.utilities.path)
            # except:
            #     pass
            self.w.close()
            QtWidgets.QApplication.quit()
            return True
        return False
