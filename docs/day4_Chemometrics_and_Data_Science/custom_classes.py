import copy
import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import pearsonr

import io
from zipfile import ZipFile
import owncloud
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class ImportData:
    """
    Custom Class for importing data from the cloud
    """
    def __init__(self, zip_filename, cloud_folder_location=None, cloud_folder_password=''):
        """
        Initialization of ImportData

        :param zip_filename: name of a ZIP file on the cloud
        :param cloud_folder_location: link to the public folder with the ZIP file on the cloud
        :param cloud_folder_password: password for the public folder
        """
        self.zip_filename = zip_filename
        self.cloud_folder_location = cloud_folder_location
        self.cloud_folder_password = cloud_folder_password

    def set_zip_filename(self, zip_filename):
        """
        Set a different value for a ZIP file
        """
        self.zip_filename = zip_filename

    def download_zip(self):
        """
        Downloads zip_filename from the cloud if it does not exist in the working directory yet
        """
        if self.cloud_folder_location is None:
            raise ValueError('You must specify a cloud folder location to download from it')
        elif os.path.exists(self.zip_filename):
            print(f"File {self.zip_filename} already exists")
        else:
            cloud_data = owncloud.Client.from_public_link(self.cloud_folder_location,
                                                          self.cloud_folder_password)
            cloud_filenames = [f.path for f in cloud_data.list(".", "infinity") if f.file_type == 'file']
            print(f"Available files: {cloud_filenames}")
            download_status = cloud_data.get_file(self.zip_filename)
            print(f"File {self.zip_filename} downloaded successfully" if download_status else "Download failed")

    def parse_zip_scan(self, shape=(130, 100)):
        """
        Loads data from DAT files within zip_filename, assuming that there is only one spectral map file
        and that the wavenumber axis is saved in a separate file

        :return: tuple with 3 elements: wavenumber axis (1d, numeric), spectral matrix (2d, numeric),
            and the scan shape
        """
        with ZipFile(self.zip_filename, 'r') as zf:
            filenames = zf.namelist()
            for fn in filenames:
                if fn.endswith("dat"):
                    file_bytes = zf.read(fn)
                    file_object = io.BytesIO(file_bytes)
                    file_object.seek(0)
                    dat = np.loadtxt(file_object)
                    if fn.startswith("X"):
                        wn = dat
                    else:
                        mat = dat

        return wn, mat.reshape((int(len(mat)/len(wn)), len(wn))), shape

    def parse_zip_spectra(self):
        """
        Loads data from TXT files within zip_filename and parses loaded data
        assuming that the directory names within the ZIP file represent class labels

        :return: tuple with 3 elements: wavenumber axis (1d, numeric), spectral matrix (2d, numeric),
            and class labels (1d, str)
        """
        data = dict()
        with ZipFile(self.zip_filename, 'r') as zf:
            filenames = zf.namelist()
            for fn in filenames:
                if fn.endswith("txt"):
                    file_bytes = zf.read(fn)
                    file_object = io.BytesIO(file_bytes)
                    file_object.seek(0)
                    data[fn] = np.loadtxt(file_object)
        print(f"Loaded {len(data)} spectra from {self.zip_filename}")

        data_cube = np.asarray(list(data.values()))
        wn = data_cube[:, :, 0].mean(0)
        mat = data_cube[:, :, 1]
        classes = [os.path.dirname(fn) for fn in data.keys()]
        return wn, mat, classes


class SpectralData:
    """
    Custom Class for spectral data
    """

    def __init__(self, x, y, classes=None):
        """
        Initialization of SpectralData

        :param x: 1d array of X-axis
        :param y: 2d array with spectra in rows; number of columns should match length of x
        :param classes: class labels used for plotting
        """
        self.x = np.array(x).astype(float)
        self.y = np.atleast_2d(y).astype(float)
        self.steps_done = []
        self.label = 'Spectra'
        self.classes = None if classes is None else np.asarray(classes)
        self.xlabel = 'Wavenumber / $\mathregular{cm^{-1}}$'
        self.ylabel = 'Intensity / arb.u.'
        self.color = '#4363d8'
        self.color_classes = sns.color_palette()
        print(f"Spectral matrix shape: {self.y.shape}")
        print(f"Wavenumber axis length: {len(self.x)}, min: {min(self.x)}, max: {max(self.x)}")
        if self.classes is not None:
            print("Classes: {}, counts: {}".format(*np.unique(self.classes, return_counts=True)))

    def get_steps(self):
        """
        Get a set of performed preprocessing steps

        :return: DataFrame
        """
        return pd.DataFrame(self.steps_done)

    def quality_corr(self, threshold=0.95, ref=None, inplace=True):
        """
        Quality filter based on the correlation with the reference

        :param threshold: minimal Pearson's correlation value needed to pass the filter
        :param ref: reference spectrum (if None, an average spectrum is used)
        :param inplace: indicates if the operation is performed inplace or on a copy of the object
        :return: self or modified copy (depending on the inplace parameter)
        """
        if ref is None:
            ref = self.y.mean(axis=0)
        corr = np.apply_along_axis(lambda m: pearsonr(m, ref)[0], 1, self.y)
        which = corr >= threshold

        res = self if inplace else copy.deepcopy(self)
        res.y = self.y[which, :]
        res.steps_done.append({"step": "quality", "method": "correlation",
                               "args": {"threshold": threshold, "ref": ref, "which": which}})
        return res

    def normalize(self, method="vector", inplace=True):
        """
        Normalize spectra

        :param method: normalization method. Can be "vector" (or "l2"), "snv", "max", "minmax", "absdev" (or "l1"),
            "integr"
        :param inplace: indicates if the operation is performed inplace or on a copy of the object
        :return: self or modified copy (depending on the inplace parameter)
        """
        m = method.lower()
        if m in ["vector", "l2"]:
            y = self.y / np.sqrt((self.y ** 2).sum(1))[:, None]
        elif m == "snv":
            y = self.y - self.y.mean(1)[:, None]
            y = y / y.std(1)[:, None]
        elif m == "max":
            y = self.y / self.y.max(1)[:, None]
        elif m == "minmax":
            y = self.y - self.y.min(1)[:, None]
            y = y / y.max(1)[:, None]
        elif m in ["absdev", "l1"]:
            y = self.y / np.abs(self.y).sum(1)[:, None]
        elif m in ["integr"]:
            y = self.y / self.y.sum(1)[:, None]
        else:
            raise ValueError("Invalid normalization method: " + str(m))

        res = self if inplace else copy.deepcopy(self)
        res.y = y
        res.ylabel = "Normalized intensity"
        res.steps_done.append({"step": "normalization", "method": m, "args": {}})
        return res

    def crop(self, ranges_list, inplace=True):
        """
        Crop X-range of the spectra

        :param ranges_list: list of list (or similar) with specified ranges to keep
        :param inplace: indicates if the operation is performed inplace or on a copy of the object
        :return: self or modified copy (depending on the inplace parameter)
        """
        rng = np.asarray([(self.x >= min(r)) & (self.x <= max(r)) for r in ranges_list]).any(0)
        x = self.x[rng]
        y = self.y[:, rng]

        res = self if inplace else copy.deepcopy(self)
        res.x = x
        res.y = y
        res.steps_done.append({"step": "crop", "method": "include_endpoints",
                               "args": {"ranges": ranges_list}})
        return res

    def savgol(self, window=5, inplace=True):
        """
        Smoothing using Savitzky-Golay filter

        :param window: parameter passed to scipy.signal.savgol_filter function
        :param inplace: indicates if the operation is performed inplace or on a copy of the object
        :return: self or modified copy (depending on the inplace parameter)
        """
        y = savgol_filter(self.y, window, 2)
        res = self if inplace else copy.deepcopy(self)
        res.y = y
        res.steps_done.append({"step": "smoothing", "method": "smooth_savgol",
                               "args": {"window": window}})
        return res

    def snip(self, iterations=40, smoothing_window=11, return_baseline=False, inplace=True):
        """
        SNIP clipping algorythm for baseline correction

        :param iterations: number of SNIP iterations
        :param smoothing_window: window of Savitzky-Golay filter applied at the 1st iteration
            (applied if the window is > 2)
        :param return_baseline: indicates if the baseline or corrected spectra (default) should be returned
        :param inplace: indicates if the operation is performed inplace or on a copy of the object
        :return: self or modified copy (depending on the inplace parameter)
        """
        bg = savgol_filter(self.y, smoothing_window, 2) if smoothing_window > 2 else self.y.copy()
        bg[bg < 0] = 0
        bg = np.log(np.log(np.sqrt(bg + 1) + 1) + 1)  # log-log-square_root (LLS) operator
        for p in range(1, iterations + 1, 1):  # optimized snip loop (about 12 times faster)
            bg[:, p:-p] = np.minimum(bg[:, p:-p], (bg[:, p * 2:] + bg[:, :-p * 2]) / 2)
        bg = (np.exp(np.exp(bg) - 1) - 1) ** 2 - 1  # back transformation of LLS operator
        if return_baseline:
            return self.x, bg

        y = self.y - bg
        res = self if inplace else copy.deepcopy(self)
        res.y = y
        res.steps_done.append({"step": "baseline", "method": "SNIP",
                               "args": {"iterations": iterations, "smoothing_window": smoothing_window}})
        return res

    def plot(self, idx=None, title=None, xlabel=None, ylabel=None, color=None,
             alpha=0.3, figsize=(8, 4), fontsize=15, legend_loc="upper left"):
        """
        Plot mean spectra with SD

        :param idx: subset of the data to be plotted (if None, the whole data set is used)
        :param title: title of the plot (if None, self.label is used)
        :param xlabel: X-label of the plot (if None, self.xlabel is used)
        :param ylabel: Y-label of the plot (if None, self.ylabel is used)
        :param color: color of the trace (if None, self.color is used)
        :param alpha: transparency level of the SD area
        :param figsize: figure size passed to seaborn.set(rc={'figure.figsize': figsize})
        :param fontsize: font size passed to matplotlib.pyplot calls
        :param legend_loc: legend location (legend_loc, see 'loc' parameter of matplotlib.pyplot.legend)
        """
        if title is None:
            title = self.label
        if xlabel is None:
            xlabel = self.xlabel
        if ylabel is None:
            ylabel = self.ylabel
        if color is not None:
            self.color = color

        steps_str = ", ".join(s["step"] for s in self.steps_done) if self.steps_done else "raw"
        y = self.y[idx, :] if idx else self.y

        if self.classes is None:
            df = pd.DataFrame({"x": self.x, f"{self.label} (mean)": y.mean(0),
                               f"{self.label} (SD)": y.std(0)})
            colors = [self.color]
            trace_names = [self.label]
        else:
            df = pd.DataFrame({"x": self.x})
            trace_names = sorted(list(set(self.classes)))
            for cl in trace_names:
                w_cl = self.classes == cl
                df[f"{cl} (mean)"] = y[w_cl, :].mean(0)
                df[f"{cl} (SD)"] = y[w_cl, :].std(0)
            df = pd.DataFrame(df)
            colors = self.color_classes

        # define figure size and style of plot
        sns.set(rc={'figure.figsize': figsize})
        patches = []
        with sns.axes_style('whitegrid'):
            for i, cl in enumerate(trace_names):
                col = colors[i % len(colors)]
                # Create an axis object for your spectral plot and color patch for legend label
                ax = sns.lineplot(data=df, x='x', y=f"{cl} (mean)", color=col)
                # Create a +/- devidation fill between top and bottom of the mean spectra by adding
                # or subtracting standard deviation from the mean
                ax.fill_between(df['x'], df[f"{cl} (mean)"],
                                df[f"{cl} (mean)"] + df[f"{cl} (SD)"],
                                alpha=alpha, color=col)
                ax.fill_between(df['x'], df[f"{cl} (mean)"] - df[f"{cl} (SD)"],
                                df[f"{cl} (mean)"], alpha=alpha, color=col)
                patches.append(mpatches.Patch(color=col, label=f"{cl} (mean)"))
                patches.append(mpatches.Patch(color=col, alpha=alpha, label=f"{cl} (SD)"))

            # define legend, X, Y, Tick labels and fontsize
            plt.legend(handles=patches, fontsize=fontsize, loc=legend_loc)
            plt.ylabel(ylabel, fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.xlabel(xlabel, fontsize=fontsize)
            plt.title(f"{title}: {steps_str}", fontsize=fontsize)
            plt.show()
