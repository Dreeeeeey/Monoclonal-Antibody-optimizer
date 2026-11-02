# CLASSES
# Define custom classes in this file.


import numpy as np

import src.constants as c


class Batch:


    def __init__(self, batch_id, glucose_0, glutamine_0, vcd_0, time_h, data):
        self.batch_id = batch_id
        self.glucose_0 = glucose_0  # [mM]
        self.glutamine_0 = glutamine_0  # [mM]
        self.vcd_0 = vcd_0  # [cells/mL]
        self.time_h = time_h  # array (time,)
        self.data = data  # array (time, feature), order = c.FEATURES

    def display_info(self):
        print(f"Batch ID: {self.batch_id:4d} | "
              f"Glucose: {self.glucose_0:5.1f} mM | "
              f"Glutamine: {self.glutamine_0:5.2f} mM | "
              f"VCD: {self.vcd_0:.3g} cells/mL | "
              f"{len(self.time_h)} time points")


class FeatureScaler:
    """Standardizes every process variable independently (mean 0, SD 1)"""

    def fit(self, series_list):
        x = np.vstack(series_list)
        self.mean_ = x.mean(axis=0)
        sd = x.std(axis=0)
        self.scale_ = np.where(sd == 0, 1.0, sd)  # avoid division by zero
        # Concentrations and cell densities cannot be negative. This is the
        # value of 0 expressed in standardized units (used to clip forecasts).
        self.lower_ = (0.0 - self.mean_) / self.scale_
        # Safety bound: a forecast may not exceed twice the largest value seen
        # in training.
        self.upper_ = (2.0 * x.max(axis=0) - self.mean_) / self.scale_
        return self

    def transform(self, x):
        return (x - self.mean_) / self.scale_

    def inverse_transform(self, z):
        return z * self.scale_ + self.mean_

    def transform_list(self, series_list):
        return [self.transform(s) for s in series_list]
