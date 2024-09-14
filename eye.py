#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 15:16:39 2024

@author: Wei Du
"""

import itertools
import os
from collections import OrderedDict

import matplotlib as mpl
import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot as plt

plt.rcdefaults()

if True:
    plt.rcParams.update(
        {
            "lines.linewidth": 2,
            "figure.facecolor": "#D7D9D6",
            "axes.facecolor": "#D7D9D6",
            "axes.prop_cycle": mpl.rcsetup.cycler(
                "color",
                [
                    "#02865B",
                    "#2E639D",
                    "#5CAECE",
                    "#786067",
                    "#8F413B",
                    "#968B5C",
                    "#7A7B8D",
                    "#65544F",
                    "#1F2327",
                    "#506D76",
                    "#A66D56",
                ],
            ),
        }
    )
    COLOR_WFM = "#8F413B"
    COLOR_EYE = "#02865B"
    COLOR_MKR = "#1F2327"
    COLOR_REFLINE = "#2E639D"
    COLOR_LOWLIGHT = COLOR_MKR
    COLOR_HIGHLIGHT = COLOR_EYE

if False:
    plt.rcParams.update(
        {
            "lines.linewidth": 2,
            "figure.facecolor": "#EFE5CE",
            "axes.facecolor": "#EFE5CE",
            "axes.prop_cycle": mpl.rcsetup.cycler(
                "color",
                [
                    # dunhuang colors
                    "#02865B",
                    "#994B42",
                    "#4E3227",
                    "#A9A8C2",
                    "#EDDA8A",
                    "#040308",
                    "#91331D",
                    "#3C839E",
                    "#C49061",
                    "#346C9D",
                    "#5D524F",
                ],
            ),
        }
    )
    COLOR_WFM = "#3F749F"
    COLOR_EYE = "#02865B"
    COLOR_MKR = "#040308"
    COLOR_REFLINE = "#91331D"
    COLOR_LOWLIGHT = COLOR_REFLINE
    COLOR_HIGHLIGHT = COLOR_EYE


class EYE:
    def __init__(self, input_path, output_path, wfm_file, ref_file, F, Fs):
        self.input_path = input_path
        self.output_path = output_path
        self.wfm_file = wfm_file
        self.ref_file = ref_file
        self.F = F
        self.Fs = Fs
        self.os = self.Fs / self.F
        # hard coded properties
        self.en_debug_plot = True
        self.eye_lw = 0.2  # doesn't apply to animation
        self.eye_window_length = 1.6  # 1.6T in eye window
        self.eye_alpha = 0.3

        # properties unknown in the beginning, will be updated later
        self.ymin = None
        self.ymax = None
        self.ref_transition_tags = None
        self.wfm_t = None
        self.wfm_v = None
        self.ref_t = None
        self.ref_v = None
        self.num_cycles = None
        self.num_points_per_cycle = None
        self.tie_t = None
        self.tie = None
        self.tie_jitter_pkpk_pS = None

        self.eye_start_i = None
        self.eye_samples_per_T = None
        self.eye_samples_per_Window = None
        self.eye_t = None
        self.eye_db = None
        self.eye_db_i = None
        self.eye_db_transition_tags = None
        self.eye_datarate_Gbps = None
        self.eye_jitter_pS_for_tag = None
        self.eye_jitter_pkpk_pS_for_tag = None
        self.eye_jitter_pS = None  # zero crossing
        self.eye_jitter_pkpk_pS = None  # zero crossing
        self.eye_width_pS = None

        self.wfm_rise_i = None
        self.wfm_rise_t = None
        self.wfm_rise_v = None
        self.wfm_fall_i = None
        self.wfm_fall_t = None
        self.wfm_fall_v = None
        self.ref_rise_i = None
        self.ref_rise_t = None
        self.ref_rise_v = None
        self.ref_fall_i = None
        self.ref_fall_t = None
        self.ref_fall_v = None
        self.ref_clk_rising_edges_t = None
        self.ref_clk_falling_edges_t = None

    def read_wfm_file(self):
        df = pd.read_csv(
            os.path.join(self.input_path, self.wfm_file), sep=" ", header=None
        )
        self.wfm_t = np.array(df[0])
        self.wfm_v = np.array(df[1])
        print(f"read {self.wfm_file}, update wfm time and voltage")
        print()

    def _upsample(self, y, up):
        """https://docs.scipy.org/doc/scipy/tutorial/interpolate/1D.html#tutorial-interpolate-1dsection"""
        n = len(y)
        n_new = int(len(y) * up)

        x = np.linspace(0, 1, n)

        bspl = scipy.interpolate.make_interp_spline(x, y, k=3)

        x_new = np.linspace(0, 1, n_new)

        y_new = bspl(x_new)

        if self.en_debug_plot:
            plt.figure(figsize=(24, 8))
            plt.plot(x[:1000], y[:1000], "-x", lw=1, label="ori")
            plt.plot(
                x_new[: int(1000 * up)],
                y_new[: int(1000 * up)],
                "-x",
                label=f"up {up}x",
            )
            plt.legend(loc="upper right")
            plt.title("upsampled")

        return y_new

    def upsample_wfm(self):
        if np.mod(self.Fs, self.F) > 0:
            print(
                f"F = {self.F:.2e}, Fs = {self.Fs:.2e}, fractional sampling, do interpolation"
            )

            old_Fs = self.Fs
            self.os = int(np.ceil(old_Fs / self.F / 10) * 10)  # new os
            self.Fs = self.F * self.os  # new Fs

            up = self.Fs / old_Fs  # upsampling

            print(
                f"F = {self.F:.2e}, Fs = {self.Fs:.2e}, up by {up}x, new os {self.os}"
            )

            self.wfm_v = self._upsample(self.wfm_v, up)
        else:
            print("integer sampling, no interpolation")

        self.wfm_t = np.arange(len(self.wfm_v)) * (1 / self.Fs * 1e9)  # nS

        print("update wfm time")
        print()

    def read_ref_file(self):
        df = pd.read_csv(os.path.join(self.input_path, self.ref_file), header=None)
        print(f"read ref file {self.ref_file}")
        return np.array(df[2]) * 2 - 1

    def roll_ref(self, ref):
        """do correlation, rorate ref to align wfm
        it could be the ref or the inverted ref that align the wfm
        """

        ref_one_cycle = np.repeat(ref, self.os)
        print(f"upsample ref samples {self.os}x")

        self.num_points_per_cycle = len(ref_one_cycle)
        wfm_one_cycle = self.wfm_v[: self.num_points_per_cycle]
        print(f"update num points per cycle {self.num_points_per_cycle}")

        ref_one_cycle_inv = -1 * ref_one_cycle

        c0 = np.correlate(wfm_one_cycle, ref_one_cycle, mode="full")
        c1 = np.correlate(wfm_one_cycle, ref_one_cycle_inv, mode="full")

        if max(c0) > max(c1):
            i = np.where(c0 == max(c0))[0][0]
            print(f"ref samples correlate wfm, roll ref {i} to align with wfm")
            new_ref = np.roll(ref_one_cycle, i)
        else:
            i = np.where(c1 == max(c1))[0][0]
            print(
                f"inverted ref samples correlate wfm, roll inverted ref {i} to align with wfm"
            )
            new_ref = np.roll(ref_one_cycle_inv, i)
        print()
        return new_ref

    def parse_wfm_ref_files(self):
        self.read_wfm_file()
        self.upsample_wfm()

        ref_one_cycle = self.roll_ref(self.read_ref_file())

        self.num_cycles = int(np.floor(len(self.wfm_v) / self.num_points_per_cycle))
        print(f"wfm has {self.num_cycles} complete cycles")

        self.wfm_v = self.wfm_v[: self.num_cycles * self.num_points_per_cycle]
        self.wfm_t = np.arange(len(self.wfm_v)) * 1 / self.Fs * 1e9  # nS
        print(f"drop wfm ending samples to keep {self.num_cycles} complete cycles")
        print("update wfm time")
        print()

        self.ref_v = np.tile(ref_one_cycle, self.num_cycles)
        self.ref_t = np.arange(len(self.ref_v)) * 1 / self.Fs * 1e9  # nS
        print(f"drop ref ending samples to keep {self.num_cycles} complete cycles")
        print("update ref time")
        print()

        wfm_min = np.floor(np.min(self.wfm_v) * 10) / 10
        wfm_max = np.ceil(np.max(self.wfm_v) * 10) / 10
        self.ymin = np.sign(wfm_min) * max(abs(wfm_min), abs(wfm_max))
        self.ymax = np.sign(wfm_max) * max(abs(wfm_min), abs(wfm_max))
        print(f"update ymin, ymax to {self.ymin:.2f}, {self.ymax:.2f} for plot")
        print()

        if self.en_debug_plot:
            plt.figure(figsize=(24, 8))
            plt.plot(ref_one_cycle, "-", label="genie pattern")
            for n in range(self.num_cycles):
                plt.plot(
                    self.wfm_v[
                        n
                        * self.num_points_per_cycle : (n + 1)
                        * self.num_points_per_cycle
                    ],
                    "-",
                    label=f"{n}",
                )
            plt.legend(loc="upper right")
            plt.title("rolled ref and wfm")
            fig_name = f"{os.path.basename(self.wfm_file)[:-4]}_check_wfm.png"
            plt.savefig(
                os.path.join(
                    self.output_path,
                    fig_name,
                )
            )
            print(f"plot {fig_name}")
        print()

    def gen_ref_transition_tags(self):
        i_transition = np.where(self.ref_v[:-1] != self.ref_v[1:])[0]
        i_transition_diff = [int(i) for i in np.diff(i_transition) / self.os]
        tags = [None] * len(self.ref_v)
        for i, transition in enumerate(i_transition):
            if i == 0 or i == len(i_transition) - 1:
                tags[transition] = None  # ignore the first and the last transition
            else:
                tags[transition] = (
                    f"{i_transition_diff[i-1]}T - {i_transition_diff[i]}T"
                )

        self.ref_transition_tags = np.array(tags)
        print("generate ref transition tags, 1T - 1T, 3T - 1T, etc.")
        print()

    def _gen_prbs7():
        """generate prbs7 bits"""
        bit = list()
        start = 1
        lfsr = start
        i = 1
        while True:
            fb = (lfsr >> 6) ^ (lfsr >> 5) & 1
            lfsr = ((lfsr << 1) + fb) & (2**7 - 1)
            bit.append(fb)
            print(i, lfsr, fb, bin(lfsr))
            if lfsr == start:
                print("repeat pattern length", i)
                break
            i = i + 1

        bit = [float(i) for i in bit]

        for i in range(2**7 - 1):
            bit[i] = 2 * (bit[i] - 0.5)

        plt.figure()
        plt.plot(bit)
        plt.title("PRBS")
        plt.show()

        u = scipy.signal.correlate(bit, bit)
        plt.figure()
        plt.plot(u)
        plt.title("PRBS corr")
        plt.show()

        # print("done!")

        return bit

    def _linear_interp_x(self, x0, y0, x1, y1, y):
        """return x for y"""
        return x0 + (y - y0) * (x1 - x0) / (y1 - y0)

    def _find_zero_crossing(self, x, y, find_rise_edge=True, interp=False):
        if find_rise_edge:
            zc_i = np.nonzero((y[1:] >= 0) & (y[:-1] < 0))[0]
        else:
            zc_i = np.nonzero((y[1:] <= 0) & (y[:-1] > 0))[0]

        if interp:
            zc_x = np.array(
                [self._linear_interp_x(x[i], y[i], x[i + 1], y[i + 1], 0) for i in zc_i]
            )
            zc_y = np.array([0]).repeat(len(zc_i))
        else:
            zc_x = x[zc_i]
            zc_y = y[zc_i]

        return zc_i, zc_x, zc_y

    def _find_zero_crossing_refer_ref(
        self, ref_i, wfm_t, wfm, find_rise_edge=True, interp=True
    ):
        """use zero crossing index of ref to find wfm"""
        _zc_i = np.array([])
        _zc_t = np.array([])
        _zc_v = np.array([])

        d = 10  # find within a small range

        for n in ref_i:
            _rng = np.arange(n - d, n + d)
            _i, _t, _v = self._find_zero_crossing(
                wfm_t[_rng], wfm[_rng], find_rise_edge, interp=interp
            )

            if len(_i) == 1 and len(_t) == 1 and len(_v) == 1:
                _zc_i = np.append(_zc_i, n - d + _i)
                _zc_t = np.append(_zc_t, _t)
                _zc_v = np.append(_zc_v, _v)
            else:
                _zc_i = np.append(_zc_i, None)
                _zc_t = np.append(_zc_t, None)
                _zc_v = np.append(_zc_v, None)

        return _zc_i, _zc_t, _zc_v

    def find_zero_crossing(self):
        # ----------------------------------------------
        # find zero crossing and do linear interpolation
        # ----------------------------------------------
        self.ref_rise_i, self.ref_rise_t, self.ref_rise_v = self._find_zero_crossing(
            self.ref_t, self.ref_v, find_rise_edge=True, interp=False
        )
        self.ref_fall_i, self.ref_fall_t, self.ref_fall_v = self._find_zero_crossing(
            self.ref_t, self.ref_v, find_rise_edge=False, interp=False
        )
        print("find ref zero crossing, seperate rising edges and falling edges")

        # rself.ef_rise_i = self.ref_rise_i[1:-1]
        # self.ref_rise_t = self.ref_rise_t[1:-1]
        # self.ref_rise_v = self.ref_rise_v[1:-1]
        # self.ref_fall_i = self.ref_fall_i[1:-1]
        # self.ref_fall_t = self.ref_fall_t[1:-1]
        # self.ref_fall_v = self.ref_fall_v[1:-1]

        (
            self.wfm_rise_i,
            self.wfm_rise_t,
            self.wfm_rise_v,
        ) = self._find_zero_crossing_refer_ref(
            self.ref_rise_i, self.wfm_t, self.wfm_v, find_rise_edge=True, interp=True
        )
        (
            self.wfm_fall_i,
            self.wfm_fall_t,
            self.wfm_fall_v,
        ) = self._find_zero_crossing_refer_ref(
            self.ref_fall_i, self.wfm_t, self.wfm_v, find_rise_edge=False, interp=True
        )
        print(
            "find wfm zero crossing, seperate rising edges and falling edges, refer ref corssing location"
        )

        if None in self.wfm_rise_i:
            print("!!! found None in wfm zero crossing from rising edge !!!")
            self.ref_rise_v = self.ref_rise_v[np.where(self.wfm_rise_i != None)]
            self.ref_rise_t = self.ref_rise_t[np.where(self.wfm_rise_i != None)]
            self.ref_rise_i = self.ref_rise_i[np.where(self.wfm_rise_i != None)]
            self.ref_fall_v = self.ref_fall_v[np.where(self.wfm_fall_i != None)]
            self.ref_fall_t = self.ref_fall_t[np.where(self.wfm_fall_i != None)]
            self.ref_fall_i = self.ref_fall_i[np.where(self.wfm_fall_i != None)]

        if None in self.wfm_fall_i:
            print(
                "!!! found None in wfm zero crossing from falling edge detected None !!!"
            )
            self.wfm_rise_v = self.wfm_rise_v[np.where(self.wfm_rise_i != None)]
            self.wfm_rise_t = self.wfm_rise_t[np.where(self.wfm_rise_i != None)]
            self.wfm_rise_i = self.wfm_rise_i[np.where(self.wfm_rise_i != None)]
            self.wfm_fall_v = self.wfm_fall_v[np.where(self.wfm_fall_i != None)]
            self.wfm_fall_t = self.wfm_fall_t[np.where(self.wfm_fall_i != None)]
            self.wfm_fall_i = self.wfm_fall_i[np.where(self.wfm_fall_i != None)]
        print()

    def gen_tie(self):
        first_zero_crossing_i = (
            self.ref_rise_i[0]
            if self.ref_rise_i[0] <= self.ref_fall_i[0]
            else self.ref_fall_i[0]
        )
        first_zero_crossing_wfm_t = (
            self.wfm_rise_t[0]
            if self.ref_rise_i[0] <= self.ref_fall_i[0]
            else self.wfm_fall_t[0]
        )
        print("determine first wfm zeros crossing location and time")
        self.ref_clk_rising_edges_t = (
            self.ref_rise_i - first_zero_crossing_i
        ) * 1 / self.Fs * 1e9 + first_zero_crossing_wfm_t
        self.ref_clk_falling_edges_t = (
            self.ref_fall_i - first_zero_crossing_i
        ) * 1 / self.Fs * 1e9 + first_zero_crossing_wfm_t
        print("generate reference clk edges and normalize to the first wfm crossing")

        tie_rise = (self.wfm_rise_t - self.ref_clk_rising_edges_t) * 1e3
        tie_fall = (self.wfm_fall_t - self.ref_clk_falling_edges_t) * 1e3
        print("determin tie, seperate rising edges and falling edges")

        if self.ref_rise_i[0] <= self.ref_fall_i[0]:
            self.tie_t = np.array(
                list(
                    itertools.chain.from_iterable(
                        zip(self.ref_clk_rising_edges_t, self.ref_clk_falling_edges_t)
                    )
                )
            )
            self.tie = np.array(
                list(itertools.chain.from_iterable(zip(tie_rise, tie_fall)))
            )
        else:
            self.tie_t = np.array(
                list(
                    itertools.chain.from_iterable(
                        zip(self.ref_clk_falling_edges_t, self.ref_clk_rising_edges_t)
                    )
                )
            )
            self.tie = np.array(
                list(itertools.chain.from_iterable(zip(tie_fall, tie_rise)))
            )
        print("combine tie to one list")
        self.tie_jitter_pkpk_pS = max(self.tie) - min(self.tie)
        print(f"TIE jiter PkPk = {self.tie_jitter_pkpk_pS}")

        if self.en_debug_plot:
            plt.figure(figsize=(24, 8))
            plt.plot(self.wfm_t, self.wfm_v, "-x", color=COLOR_WFM, label="wfm")
            plt.vlines(
                self.tie_t,
                ymin=self.ymin,
                ymax=self.ymax,
                color=COLOR_REFLINE,
                label="ref edge",
            )
            plt.scatter(
                self.wfm_rise_t,
                self.wfm_rise_v,
                s=64,
                marker="x",
                c=COLOR_MKR,
                zorder=99,
                label="zero crossing - rise",
            )
            plt.scatter(
                self.wfm_fall_t,
                self.wfm_fall_v,
                s=64,
                marker="+",
                c=COLOR_MKR,
                zorder=99,
                label="zero crossing - rise",
            )
            plt.xlim((self.wfm_t[0], self.wfm_t[self.num_points_per_cycle]))
            plt.legend(loc="upper right")
            plt.title("check zero crossing (zoom in)")
            fig_name = f"{os.path.basename(self.wfm_file)[:-4]}_check_zero_crossing.png"
            plt.savefig(
                os.path.join(
                    self.output_path,
                    fig_name,
                )
            )
            print(f"plot {fig_name}")
        print()

    def gen_eye_db(self):
        self.eye_start_i = int(
            min(self.wfm_rise_i[0], self.wfm_fall_i[0])
            + int((1.5 - (self.eye_window_length / 2)) * self.os)
            + 1
        )
        self.eye_samples_per_T = int(self.Fs / self.F)
        self.eye_samples_per_Window = int(self.eye_window_length * (self.Fs / self.F))
        print(f"update eye start location {self.eye_start_i}")
        print(f"update eye samples per UI {self.eye_samples_per_T}")
        print(
            f"update eye samples per eye Windows {self.eye_samples_per_Window} for {self.eye_window_length} UI"
        )

        self.eye_db = []
        self.eye_db_i = []
        self.eye_db_transition_tags = []

        for k in range(
            int((len(self.wfm_v) - self.eye_start_i) / self.eye_samples_per_Window) - 3
        ):
            slice_index = (
                self.eye_start_i
                + k * self.eye_samples_per_T
                + np.arange(self.eye_samples_per_Window)
            )

            self.eye_db_i.append(slice_index)
            self.eye_db.append(self.wfm_v[slice_index])

            # only tag the 1st edge of the eye
            n = int(len(slice_index) / 2)
            tag_index = np.where(self.ref_transition_tags[slice_index[:n]] != None)[0]
            if len(tag_index) > 0:
                self.eye_db_transition_tags.append(
                    self.ref_transition_tags[slice_index][tag_index[0]]
                )
            else:
                self.eye_db_transition_tags.append(None)

        self.eye_db = np.array(self.eye_db).T
        self.eye_db_i = np.array(self.eye_db_i).T
        self.eye_db_transition_tags = np.array(self.eye_db_transition_tags)
        print("update eye database")
        print("update eye database index (sample locations in wfm)")
        print("update eye transition tags")
        print()

    def gen_eye_jitter(self, tag=None):
        """use 1st 1/3 of eye database for 1st eye crossing, last 1/3 for 2nd eye crossing
        return jitter, jitter in pS
        and eye timing centered to zero crossing
        and datarate
        tag: "1T - 1T"
        """

        if self.eye_db_transition_tags is not None and tag is not None:
            tag_index = np.where(self.eye_db_transition_tags == tag)[0]
            s = f"eye jitter for {tag}"
            print("-" * len(s))
            print(s)
            print("-" * len(s))
            print()
            if len(tag_index) == 0:
                print(f"eye database doesn't have transition {tag}")
                return

            loc_eye_db = self.eye_db[:, tag_index]
        else:
            s = "eye jitter for all transitions"
            print("-" * len(s))
            print(s)
            print("-" * len(s))
            print()
            loc_eye_db = self.eye_db

        n_rows = len(loc_eye_db)
        c = int(loc_eye_db.size / n_rows)

        n0 = int(n_rows / 3)
        n1 = int(n_rows / 3 * 2)
        loc_eye_t = np.arange(n_rows) * 1 / self.Fs * 1e12  # pS
        x0 = loc_eye_t[:n0]
        x1 = loc_eye_t[n1:]

        thr = 0
        print(f"jitter on level {thr:.2f}V")

        jitter_t0 = []
        jitter_t1 = []

        for j in range(c):
            # 1st eye crossing

            y0 = loc_eye_db[:n0, j]

            if y0[0] < y0[int(len(y0) / 2)]:  # rising edge
                zc_i = np.nonzero((y0[1:] >= thr) & (y0[:-1] < thr))[0]
                if len(zc_i) > 0:
                    i = zc_i[0]
                    jitter_t0.append(
                        self._linear_interp_x(x0[i], y0[i], x0[i + 1], y0[i + 1], thr)
                    )

            if y0[0] > y0[int(len(y0) / 2)]:  # falling edge
                zc_i = np.nonzero((y0[1:] <= thr) & (y0[:-1] > thr))[0]
                if len(zc_i) > 0:
                    i = zc_i[0]
                    jitter_t0.append(
                        self._linear_interp_x(x0[i], y0[i], x0[i + 1], y0[i + 1], thr)
                    )

            # 2nd eye crossing

            y1 = loc_eye_db[n1:, j]

            if y1[-1 * int(len(y1) / 2)] < y1[-1]:  # rising edge
                zc_i = np.nonzero((y1[1:] >= thr) & (y1[:-1] < thr))[0]
                if len(zc_i) > 0:
                    i = zc_i[0]
                    jitter_t1.append(
                        self._linear_interp_x(x1[i], y1[i], x1[i + 1], y1[i + 1], thr)
                    )

            if y1[-1 * int(len(y1) / 2)] > y1[-1]:  # falling edge
                zc_i = np.nonzero((y1[1:] <= thr) & (y1[:-1] > thr))[0]
                if len(zc_i) > 0:
                    i = zc_i[0]
                    jitter_t1.append(
                        self._linear_interp_x(x1[i], y1[i], x1[i + 1], y1[i + 1], thr)
                    )

            # if t > 240:
            #     print("debug")

        crossing_t0 = np.mean(jitter_t0)
        crossing_t1 = np.mean(jitter_t1)
        print(f"first eye crossing at {crossing_t0:.2f}pS")
        print(f"second eye crossing at {crossing_t1:.2f}pS")

        if tag is None:
            self.eye_jitter_pS = np.array(jitter_t0) - crossing_t0
            self.eye_jitter_pkpk_pS = np.max(jitter_t0) - np.min(jitter_t0)
            print(f"eye jitter PkPk {self.eye_jitter_pkpk_pS:.2f}pS")
        else:
            self.eye_jitter_pS_for_tag = np.array(jitter_t0) - crossing_t0
            self.eye_jitter_pkpk_pS_for_tag = np.max(jitter_t0) - np.min(jitter_t0)
            print(f"eye jitter PkPk {self.eye_jitter_pkpk_pS_for_tag:.2f}pS")
        if tag is None:
            self.eye_t = (loc_eye_t - np.mean(jitter_t0)) / (1 / self.F * 1e12)
            self.eye_width_pS = crossing_t1 - crossing_t0
            self.eye_datarate_Gbps = 1 / (self.eye_width_pS * 1e-12) * 1e-9
            print(f"update eye width {self.eye_width_pS:.2f}pS")
            print(f"update eye datarate {self.eye_datarate_Gbps:.3f}Gbps")
        else:
            print("don't update eye width")
            print("don't update eye datarate")
        print()

    def draw_eye(self):
        print("draw eye")
        fig = plt.figure(figsize=(24, 9))
        grid = plt.GridSpec(1, 2, hspace=0.22, wspace=0.22)
        ax0 = fig.add_subplot(grid[0, 0])  # for wfm
        ax1 = fig.add_subplot(grid[0, 1])  # for eye
        ax0.plot(self.wfm_t, self.wfm_v, "-x", label="wfm")
        ax0.set_xlabel("nS")
        ax0.set_ylabel("mV")
        ax0.legend(loc="upper right")
        ax0.grid()
        ax1.plot(
            self.eye_t,
            self.eye_db,
            color=COLOR_EYE,
            alpha=self.eye_alpha,
            lw=self.eye_lw,
        )
        ax1.set_xlim([-0.5, 1.5])
        ax1.set_xticks([-0.25, 0, 0.25, 0.5, 0.75, 1.00, 1.25])
        ax1.set_xlabel("T")
        ax1.set_ylabel("mV")
        ax1.set_ylim((self.ymin, self.ymax))
        ax1.grid()
        ax1.set_title(
            f"{self.eye_datarate_Gbps:.3f} Gbps, {int(np.size(self.eye_db) / len(self.eye_db))} UI"
        )

        fig_name = os.path.basename(self.wfm_file)[:-4] + "_eye.png"
        plt.savefig(os.path.join(self.output_path, fig_name))
        print(f"plot {fig_name}")
        print()

    def draw_eye_and_wfm(self):
        # plot zoom in if wfm is too long
        if len(self.tie_t) > 15:
            n = 15
            print("draw eye and wfm, zoom in wfm")
        else:
            print("draw eye and wfm")
            n = None

        if n is not None:  # zoom in to 10 edges
            slice_index_wfm_rise = np.where(self.wfm_rise_t <= self.tie_t[n])
            slice_index_wfm_fall = np.where(self.wfm_fall_t <= self.tie_t[n])
            slice_index_tie = np.where(self.tie_t <= self.tie_t[n])
            slice_index_wfm = np.where(self.wfm_t <= self.tie_t[n])
            slice_index_ref = np.where(self.ref_t <= self.tie_t[n])

            loc_wfm = self.wfm_v[slice_index_wfm]
            loc_wfm_t = self.wfm_t[slice_index_wfm]
            loc_ref = self.ref_v[slice_index_ref]
            loc_ref_t = self.ref_t[slice_index_ref]
            loc_tie = self.tie[slice_index_tie]
            loc_tie_t = self.tie_t[slice_index_tie]
            loc_wfm_rise_t = self.wfm_rise_t[slice_index_wfm_rise]
            loc_wfm_fall_t = self.wfm_fall_t[slice_index_wfm_fall]
        else:
            loc_wfm = self.wfm_v
            loc_wfm_t = self.wfm_t
            loc_ref = self.ref_v
            loc_ref_t = self.ref_t
            loc_tie = self.tie
            loc_tie_t = self.tie_t
            loc_wfm_rise_t = self.wfm_rise_t
            loc_wfm_fall_t = self.wfm_fall_t

        fig = plt.figure(figsize=(24, 9))
        grid = plt.GridSpec(3, 3, hspace=0.22, wspace=0.22)
        ax0 = fig.add_subplot(grid[0, 0:2])  # for wfm
        ax1 = fig.add_subplot(grid[1, 0:2], sharex=ax0)  # for genie pattern
        ax2 = fig.add_subplot(grid[2, 0:2], sharex=ax0)  # for tie
        ax3 = fig.add_subplot(grid[2, 2])  # for tie or eye jitter hist
        ax4 = fig.add_subplot(grid[0:2, 2])  # for eye
        ax0.plot(
            loc_wfm_t,
            loc_wfm,
            # "-x",
            color=COLOR_WFM,
            label="wfm" if n is None else "wfm (zoom in)",
        )
        ax0.vlines(
            loc_tie_t,
            ymin=self.ymin,
            ymax=self.ymax,
            color=COLOR_REFLINE,
            label="ref edge",
        )
        ax0.scatter(
            loc_wfm_rise_t,
            [0] * len(loc_wfm_rise_t),
            s=64,
            marker="x",
            c=COLOR_MKR,
            zorder=99,
            label="zero crossing - rise",
        )
        ax0.scatter(
            loc_wfm_fall_t,
            [0] * len(loc_wfm_fall_t),
            s=64,
            marker="+",
            c=COLOR_MKR,
            zorder=99,
            label="zero crossing - fall",
        )
        ax0.set_ylabel("mV")
        ax0.legend(loc="upper right")
        ax0.grid()
        ax0.set_title(
            f"wfm length: {len(self.wfm_v)} points, {int(len(self.wfm_v) * (self.wfm_t[1] - self.wfm_t[0]))}nS"
        )
        ax1.plot(
            loc_ref_t,
            loc_ref,
            "-",
            color=COLOR_WFM,
            label="genie" if n is None else "genie (zoom in)",
        )
        # ax1.scatter(
        #     ref_rise_t,
        #     [0] * len(ref_rise_t),
        #     s=64,
        #     marker="x",
        #     c="k",
        #     zorder=99,
        #     label="zero crossing - rise",
        # )
        # ax1.scatter(
        #     ref_fall_t,
        #     [0] * len(ref_fall_t),
        #     s=64,
        #     marker="+",
        #     c="k",
        #     zorder=99,
        #     label="zero crossing - fall",
        # )

        ax1.set_ylim([-1.2, 1.2])
        ax1.set_yticks([-1, 0, 1])
        ax1.legend(loc="upper right")
        ax1.grid()

        ax2.plot(
            loc_tie_t,
            loc_tie,
            "-x",
            color=COLOR_WFM,
            label="tie" if n is None else "tie (zoom in)",
        )
        ax2.legend(loc="upper right")
        ax2.grid()
        ax2.set_xlabel("nS")
        ax2.set_ylabel("pS")

        if self.eye_jitter_pS is None:
            # ax3.hist(tie, facecolor="k", edgecolor="w", bins=50, orientation="horizontal")
            ax3.hist(
                self.tie,
                facecolor=COLOR_WFM,
                edgecolor="w",
                bins=50,
                label="tie",
            )
            ax3.set_xlabel("pS")
        else:
            ax3.hist(
                self.eye_jitter_pS,
                facecolor=COLOR_EYE,
                # edgecolor="w",
                bins=50,
                label="jitter",
            )
            ax3.set_xlabel("pS")
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)
        ax3.legend(loc="upper right")

        ax4.plot(
            self.eye_t,
            self.eye_db,
            color=COLOR_EYE,
            alpha=self.eye_alpha,
            lw=self.eye_lw,
        )
        ax4.set_xlim([-0.5, 1.5])
        ax4.set_xticks([-0.25, 0, 0.25, 0.5, 0.75, 1.00, 1.25])
        ax4.set_xlabel("T")
        ax4.set_ylabel("mV")
        ax4.set_ylim((self.ymin, self.ymax))
        ax4.set_title(
            f"{self.eye_datarate_Gbps:.3f} Gbps, {int(np.size(self.eye_db) / len(self.eye_db))} UI"
        )
        ax4.grid()

        fig_name = os.path.basename(self.wfm_file)[:-4] + "_eye_and_wfm.png"
        plt.savefig(os.path.join(self.output_path, fig_name))
        print(f"plot {fig_name}")
        print()

    def draw_eye_by_tag(self, tag="1T - 1T", fig_name=None):
        print("draw eye by transition tag")
        tag_index = np.where(self.eye_db_transition_tags == tag)[0]
        if len(tag_index) == 0:
            print(f"no transition {tag}")
            return

        loc_eye_db = self.eye_db[:, tag_index]
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
        ax.plot(
            self.eye_t,
            loc_eye_db,
            color=COLOR_EYE,
            alpha=self.eye_alpha,
            lw=self.eye_lw,
        )
        ax.set_xlim([-0.5, 1.5])
        ax.set_xticks([-0.25, 0, 0.25, 0.5, 0.75, 1.00, 1.25])
        ax.set_xlabel("T")
        ax.set_ylabel("mV")
        ax.set_ylim((self.ymin, self.ymax))
        ax.set_title(
            f"{tag} (1st crossing), {int(np.size(loc_eye_db) / len(loc_eye_db))} UI"
        )
        # ax.legend(loc="upper right")
        ax.grid()

        fig_name = os.path.basename(self.wfm_file)[:-4] + f"_{tag}_eye.png"
        plt.savefig(os.path.join(self.output_path, fig_name))
        print(f"plot {fig_name}")
        print()

    def draw_eye_wfm_and_ref_animation(self, nums=100):
        print("draw eye, wfm and ref animation")
        fig = plt.figure(figsize=(24, 10))
        grid = plt.GridSpec(2, 2, hspace=0.22, wspace=0.22)
        ax0 = fig.add_subplot(grid[0, 0])  # for wfm
        # ax1 = fig.add_subplot(grid[1, 0], sharex=ax0)  # for genie pattern
        ax1 = fig.add_subplot(grid[1, 0])  # for genie pattern
        ax2 = fig.add_subplot(grid[0:, 1])  # for eye

        lines_ax0 = [None] * 2
        lines_ax1 = [None] * 2
        (lines_ax0[0],) = ax0.plot([], [], "-", color=COLOR_LOWLIGHT, lw=2, label="wfm")
        (lines_ax0[1],) = ax0.plot([], [], "-", color=COLOR_HIGHLIGHT, lw=2, label="")
        (lines_ax1[0],) = ax1.plot(
            [], [], "-", color=COLOR_LOWLIGHT, lw=2, label="genie"
        )
        (lines_ax1[1],) = ax1.plot([], [], "-", color=COLOR_HIGHLIGHT, lw=2, label="")

        ax0.set_ylabel("mV")
        # ax0.legend()
        ax0.grid()
        ax1.set_ylim([-1.2, 1.2])
        ax1.set_yticks([-1, 0, 1])
        ax1.set_xlabel("nS")
        # ax1.legend()
        ax1.grid()
        ax2.set_xlim([-0.5, 1.5])
        ax2.set_xticks([-0.25, 0, 0.25, 0.5, 0.75, 1.00, 1.25])
        ax2.set_xlabel("T")
        ax2.set_ylabel("mV")
        ax2.set_ylim((self.ymin, self.ymax))
        ax2.grid()

        for k in range(nums):
            ext = 5

            index_wfm_active = (
                self.eye_start_i
                + k * self.eye_samples_per_T
                + np.arange(self.eye_samples_per_Window)
            )
            index_min = max(0, index_wfm_active[0] - ext * len(index_wfm_active))
            index_max = index_wfm_active[-1] + ext * len(index_wfm_active)

            wfm_all = self.wfm_v[range(index_min, index_max)]
            wfm_all_t = self.wfm_t[range(index_min, index_max)]
            wfm_active = self.wfm_v[index_wfm_active]
            wfm_active_t = self.wfm_t[index_wfm_active]

            ref_all = self.ref_v[range(index_min, index_max)]
            ref_all_t = self.ref_t[range(index_min, index_max)]
            ref_active = self.ref_v[index_wfm_active]
            ref_active_t = self.ref_t[index_wfm_active]

            lines_ax0[0].set_xdata(wfm_all_t)
            lines_ax0[0].set_ydata(wfm_all)
            lines_ax0[1].set_xdata(wfm_active_t)
            lines_ax0[1].set_ydata(wfm_active)

            lines_ax1[0].set_xdata(ref_all_t)
            lines_ax1[0].set_ydata(ref_all)
            lines_ax1[1].set_xdata(ref_active_t)
            lines_ax1[1].set_ydata(ref_active)

            ax2.plot(self.eye_t, self.eye_db[:, :k], color=COLOR_LOWLIGHT, lw=2)
            ax2.plot(self.eye_t, self.eye_db[:, k], color=COLOR_HIGHLIGHT, lw=2)
            ax2.legend([f"UI Num {k}"], loc="upper right")

            ax0.relim()
            ax0.autoscale_view(True, True, True)
            ax1.relim()
            ax1.autoscale_view(True, True, True)

            plt.draw()
            # plt.pause(1)
            # time.sleep(1)

            btnpress = plt.waitforbuttonpress(2)  # try press whitespace
            if btnpress:
                plt.waitforbuttonpress(-1)

    def draw_eye_and_wfm_animation(self, nums=100):
        print("draw eye and wfm animation")
        fig = plt.figure(figsize=(24, 10))
        grid = plt.GridSpec(1, 2, hspace=0.22, wspace=0.22)
        ax0 = fig.add_subplot(grid[0, 0])  # for wfm
        ax2 = fig.add_subplot(grid[0, 1])  # for eye

        lines_ax0 = [None] * 2
        (lines_ax0[0],) = ax0.plot([], [], "-", color=COLOR_LOWLIGHT, lw=2, label="wfm")
        (lines_ax0[1],) = ax0.plot([], [], "-", color=COLOR_HIGHLIGHT, lw=2, label="")

        ax0.set_ylabel("mV")
        # ax0.legend()
        ax0.grid()
        ax0.set_xlabel("nS")
        ax2.set_xlim([-0.5, 1.5])
        ax2.set_xticks([-0.25, 0, 0.25, 0.5, 0.75, 1.00, 1.25])
        ax2.set_xlabel("T")
        ax2.set_ylabel("mV")
        ax2.set_ylim((self.ymin, self.ymax))
        ax2.grid()

        for k in range(nums):
            ext = 5

            index_wfm_active = (
                self.eye_start_i
                + k * self.eye_samples_per_T
                + np.arange(self.eye_samples_per_Window)
            )
            index_min = max(0, index_wfm_active[0] - ext * len(index_wfm_active))
            index_max = index_wfm_active[-1] + ext * len(index_wfm_active)

            wfm_all = self.wfm_v[range(index_min, index_max)]
            wfm_all_t = self.wfm_t[range(index_min, index_max)]
            wfm_active = self.wfm_v[index_wfm_active]
            wfm_active_t = self.wfm_t[index_wfm_active]

            lines_ax0[0].set_xdata(wfm_all_t)
            lines_ax0[0].set_ydata(wfm_all)
            lines_ax0[1].set_xdata(wfm_active_t)
            lines_ax0[1].set_ydata(wfm_active)

            ax2.plot(self.eye_t, self.eye_db[:, :k], color=COLOR_LOWLIGHT, lw=2)
            ax2.plot(self.eye_t, self.eye_db[:, k], color=COLOR_HIGHLIGHT, lw=2)
            ax2.legend([f"UI Num {k}"], loc="upper right")

            ax0.relim()
            ax0.autoscale_view(True, True, True)

            plt.draw()
            # plt.pause(1)
            # time.sleep(1)

            btnpress = plt.waitforbuttonpress(2)  # try press whitespace
            if btnpress:
                plt.waitforbuttonpress(-1)

    def draw_eye_by_tag_animation(self, tag="1T - 1T"):
        print("draw eye animation by transition tag")
        tag_index = np.where(self.eye_db_transition_tags == tag)[0]
        if len(tag_index) == 0:
            print(f"no transition {tag}")
            return

        loc_eye_db = self.eye_db[:, tag_index]
        loc_eye_db_i = self.eye_db_i[:, tag_index]

        fig = plt.figure(figsize=(24, 10))
        grid = plt.GridSpec(1, 2, hspace=0.22, wspace=0.22)
        ax0 = fig.add_subplot(grid[0, 0])  # for wfm
        ax1 = fig.add_subplot(grid[0, 1])  # for eye

        lines_ax0 = [None] * 2
        (lines_ax0[0],) = ax0.plot([], [], "-", color=COLOR_LOWLIGHT, lw=2, label="wfm")
        (lines_ax0[1],) = ax0.plot([], [], "-", color=COLOR_HIGHLIGHT, lw=2, label="")

        ax0.set_ylabel("mV")
        # ax0.legend()
        ax0.grid()
        ax0.set_xlabel("nS")
        ax1.set_xlim([-0.5, 1.5])
        ax1.set_xticks([-0.25, 0, 0.25, 0.5, 0.75, 1.00, 1.25])
        ax1.set_xlabel("T")
        ax1.set_ylabel("mV")
        ax1.set_ylim((self.ymin, self.ymax))
        ax1.grid()
        ax1.set_title(f"{tag} (1st crossing)")

        for k in range(int(np.size(self.eye_db) / len(self.eye_db))):
            ext = 5

            index_wfm_active = loc_eye_db_i[:, k]
            index_min = max(0, index_wfm_active[0] - ext * len(index_wfm_active))
            index_max = index_wfm_active[-1] + ext * len(index_wfm_active)

            wfm_all = self.wfm_v[range(index_min, index_max)]
            wfm_all_t = self.wfm_t[range(index_min, index_max)]
            wfm_active = self.wfm_v[index_wfm_active]
            wfm_active_t = self.wfm_t[index_wfm_active]

            lines_ax0[0].set_xdata(wfm_all_t)
            lines_ax0[0].set_ydata(wfm_all)
            lines_ax0[1].set_xdata(wfm_active_t)
            lines_ax0[1].set_ydata(wfm_active)

            ax1.plot(self.eye_t, loc_eye_db[:, :k], color=COLOR_LOWLIGHT, lw=2)
            ax1.plot(self.eye_t, loc_eye_db[:, k], color=COLOR_HIGHLIGHT, lw=2)
            ax1.legend([f"UI Num {k}"], loc="upper right")

            ax0.relim()
            ax0.autoscale_view(True, True, True)

            plt.draw()
            # plt.pause(1)
            # time.sleep(1)

            btnpress = plt.waitforbuttonpress(2)  # try press whitespace
            if btnpress:
                plt.waitforbuttonpress(-1)

    def gen_eye_report(self):
        d = OrderedDict()
        d["Source File Location"] = self.input_path
        d["Output File Location"] = self.output_path
        d["WFM file"] = self.wfm_file
        d["REF file"] = self.ref_file
        d["F (Gbps)"] = self.F * 1e-9
        d["Fs (Gbps)"] = self.Fs * 1e-9
        d["OS"] = self.os
        d["EYE window length (UI)"] = self.eye_window_length
        d["WFM - number of cycles"] = self.num_cycles
        d["WFM - point per cycle"] = self.num_points_per_cycle
        d["EYE - start location in WFM"] = self.eye_start_i
        d["EYE - samples per UI"] = self.eye_samples_per_T
        d["EYE - samples per Window"] = self.eye_samples_per_Window
        d["EYE - DataRate (Gbps)"] = self.eye_datarate_Gbps
        d["EYE - width (nS)"] = self.eye_width_pS
        d["EYE - Zero Crossing Jitter PkPk (pS)"] = self.eye_jitter_pkpk_pS
        d["TIE - Zero Crossing Jitter PkPk (pS)"] = self.tie_jitter_pkpk_pS

        n = max([len(x) for x in d.keys()])
        print()
        print("-------------")
        print("eye properties")
        print("-------------")
        print()
        for k, v in d.items():
            print(f"{str.rjust(k, n)} : {v}")
        print


eye = EYE(
    input_path=os.path.join(os.getcwd(), "eye"),
    output_path=os.path.join(os.getcwd(), "sav"),
    wfm_file="wfm.dat",
    ref_file="prbs7.txt",
    F=3e9,
    Fs=40e9,
)

eye.parse_wfm_ref_files()
eye.gen_ref_transition_tags()
eye.find_zero_crossing()
eye.gen_tie()
eye.gen_eye_db()
eye.gen_eye_jitter(tag=None)
eye.gen_eye_jitter(tag="3T - 1T")
# eye.draw_eye_and_wfm_animation(nums=100)
# eye.draw_eye()
eye.draw_eye_and_wfm()

for tag in ["1T - 1T", "3T - 1T"]:
    eye.draw_eye_by_tag(tag=tag)

# eye.draw_eye_by_tag_animation(tag="3T - 1T")
# eye.draw_eye_wfm_and_ref_animation(nums=100)
