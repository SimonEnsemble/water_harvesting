import marimo

__generated_with = "0.15.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import numpy as np
    import random
    from scipy import interpolate
    from scipy.optimize import linprog, differential_evolution
    from scipy.stats import truncnorm
    from dataclasses import dataclass
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LinearRegression
    import seaborn as sns
    import os
    import copy
    import warnings
    import kneefinder

    # matplotlib styles
    from aquarel import load_theme
    theme = load_theme("boxy_light").set_font(size=16).set_color(palette=sns.color_palette("pastel")).set_title(size='medium').set_overrides({
        "axes.spines.right": False,
        "axes.spines.top": False
    })
    # .set_color(
    #     palette=["#458588", "#d65d0e", "#98971a", "#cc241d", "#b16286", "#d79921"]
    # )
    theme.apply()

    # date format
    my_date_format_str = '%b-%d'
    my_date_format = mdates.DateFormatter(my_date_format_str)
    return (
        IsotonicRegression,
        LinearRegression,
        copy,
        differential_evolution,
        kneefinder,
        linprog,
        mdates,
        mo,
        mpl,
        my_date_format,
        my_date_format_str,
        np,
        os,
        pd,
        plt,
        random,
        sns,
        truncnorm,
        warnings,
    )


@app.cell
def _(sns):
    sns.color_palette("pastel")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# optimizing MOF-based water harvesters""")
    return


@app.cell
def _(os):
    os.getcwd() # current working directory.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # 💧 modeling water adsorption in the MOFs

    ::icon-park:data:: experimental water adsorption data in MOFs; raw data stored in `data/`.

    | MOF | original reference | data extrd_or_n method | confirmed data fidelity | notes | Ashlee's list |
    | -- | -- | -- | -- | -- | -- |
    | MOF-801 | [link](https://doi.org/10.1038/s41467-018-03162-7) | plot digitized from SI Fig. 6a | ✅ | | ✅ |
    | KMF-1 | [link](https://www.nature.com/articles/s41467-020-18968-7) | plot digitized from Fig. 2B | ✅ |  |✅ |
    | CAU-23 | [link](https://www.nature.com/articles/s41467-019-10960-0)| plot digitized from Fig 2 | ✅ | | ✅ |
    | MIL-160 | [link](https://onlinelibrary.wiley.com/doi/10.1002/adma.201502418) | plot digitized from SI Fig. 4 |✅ | |✅ |
    | Y-shp-MOF-5 | [link](https://pubs.acs.org/doi/10.1021/jacs.7b04132) | plot digitized from Fig. 2 | ❌ | too severe hysteresis | ✅ |
    | MOF-303 | [link](https://www.science.org/doi/10.1126/science.abj0890) | plot digitized from Fig. 1 A |✅ | | ✅ |
    | CAU-10H | [link](https://pubs.rsc.org/en/content/articlelanding/2014/dt/c4dt02264e)| plot digitized from Fig. 2 | ✅ | caution: moderate hysteresis | ✅ |
    | Al-Fum | [link](https://pubs.rsc.org/en/content/articlelanding/2014/ra/c4ra03794d) | plot digitized from Fig. 3 | ✅ | |✅ |
    | MIP-200 | [link](https://www.nature.com/articles/s41560-018-0261-6) | plot digitized from Fig. 2 | ✅ ||✅ |
    | MOF-801-G | [link](https://www.science.org/doi/10.1126/sciadv.aat3198) | plot digitized from Fig. 2 | 

    we extracted all water adsorption data from plots in the papers using [plot digitizer](https://www.graphreader.com/v2). we took only the _adsorption_ branch, neglecting hysteresis.

    below, our class `MOFWaterAds` aims to:

    * read in the raw adsorption data
    * visualize the raw adsorption data
    * employ Polanyi potential theory to predict adsorption in MOFs at any temperature and pressure.
    """
    )
    return


@app.cell
def _(random, sns):
    # list of MOFs
    random.seed(97330)
    mofs = ["MOF-801", "KMF-1", "CAU-23", "MIL-160", "MOF-303", "CAU-10-H", "Al-Fum", "MIP-200"]
    random.shuffle(mofs)

    # maps MOF to the temperatures at which we possess adsorption data
    mof_to_data_temperatures = {
        "MOF-801": [15, 25, 45, 65],
        "KMF-1": [25],
        "CAU-23": [25, 40, 60],
        "MIL-160": [20, 30, 40],
        "MOF-303": [25],
        "CAU-10-H": [25, 40, 60],
        "Al-Fum": [25, 40, 60],
        "MIP-200": [30]
    }

    mof_to_color = dict(zip(mofs, sns.color_palette("husl", len(mofs))))
    mof_to_color["MOF-801G"] = "black"

    mof_to_marker = dict(zip(mofs, ['o', 's', 'd', 'X', '>', '<', 'P', 'D'])) # h next
    return mof_to_color, mof_to_data_temperatures, mof_to_marker, mofs


@app.cell
def _():
    R = 8.314 / 1000.0 # universal gas constant [kJ/(mol-K)]
    return (R,)


@app.cell
def _(os):
    fig_dir = "figs"
    os.makedirs(fig_dir, exist_ok=True) # for storing figures
    return (fig_dir,)


@app.cell
def _(mpl):
    # stuff for data viz

    # commonly-used plot labels
    axis_labels = {
        'pressure': 'relative humidity',
        'adsorption': 'water uptake\n[kg H$_2$O/kg MOF]',
        'potential': 'Polanyi adsorption potential [kJ/mol]'
    }

    # mapping temperature to color
    temperature_cmap = mpl.colormaps["inferno"]
    temperature_cmap_norm = mpl.colors.Normalize(vmin=10.0, vmax=90.0)

    def T_to_color(temperature):
        if temperature < temperature_cmap_norm.vmin or temperature > temperature_cmap_norm.vmax:
            raise Exception("out of temperature normalization range.")

        return temperature_cmap(temperature_cmap_norm(temperature))

    temperature_cmap
    return T_to_color, axis_labels


@app.cell
def _(IsotonicRegression, LinearRegression, np):
    # use isotonic regression and linear regression for extrapolation.
    def train_A_to_n_map(As, ns):
        assert (np.sort(As) == As).all()

        # isotonic regression for everything but extrapolation to larger A
        ir = IsotonicRegression(increasing=False, out_of_bounds='clip')
        ir.fit(As.reshape(len(As), 1), ns)

        # for extrapolation (large P i.e. small A)
        lr = LinearRegression()
        lr.fit(As[:2].reshape(2, 1), ns[:2])
        A_min = As[0]

        def A_to_n(A):
            A_input = np.array([A]).reshape(-1, 1)
            if A < A_min:
                return lr.predict(A_input)[0]
            else:
                return ir.predict(A_input)[0]

        return A_to_n
    return (train_A_to_n_map,)


@app.cell
def _(
    R,
    T_to_color,
    axis_labels,
    fig_dir,
    kneefinder,
    np,
    pd,
    plt,
    train_A_to_n_map,
):
    class MOFWaterAds:
        def __init__(self, mof, data_temperatures):
            """
            mof (string): name of MOF. e.g. "MOF-801"
            data_temperatures: list of data temperatures in degrees C.
            """
            print(f"CONSTRUCTING {mof}")
            self.mof = mof
            self.data_temperatures = np.sort(data_temperatures)

            # fit char. curve
            self.fit_characteristic_curves()

        def _read_ads_data(self, temperature):
            """
            read_ads_data(mof, temperature)

            temperature (integer for file-reading): degree C
            """
            # filename convention
            filename = 'data/{}_{}C.csv'.format(self.mof, temperature)

            # reads adsorption isotherm data
            ads_data = pd.read_csv(filename)

            # if the first row is zero RH, drop it.
            if ads_data['RH[%]'][0] == 0:
                ads_data = ads_data.drop(0)
                ads_data = ads_data.reset_index(drop=True)

            # convert humidity to P/P_0
            if ads_data['RH[%]'].max() > 1.0: # truly a percent
                ads_data['P/P_0'] = ads_data['RH[%]'] / 100
            else: # RH ranges from 0 to 1
                ads_data['P/P_0'] = ads_data['RH[%]']

            # Gets rid of the Humidity column, now we're using P/P_0
            ads_data = ads_data.drop(columns=['RH[%]'])

            # sort by pressure
            ads_data.sort_values(by="P/P_0", inplace=True, ignore_index=True)

            # Polanyi adsorption potential for every P/P_0 in the data set
            ads_data["A [kJ/mol]"] = -R * (temperature + 273.15) * np.log(ads_data['P/P_0'])

            return ads_data

        def find_transition_p(self, temperature, data_or_predicted):
            """
            knee detection algo to find transition pressure.
            """
            if data_or_predicted == "data":
                # read data
                data = self._read_ads_data(temperature)

                ps = data["P/P_0"]
                ws = data['Water Uptake [kg kg-1]']  
            elif data_or_predicted == "predicted":
                ps = np.linspace(0.01, 1, 100)
                ws = [self.predict_water_adsorption(temperature, p) for p in ps]
            else:
                raise Exception("data_or_predicted invalid.")

            kf = kneefinder.KneeFinder(ps, ws)
            p_star, w_star = kf.find_knee()

            return p_star, w_star

        def fit_characteristic_curves(self):
            self.ads_of_As = []
            for temperature in self.data_temperatures:
                # read in adsorption isotherm data at fit temperature
                data = self._read_ads_data(temperature)
                print(f"\tfitting Polanyi curve to T={temperature}")

                # sort rows by A values
                data.sort_values(by='A [kJ/mol]', inplace=True, ignore_index=True)

                As = data['A [kJ/mol]'].values
                ns = data['Water Uptake [kg kg-1]'].values

                # linear regressor for end
                # monotonic interpolation of water ads. as a function of Polanyi potential A.
                self.ads_of_As.append(train_A_to_n_map(As, ns))

        def predict_water_adsorption(self, temperature, p_over_p0):
            """
            use Polyanyi potential to predict water adsorption in a MOF, 
            given the temperature, P/P_0, and the characteristic curve.

            to do so, we 
            (i) calculate the Polyanyi potential
            (ii) look up the water adsorption at that potential, on the char. curve.
            """
            if p_over_p0 > 1.0:
                raise Exception("RH must fall in [0, 1]...")

            # Calculate the Polanyi potential [kJ/mol]
            A = -R * (temperature + 273.15) * np.log(p_over_p0)

            if np.isinf(A):
                return 0.0

            A_input = np.array([A]).reshape(-1, 1)

            # compute water adsorption at this A on all char. curves
            n_preds = [ads_of_A(A) for ads_of_A in self.ads_of_As] # kg/kg

            # interpolate
            return np.interp(temperature, self.data_temperatures, n_preds)

        def viz_adsorption_isotherm(self, temperature, incl_predictions=True):
            data = self._read_ads_data(temperature)

            color = T_to_color(temperature)

            plt.figure()
            plt.xlabel(axis_labels['pressure'])
            plt.ylabel(axis_labels['adsorption'])
            plt.scatter(data['P/P_0'], data['Water Uptake [kg kg-1]'], clip_on=False, color=color, label="data")
            if incl_predictions:
                p_ovr_p0s = np.linspace(0, 1, 100)[1:]
                plt.plot(
                    p_ovr_p0s, [self.predict_water_adsorption(temperature, p_ovr_p0) for p_ovr_p0 in p_ovr_p0s], 
                    color=color, label="theory"
                )

            plt.legend(title="T = {}°C".format(temperature))
            plt.ylim(ymin=0)
            plt.xlim(xmin=0)
            plt.title(self.mof)
            plt.show()

        def viz_adsorption_isotherms(self, incl_predictions=True, save=False):
            plt.figure()
            plt.title('water adsorption isotherms')
            plt.xlabel(axis_labels['pressure'])
            plt.ylabel(axis_labels['adsorption'])
            for temperature in self.data_temperatures:
                # read ads isotherm data
                data = self._read_ads_data(temperature)

                if incl_predictions:
                    p_ovr_p0s = np.linspace(0, 1, 250)
                    plt.plot(
                        p_ovr_p0s, [self.predict_water_adsorption(temperature, p_ovr_p0) for p_ovr_p0 in p_ovr_p0s], 
                        color=T_to_color(temperature)
                    )
                    plt.scatter(
                        data['P/P_0'], data['Water Uptake [kg kg-1]'], 
                        marker="s",
                        clip_on=False, color=T_to_color(temperature), label="{}°C".format(temperature)
                    )   
                else:
                     # draw data
                    plt.plot(
                        data['P/P_0'], data['Water Uptake [kg kg-1]'], 
                        marker="s",
                        clip_on=False, color=T_to_color(temperature), label="{}°C".format(temperature)
                    )      
            plt.ylim(ymin=0)
            plt.xlim(xmin=0)
            plt.title(self.mof)
            plt.legend(prop={'size': 14})
            if save:
                plt.savefig(fig_dir + f"/all_ads_isotherms_{self.mof}.pdf", format="pdf", bbox_inches='tight')
            plt.show()

        def viz_predicted_adsorption_isotherms(self, temperatures):
            plt.figure()
            plt.xlabel(axis_labels['pressure'])
            plt.ylabel(axis_labels['adsorption'])

            p_ovr_p0s = np.linspace(0, 1, 250)
            for temperature in temperatures:
                plt.plot(
                    p_ovr_p0s, [self.predict_water_adsorption(temperature, p_ovr_p0) for p_ovr_p0 in p_ovr_p0s], 
                    color=T_to_color(temperature), label="{}°C".format(temperature)
                )

            plt.ylim(ymin=0)
            plt.xlim(xmin=0)
            plt.title(self.mof)
            plt.legend(prop={'size': 14})
            plt.show()

        def plot_characteristic_curves(self, incl_model=True, save=False):
            plt.figure()
            plt.xlabel(axis_labels['potential'])
            plt.ylabel(axis_labels['adsorption'])

            A_max = -1.0 # for determining axis limits
            for i, temperature in enumerate(self.data_temperatures):
                # read ads isotherm data
                data = self._read_ads_data(temperature)

                # draw data
                plt.scatter(
                    data['A [kJ/mol]'], data['Water Uptake [kg kg-1]'], 
                    clip_on=False, color=T_to_color(temperature), label="{}°C".format(temperature)
                )

                # track A_max
                if data['A [kJ/mol]'].max() > A_max:
                    A_max = data['A [kJ/mol]'].max()

                if incl_model:
                    As = np.linspace(0, A_max)
                    plt.plot(As, [self.ads_of_As[i](Ai) for Ai in As], color=T_to_color(temperature))

            plt.ylim(ymin=0)
            plt.xlim(xmin=0)
            plt.title(self.mof)
            plt.legend(prop={'size': 14})
            if save:
                plt.savefig(fig_dir + f"/char_curve_{self.mof}.pdf", format="pdf", bbox_inches='tight')
            plt.show()
    return (MOFWaterAds,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""let's show off the capability of `MOFWaterAds` for an example MOF.""")
    return


@app.cell
def _(MOFWaterAds, mof_to_data_temperatures):
    mof = MOFWaterAds(
        # name of MOF crystal structure
        "MOF-801", 
        # list of temperatures for which we have data [°C]
        mof_to_data_temperatures["MOF-801"]
    )
    return (mof,)


@app.cell
def _(mof):
    mof._read_ads_data(
        # temperature [°C]
        45
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""use Polanyi theory to predict water adsorption at a different temperature and relative humidity.""")
    return


@app.cell
def _(mof):
    mof.predict_water_adsorption(
        # temperature [°C]
        25.0,
        # RH (frd_or_n)
        0.2
    ) # g H20 / g MOF
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""::emojione:eye:: visualize the adsorption isotherm data (dots) and Polanyi theory fit (lines).""")
    return


@app.cell
def _(mof):
    mof.viz_adsorption_isotherm(25)
    return


@app.cell
def _(mof):
    mof.find_transition_p(45, "data")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""visualize all measured adsorption data.""")
    return


@app.cell
def _(mof):
    mof.viz_adsorption_isotherms(save=True, incl_predictions=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""plot the characteristic curves (both data and theory).""")
    return


@app.cell
def _(mof):
    mof.plot_characteristic_curves(save=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""compute MAE of predicted vs actual adsorption at different temperatures.""")
    return


@app.cell
def _(mof, mof_to_data_temperatures):
    for _T in mof_to_data_temperatures["MOF-801"]:
        _p_data, _w_data = mof.find_transition_p(_T, "data")
        _p_pred, _w_data = mof.find_transition_p(_T, "predicted")
        print(f"pred vs ")
    return


@app.cell
def _(mof):
    mof.viz_predicted_adsorption_isotherms([15 + 5 * i for i in range(10)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## ::fxemoji:eye:: viz water adsorption in all of the MOFs""")
    return


@app.cell
def _(MOFWaterAds, mof_to_data_temperatures, mofs):
    mof_water_ads = {mof: MOFWaterAds(mof, mof_to_data_temperatures[mof]) for mof in mofs}
    return (mof_water_ads,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### KMF-1""")
    return


@app.cell
def _(mof_water_ads):
    mof_water_ads["KMF-1"].viz_adsorption_isotherms()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### CAU-23""")
    return


@app.cell
def _(mof_water_ads):
    mof_water_ads["CAU-23"].viz_adsorption_isotherms(save=True, incl_predictions=True)
    return


@app.cell
def _(mof_water_ads):
    mof_water_ads["CAU-23"].plot_characteristic_curves(save=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## MIL-160""")
    return


@app.cell
def _(mof_water_ads):
    mof_water_ads["MIL-160"].viz_adsorption_isotherms(incl_predictions=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## MOF-303""")
    return


@app.cell
def _(mof_water_ads):
    mof_water_ads["MOF-303"].viz_adsorption_isotherms()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## CAU-10H""")
    return


@app.cell
def _(mof_water_ads):
    mof_water_ads["CAU-10-H"].viz_adsorption_isotherms(incl_predictions=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Al-Fum""")
    return


@app.cell
def _(mof_water_ads):
    mof_water_ads["Al-Fum"].viz_adsorption_isotherms()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## MIP-200""")
    return


@app.cell
def _(mof_water_ads):
    mof_water_ads["MIP-200"].viz_adsorption_isotherms()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🪟 viz all room-temperature isotherms on one plot

    (not all data are at 25 degrees C, so we just plot predictions.)
    """
    )
    return


@app.cell
def _(axis_labels, fig_dir, mof_to_color, mof_water_ads, np, plt):
    def viz_all_predicted_adsorption_isotherms(temperature, mof_water_ads, draw_knee=False):
        fig = plt.Figure()
        plt.xlabel(axis_labels["pressure"])
        plt.ylabel(axis_labels["adsorption"])

        p_ovr_p0s = np.linspace(0, 1, 250)
        for mof in mof_water_ads.keys():
            plt.plot(
                p_ovr_p0s, [mof_water_ads[mof].predict_water_adsorption(temperature, p_ovr_p0) for p_ovr_p0 in p_ovr_p0s], 
                color=mof_to_color[mof], linewidth=3, label=mof
            )

            if draw_knee:
                p_star, w_star = mof_water_ads[mof].find_transition_p(temperature, "predicted")
                plt.scatter(p_star, w_star, marker="o", color=mof_to_color[mof])

        plt.title("T = {}°C".format(temperature))
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig(fig_dir + "/RT_ads_isotherms.pdf", format="pdf", bbox_inches="tight")
        plt.show()

    viz_all_predicted_adsorption_isotherms(25, mof_water_ads, draw_knee=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""transition pressures vs. maxima (predicted) at room temperature""")
    return


@app.cell
def _(fig_dir, mof_to_color, mof_to_marker, mof_water_ads, mofs, plt):
    def viz_step_locations(mof_water_ads, temperature):
        plt.figure()
        plt.xlabel("step location [relative humidity]")
        plt.ylabel("water uptake [kg H$_2$O/ kg MOF]\nat 100% relative humidity")
        for mof in mofs:
            # water ads at 100% RH
            w_max = mof_water_ads[mof].predict_water_adsorption(temperature, 1.0)
            # transition pressure  
            p_star, w_star = mof_water_ads[mof].find_transition_p(temperature, "predicted")
            print(f"{mof}: p* = {p_star} RH, w = {w_max}")

            plt.scatter([p_star], [w_max], color=mof_to_color[mof], label=mof, s=80, marker=mof_to_marker[mof])
        plt.xlim(0, 0.5)
        plt.ylim(0, 0.6)
        plt.title("T = {}°C".format(temperature))
        plt.legend(ncol=2)
        plt.savefig(fig_dir + "/isotherm_transitions.pdf", format="pdf", bbox_inches="tight")
        plt.show()

    viz_step_locations(mof_water_ads, 25)
    return


@app.cell
def _(mof):
    mof.data_temperatures
    return


@app.cell
def _(axis_labels, fig_dir, mof_to_color, mof_to_marker, plt):
    def viz_all_measured_adsorption_isotherms(mof_water_ads, mofs, save_tag=""):
        if save_tag == "":
            fig = plt.Figure(figsize=(6.4*0.8, 4.8*.8))
        else:
            fig = plt.Figure()
        plt.xlabel(axis_labels["pressure"])
        plt.ylabel(axis_labels["adsorption"])

        for mof in mofs:
            # choose a temperature
            if 25 in mof_water_ads[mof].data_temperatures:
                T = 25
            elif 20 in mof_water_ads[mof].data_temperatures:
                T = 20
            elif 30 in mof_water_ads[mof].data_temperatures:
                T = 30

            data = mof_water_ads[mof]._read_ads_data(T)

            # print max and transition pressure
            print(mof)
            print("\tp* = ", mof_water_ads[mof].find_transition_p(T, "data"))
            print("\tmax: ", data['Water Uptake [kg kg-1]'].max())

            plt.plot(
                data['P/P_0'], data['Water Uptake [kg kg-1]'],
                color=mof_to_color[mof], linewidth=2, label=f"{mof} [{T}°C]",
                # markerfacecolors='none', markeredgecolors=mof_to_color[mof], 
                marker=mof_to_marker[mof], #s=50
                markersize=8
            )
        if save_tag == "":
            plt.legend(prop={'size': 12})# bbox_to_anchor=(1.05, 0.5), loc='center left')
        else:
            # plt.legend(loc="lower right")
            plt.grid(False)
        plt.xlim([0, 1])
        plt.ylim([0, 0.5])
        # plt.tight_layout()
        plt.savefig(fig_dir + "/MOF_water_adsorption" + save_tag + ".pdf", format="pdf", bbox_inches='tight')
        plt.show()
    return (viz_all_measured_adsorption_isotherms,)


@app.cell
def _(mof_water_ads, mofs, viz_all_measured_adsorption_isotherms):
    viz_all_measured_adsorption_isotherms(mof_water_ads, mofs)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # 💧 water vapor pressure

    Antoine Equation Parameters from NIST [here](https://webbook.nist.gov/cgi/cbook.cgi?ID=C7732185&Mask=4), valid 293 K to 343 K i.e. 20 deg C to 70 deg C, from Gubkov, Fermor, et al., 1964.
    """
    )
    return


@app.cell
def _(np, warnings):
    # input  T  : deg C
    # output P* : bar
    def water_vapor_presssure(T):
        if T < 273 - 273.15 or T > 343 - 273.15:
            warnings.warn(f"{T}°C outside T range of Antoinne eqn.")
        A = B = C = np.nan
        # coefficients for the following setup:
        #  log10(P) = A − (B / (T + C))
        #     P = vapor pressure (bar)
        #     T = temperature (K)
        if T > 293 - 273.15:
            # valid for 293. to 343 K
            A = 6.20963
            B = 2354.731
            C = 7.559
        if T < 293 - 273.15: # cover a bit lower temperatures
            # valid for 273. to 303 K
            A = 5.40221
            B = 1838.675
            C = -31.737
        return 10.0 ** (A - B / ((T + 273.15) + C))
    return (water_vapor_presssure,)


@app.cell
def _(water_vapor_presssure):
    water_vapor_presssure(100.0) # around 1 bar b/c water boils.
    return


@app.cell
def _(fig_dir, np, plt, water_vapor_presssure):
    def viz_water_vapor_presssure():
        Ts = np.linspace(273, 343, 100) - 273.15 # deg C

        plt.figure()
        plt.xlabel("T [°C]")
        plt.ylabel("P* [bar]")
        plt.plot(Ts, [water_vapor_presssure(T_i) for T_i in Ts], linewidth=3)
        plt.savefig(fig_dir + "/water_vapor_pressure.pdf", format="pdf", bbox_inches="tight")
        plt.show()

    viz_water_vapor_presssure()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **the desorption process**. 

    1. we take air at daytime temperature $T_d$ and relative humidity $h_d\in[0, 1]$.
    the associated daytime partial pressure of water is $p_d = h_d p^* (T_d)$, where the vapor pressure of water at this daytime temperature $p^* (T_d)$ is from Antoine's equation. this air is at 1 atm.
    2. we heat up this daytime air using the sun, to a temperature $T_s>T_d$. this is a constant-pressure process. so, the percent water in the air remains the same. the fraction water is just the partial pressure divided by total pressure (ideal gas law). since the total pressure is constant, the partial pressure of water remains the same, then, at $p_d$. i.e. $p_s=p_d$. however, the hotter air can hold more water, since the saturation pressure of water vapor increases with tempearture. so we calculate the water vapor saturation pressure $p^*(T_s)$ at this new temperature and compute $h_s=p_d/p^*(T_s)$ as the relative humidity at these hotter conditions.

    putting it all together, the new relative humidity is: 

    $h_s=p_d/p^*(T_s)=h_d p^* (T_d)/p^*(T_s)$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # ⛅ weather data

    📍 Tuscon, Arizona. in 2024.

    ::rivet-icons:data::  [NOAA](https://www.ncei.noaa.gov/access/crn/qcdatasets.html) `Hourly02` data set.
    """
    )
    return


@app.cell
def _():
    time_to_color = {'day': "C1", "night": "C4"}
    time_to_color["ads"] = time_to_color["night"]
    time_to_color["des"] = time_to_color["day"]
    return (time_to_color,)


@app.cell
def _():
    city_to_state = {'Tucson': 'AZ', 'Socorro': 'NM'}
    return (city_to_state,)


@app.cell
def _(os):
    wdata_dir = "data/NOAA_weather_data"
    wfiles = os.listdir(wdata_dir)
    list(filter(lambda wfile: "CRNH" in wfile, wfiles))
    return


@app.cell
def _(
    city_to_state,
    fig_dir,
    mdates,
    my_date_format,
    my_date_format_str,
    np,
    os,
    pd,
    plt,
    time_to_color,
    water_vapor_presssure,
):
    class Weather:
        def __init__(self, month, year, location, day_min=1, day_max=33, time_to_hour={'day': 15, 'night': 5}):
            self.month = month
            self.year = year
            self.location = location

            self.duration = np.array([-2, -1, 0, 1, 2]) # for averaging T, humidity

            print(f"reading {year} {location} weather for {month}/{day_min} - {month}/{day_max}.")
            print("\tnighttime adsorption hr: ", time_to_hour["night"])
            print("\tdaytime harvest hr: ", time_to_hour["day"])
            print("\tads/des duration [hr]: ", len(self.duration))

            self.relevant_weather_cols = ["T_HR_AVG", "RH_HR_AVG", "SUR_TEMP", "SUR_RH_HR_AVG"] # latter inferred

            self.time_to_hour = time_to_hour
            self._read_raw_weather_data()
            self._filter_missing()
            self._process_datetime_and_filter(range(day_min, day_max+1))
            self._minimalize_raw_data()

            self._day_night_data()

            self._gen_ads_des_conditions()

            # for plots
            _start_date = self.ads_des_conditions["date"].min().strftime(my_date_format_str)# .date()
            _end_data = self.ads_des_conditions["date"].max().strftime(my_date_format_str)# .date()
            self.loc_title = f"{self.location}, {city_to_state[self.location]}."
            self.loc_timespan_title = f"{self.loc_title} {_start_date} to {_end_data}."

        def _read_raw_weather_data(self):
            wdata_dir = "data/NOAA_weather_data"
            wfiles = os.listdir(wdata_dir)
            assert [self.location in wfile for wfile in wfiles]

            filename = list(filter(lambda wfile: self.location in wfile and str(self.year) in wfile, wfiles))
            assert len(filename) == 1
            filename = filename[0]
            print(f"\t...reading weather data from {filename}")

            col_names = open(wdata_dir + "/headers.txt", "r").readlines()[1].split()

            raw_data = pd.read_csv(wdata_dir + "/" + filename, sep=",", names=col_names, dtype={'LST_DATE': str})

            self.raw_data = raw_data


        def _process_datetime_and_filter(self, days):
            # convert to pandas datetime
            self.raw_data["date"] = pd.to_datetime(self.raw_data["LST_DATE"])

            # keep only self.month of 2024
            self.raw_data = self.raw_data[self.raw_data["date"].dt.year == self.year] # keep only 2024
            self.raw_data = self.raw_data[self.raw_data["date"].dt.month == self.month] # keep only 2024

            # day filter
            self.raw_data = self.raw_data[[d in days for d in self.raw_data["date"].dt.day]]

            # get hours
            self.raw_data["time"] = [pd.Timedelta(hours=h) for h in self.raw_data["LST_TIME"] / 100]
            self.raw_data["datetime"] = self.raw_data["date"] + self.raw_data["time"]

            self._infer_surface_RH()

        def _infer_surface_RH(self):
            # compute new relative humidity at surface temperature, for heated air
            self.raw_data["SUR_RH_HR_AVG"] = self.raw_data.apply(
                lambda day: day["RH_HR_AVG"] * water_vapor_presssure(day["T_HR_AVG"]) / \
                        water_vapor_presssure(day["SUR_TEMP"]), axis=1
            )

        def viz_timeseries(self, save=False, incl_legend=True, legend_dx=0.0, legend_dy=0.0, toy=False):
            place_to_color = {'air': "k", 'surface': "k"}

            fig, axs = plt.subplots(2, 1, sharex=True)#, figsize=(6.4*0.8, 4.8*.8))
            if toy:
                for ax in axs:
                    ax.grid(False)
            plt.xticks(rotation=90, ha='center')
            n_days = len(self.wdata["night"]["datetime"])
            axs[1].xaxis.set_major_locator(
                mdates.AutoDateLocator(minticks=n_days-1, maxticks=n_days+1)
            )

            if not toy:
                axs[0].set_title(self.loc_title)

            # T
            if not toy:
                axs[0].plot(
                    self.raw_data["datetime"], self.raw_data["T_HR_AVG"], 
                    label="bulk air", color=place_to_color["air"], linewidth=2
                )
                axs[0].plot(
                    self.raw_data["datetime"], self.raw_data["SUR_TEMP"], 
                    label="soil surface", color=place_to_color["surface"], linewidth=2, linestyle="--"
                )
            axs[0].set_ylabel("temperature [°C]")
            axs[0].scatter(
                self.wdata["night"]["datetime"], self.wdata["night"]["T_HR_AVG"],
                edgecolors="black", clip_on=False,
                marker="^", color=time_to_color["night"], zorder=10, label="adsorption\nconditions", 
                s=200 if toy else 100 
            ) # nighttime air temperature
            axs[0].scatter(
                self.wdata["day"]["datetime"], self.wdata["day"]["SUR_TEMP"],
                edgecolors="black", clip_on=False,
                marker="v", color=time_to_color["day"], zorder=10, label="desorption\nconditions",
                s=200 if toy else 100 
            ) # daytime surface temperature
            # axs[0].set_title(self.location)
            axs[0].set_ylim(10, 65)
            axs[0].set_yticks([10 * _i for _i in range(1, 7)])
            axs[0].set_xlim(self.raw_data["datetime"].min(), self.raw_data["datetime"].max())

            # RH
            if not toy:
                axs[1].plot(
                    self.raw_data["datetime"], self.raw_data["RH_HR_AVG"] / 100, 
                    color=place_to_color["air"], label="bulk air"
                )
                axs[1].plot(
                    self.raw_data["datetime"], self.raw_data["SUR_RH_HR_AVG"] / 100, 
                    color=place_to_color["surface"], label="near-surface air", linestyle="--"
                )
            axs[1].set_ylabel("relative humidity")
            axs[1].scatter(
                self.wdata["night"]["datetime"], self.wdata["night"]["RH_HR_AVG"] / 100,
                edgecolors="black", clip_on=False,
                marker="^", color=time_to_color["night"], zorder=10, s=200 if toy else 100,  label="capture conditions"
            ) # nighttime RH
            axs[1].scatter(
                self.wdata["day"]["datetime"], self.wdata["day"]["SUR_RH_HR_AVG"] / 100,
                edgecolors="black", clip_on=False,
                marker="v", color=time_to_color["day"], zorder=10, s=200 if toy else 100, label="release conditions"
            ) # day surface RH
            axs[1].set_ylim(0, 1)
            axs[1].set_yticks([0.2 * _i for _i in range(6)])
            if self.daynight_wdata.shape[0] > 1:
                axs[1].xaxis.set_major_formatter(my_date_format)
            if incl_legend:
                axs[1].legend(
                    prop={'size': 13}, ncol=1 if toy else 2, 
                    bbox_to_anchor=(0., 1.0 + legend_dy, 1.0 + legend_dx, .1), loc="center"
                )#, loc="center left")

            # shade harvesting and desorption window
            for time in ["day", "night"]:
                for id in self.wdata[time].index:
                    now = self.wdata[time].loc[id, "datetime"]
                    start = now + pd.Timedelta(hours=self.duration.min())
                    end   = now + pd.Timedelta(hours=self.duration.max())
                    for ax in axs:
                        ax.axvspan(start, end, color=time_to_color[time], alpha=0.2)

            # already got legend above
            if save:
                plt.savefig(fig_dir + f"/weather_{self.loc_timespan_title}.pdf", format="pdf", bbox_inches="tight")

            plt.show()

        def viz_daynight_data(self):
            fig, axs = plt.subplots(2, 1, sharex=True)
            plt.xticks(rotation=90, ha='center')

            # T
            axs[0].set_ylabel("T [°C]")
            axs[0].plot(
                self.daynight_wdata["datetime"], self.daynight_wdata["night_T_HR_AVG"], 
                marker="s", label="night air", color=time_to_color["night"]
            )
            axs[0].plot(
                self.daynight_wdata["datetime"], self.daynight_wdata["day_T_HR_AVG"], 
                marker="s", label="day air", color=time_to_color["day"], linestyle="--"
            )
            axs[0].plot(
                self.daynight_wdata["datetime"], self.daynight_wdata["day_SUR_TEMP"], 
                marker="o", label="day surface", color=time_to_color["day"]
            )
            axs[0].legend(bbox_to_anchor=(1.05, 1))

            # RH
            axs[1].set_ylabel("RH [%]")

            # axs[1].set_ylim([0, 100])
            axs[1].plot(
                self.daynight_wdata["datetime"], self.daynight_wdata["night_RH_HR_AVG"], 
                marker="s", label="night air", color=time_to_color["night"]
            )
            axs[1].plot(
                self.daynight_wdata["datetime"], self.daynight_wdata["day_RH_HR_AVG"], 
                marker="s", label="day air", color=time_to_color["day"], linestyle="--"
            )
            axs[1].plot(
                self.daynight_wdata["datetime"], self.daynight_wdata["day_SUR_RH_HR_AVG"], 
                marker="s", label="day surface", color=time_to_color["day"]
            )

            axs[0].set_title(self.location)

            plt.show()

        def _minimalize_raw_data(self):
            self.raw_data = self.raw_data[["datetime"] + self.relevant_weather_cols]

        def _day_night_data(self):
            # get separate day and night data frames with precise time stamp
            # useful for checking and for plotting as a time series with all of the data
            self.wdata = dict()
            for time in ["day", "night"]:
                self.wdata[time] = self.raw_data[self.raw_data["datetime"].dt.hour == self.time_to_hour[time]]
                self.wdata[time] = self.raw_data[self.raw_data["datetime"].dt.hour == self.time_to_hour[time]]
                # over-write with average conditions
                for col in self.relevant_weather_cols:
                    for id in self.wdata[time].index:
                        # current datetime
                        dt = self.wdata[time].loc[id, "datetime"]

                        # set of relevant date times
                        relevant_dts = [dt + pd.Timedelta(hours=x) for x in self.duration]

                        # boolean mask for relevant rows to average in the raw data
                        ids_avg = [dt in relevant_dts for dt in self.raw_data["datetime"]]

                        assert np.sum(ids_avg) == len(self.duration)

                        # compute avg
                        avg = self.raw_data[ids_avg][col].mean()

                        # overwrite with avg
                        self.wdata[time].loc[id, col] = avg
                assert self.raw_data["datetime"].dt.day.nunique() == self.wdata[time].shape[0]

            ###
            #   create abstract data frame that removes details of the time
            #   each row is a day-night cycle
            ###
            reduced_wdata = dict()
            for time in ["day", "night"]:
                reduced_wdata[time] = self.wdata[time].rename(
                    columns={col: time + "_" + col for col in self.relevant_weather_cols}
                )
                reduced_wdata[time]["datetime"] = reduced_wdata[time]["datetime"].dt.normalize()

            self.daynight_wdata = pd.merge(
                reduced_wdata["night"], reduced_wdata["day"],
                on="datetime", how="outer"
            )

            self.daynight_wdata.sort_values(by="datetime", inplace=True)

            # assert sequence day by day ie none missing
            days = self.daynight_wdata.loc[1:, "datetime"].dt.day.values
            if len(days) > 1:
                days_shifted_by_one = self.daynight_wdata.loc[0:self.daynight_wdata.index[-2], "datetime"].dt.day.values
                assert np.all((days - days_shifted_by_one) == 1)

        def _gen_ads_des_conditions(self):
            self.ads_des_conditions = self.daynight_wdata.rename(columns=
                {
                    "datetime": "date",
                    # adsorptin conditions (night)
                    "night_T_HR_AVG": 'ads T [°C]',
                    "night_RH_HR_AVG": 'ads P/P0',
                    # desorption conditions (day)
                    "day_SUR_TEMP": 'des T [°C]',
                    "day_SUR_RH_HR_AVG": 'des P/P0'
                }
            )
            for rh_col in ['des P/P0', 'ads P/P0']:
                self.ads_des_conditions[rh_col] = self.ads_des_conditions[rh_col] / 100.0

            self.ads_des_conditions = self.ads_des_conditions[
                ['date', 'ads T [°C]', 'ads P/P0', 'des T [°C]', 'des P/P0']
            ]

        def _filter_missing(self):
            print("filtering # missing in raw: ", np.sum(self.raw_data["T_HR_AVG"] < -999.0))
            self.raw_data = self.raw_data[self.raw_data["T_HR_AVG"] > -999.0]
    return (Weather,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    📍 options: 

    * Tucson, AZ
    * Socorro, NM
    """
    )
    return


@app.cell
def _(Weather):
    weather = Weather(6, 2024, "Tucson", day_min=1, day_max=10)
    weather = Weather(6, 2024, "Socorro", day_min=1, day_max=10)
    # weather = Weather(8, 2024, "Tucson", day_min=11, day_max=20)
    weather.raw_data
    return (weather,)


@app.cell
def _(weather):
    weather.daynight_wdata
    return


@app.cell
def _(weather):
    weather.viz_timeseries(save=True, incl_legend=True)
    return


@app.cell
def _(weather):
    weather.viz_daynight_data()
    return


@app.cell
def _(weather):
    weather.daynight_wdata
    return


@app.cell
def _(weather):
    weather.ads_des_conditions
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# 🚿 modeling water delivery of each MOF on each day""")
    return


@app.cell
def _(warnings):
    def predict_water_delivery(weather, mof_water_ads):
        # begin with weather data
        water_del = weather.ads_des_conditions.copy()

        # predict water delivery of each MOF
        for mof in mof_water_ads.keys():
            # function giving water ads in this MOF at a given T [deg C] and RH [%]
            water_ads = lambda t, p_ovr_p0 : mof_water_ads[mof].predict_water_adsorption(t, p_ovr_p0)

            # compute water uptake at nighttime adsorption conditions
            ads_col_name = mof + " night ads [g/g]"
            water_del[ads_col_name] = water_del.apply(
                lambda day: water_ads(day["ads T [°C]"], day["ads P/P0"]), axis=1
            )

            # compute water uptake held during day at desorption conditions
            des_col_name = mof + " day ads [g/g]"
            water_del[des_col_name] = water_del.apply(
                lambda day: water_ads(day["des T [°C]"], day["des P/P0"]), axis=1
            )

            # compute water delivery
            del_col_name = mof + " water delivery [g/g]"
            water_del[del_col_name] = water_del[ads_col_name] - water_del[des_col_name]

            # handle case where water delivery is zero
            for i in range(water_del.shape[0]):
                if water_del.loc[i, del_col_name] < 0.0:
                    date_w_0 = water_del.loc[i, "date"]
                    warnings.warn(f"warning: water delivery zero for {mof} on {date_w_0}!")
            water_del.loc[water_del[del_col_name] < 0.0, del_col_name] = 0.0

        return water_del
    return (predict_water_delivery,)


@app.function
def trim_water_delivery_data(water_del, mof):
    cols = ["date", "ads T [°C]", "ads P/P0", "des T [°C]", "des P/P0"] + \
        [col for col in water_del.columns if mof in col]
    return water_del[cols]


@app.cell
def _(mof_water_ads, predict_water_delivery, weather):
    water_del = predict_water_delivery(weather, mof_water_ads)
    water_del
    return (water_del,)


@app.cell
def _(water_del):
    # lookit just water dels
    water_del[["date"] + [mof + " water delivery [g/g]" for mof in ["KMF-1", "MOF-303", "MIL-160"]]]
    # water_del[["date"] + [mof + " water delivery [g/g]" for mof in mofs]]
    return


@app.cell
def _(water_del):
    # lookit water del for a single MOF
    trim_water_delivery_data(water_del, "KMF-1")
    return


@app.cell
def _(mofs, water_del):
    # print average daily water delivery
    for _mof in mofs:
        _wd = trim_water_delivery_data(water_del, _mof)
        print("mof: ", _mof)
        print("avg del: ", _wd[_mof + " water delivery [g/g]"].mean())
    return


@app.cell
def _(
    T_to_color,
    axis_labels,
    fig_dir,
    my_date_format_str,
    np,
    plt,
    time_to_color,
):
    def viz_water_delivery(water_del, mof, day_id, mof_water_ads, weather):
        water_del_MOF = trim_water_delivery_data(water_del, mof)

        fig = plt.figure(figsize=(6.4*0.8, 4.8*.8))
        plt.xlabel(axis_labels["pressure"])
        plt.ylabel(axis_labels["adsorption"])
        plt.xlim(0.0, 1.0)

        adsdes_to_daynight = {'ads': 'night', 'des': 'day'}

        p_ovr_p0s = np.linspace(0.01, 1, 100)
        for ads_or_des in ["ads", "des"]:
            # condition
            T        = water_del_MOF.loc[day_id, ads_or_des + " T [°C]"]
            p_ovr_p0 = water_del_MOF.loc[day_id, ads_or_des + " P/P0"]

            # adsorption isotherm
            plt.plot(
                    p_ovr_p0s, [mof_water_ads[mof].predict_water_adsorption(T, p_ovr_p0) for p_ovr_p0 in p_ovr_p0s], 
                    color=T_to_color(T), linewidth=3, label=f"T = {T:.1f} °C ({adsdes_to_daynight[ads_or_des]})"
            )

            # condition
            plt.scatter(
                p_ovr_p0, mof_water_ads[mof].predict_water_adsorption(T, p_ovr_p0), 
                marker="^" if ads_or_des == "ads" else "v", s=200, zorder=100, color=time_to_color[ads_or_des], 
                label="capture conditions" if ads_or_des == "ads" else "release conditions", edgecolors="black", clip_on=False
            )
        plt.title(mof)
        date = water_del_MOF.loc[day_id, "date"].strftime(my_date_format_str)# date()
        lg = plt.legend(title=f"{date} in {weather.loc_title}", prop={'size': 12})
        lg.get_title().set_fontsize(14) 
        plt.savefig(fig_dir + f"/{mof}_ads_des_{date}_{weather.loc_title}.pdf", format="pdf", bbox_inches="tight")
        plt.show()
    return (viz_water_delivery,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""illustrate that CAU-23 picks up where MOF-303 fails.""")
    return


@app.cell
def _(mof_water_ads, viz_water_delivery, water_del, weather):
    if weather.location == "Socorro":
        viz_water_delivery(water_del, "CAU-23", 9, mof_water_ads, weather)
    return


@app.cell
def _(mof_water_ads, viz_water_delivery, water_del, weather):
    if weather.location == "Socorro":
        viz_water_delivery(water_del, "MOF-303", 8, mof_water_ads, weather)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""illustrate that CAU-23 picks up where MOF-303 and KMF-1 fail. also day 10 anomaly.""")
    return


@app.cell
def _(mof_water_ads, viz_water_delivery, water_del, weather):
    if weather.location == "Socorro":
        viz_water_delivery(water_del, "CAU-23", 9, mof_water_ads, weather)
    return


@app.cell
def _(mof_water_ads, viz_water_delivery, water_del, weather):
    if weather.location == "Socorro":
        viz_water_delivery(water_del, "MOF-303", 9, mof_water_ads, weather)
    return


@app.cell
def _(mof_water_ads, viz_water_delivery, water_del, weather):
    if weather.location == "Socorro":
        viz_water_delivery(water_del, "KMF-1", 9, mof_water_ads, weather)
    return


@app.cell
def _(mo):
    mo.md(r"""day 4 anomaly.""")
    return


@app.cell
def _(mof_water_ads, viz_water_delivery, water_del, weather):
    if weather.location == "Socorro":
        viz_water_delivery(water_del, "MIL-160", 3, mof_water_ads, weather)
    return


@app.cell
def _(mof_water_ads, viz_water_delivery, water_del, weather):
    if weather.location == "Socorro":
        viz_water_delivery(water_del, "CAU-23", 3, mof_water_ads, weather)
    return


@app.cell
def _(fig_dir, mdates, mof_to_color, mof_to_marker, my_date_format, plt):
    def viz_water_delivery_time_series(water_del, weather, mofs, savetag="", toy=False):
        n_days = len(water_del)

        if toy:
            plt.figure()
        else:
            plt.figure(figsize=(6.4 * 0.8, 3.25))
        plt.ylabel("water delivery\n[kg H$_2$O/kg MOF]")
        plt.xticks(rotation=90, ha='center')
        for mof in mofs:
            plt.plot(water_del["date"], water_del[mof + " water delivery [g/g]"], marker=mof_to_marker[mof], 
                     color=mof_to_color[mof], label=mof, clip_on=False, markersize=7
            )
        if not toy:
            plt.legend(bbox_to_anchor=(1.02, 1), prop={'size': 12})
        plt.ylim(ymin=0.0)
        if not toy:
            plt.title(weather.loc_title)
        plt.gca().xaxis.set_major_formatter(my_date_format)
        plt.gca().xaxis.set_major_locator(
                mdates.AutoDateLocator(minticks=n_days-1, maxticks=n_days+1)
            )
        if toy:
            plt.grid(False)
        plt.savefig(
            fig_dir + f"/daily_water_delivery_by_mof_{weather.loc_timespan_title}" + savetag + ".pdf", 
            format="pdf", bbox_inches="tight"
        )
        plt.show()
    return (viz_water_delivery_time_series,)


@app.cell
def _(plt, time_to_color):
    def viz_water_delivery_time_series_mof(water_del, mof):    
        plt.figure(figsize=(6.4, 3.5))
        plt.ylabel("water adsorption\n[kg H$_2$O/kg MOF]")
        plt.xticks(rotation=90, ha='center')
        for i in range(water_del.shape[0]):
            plt.vlines(
                water_del["date"], water_del[mof + " day ads [g/g]"], water_del[mof + " night ads [g/g]"], 
                     color="gray", linestyle="--"
            )
            plt.scatter(
                water_del["date"], water_del[mof + " day ads [g/g]"], color=time_to_color["day"], marker="^", 
                label="day" if i == 0 else "", edgecolor="black", s=100, clip_on=False
            )
            plt.scatter(
                water_del["date"], water_del[mof + " night ads [g/g]"], color=time_to_color["night"], marker="v",
                label="night" if i == 0 else "", edgecolor="black", s=100, clip_on=False
            )
        plt.ylim(ymin=0)
        plt.legend(loc="center")
        plt.title(mof)
        plt.show()
    return (viz_water_delivery_time_series_mof,)


@app.cell
def _(viz_water_delivery_time_series_mof, water_del):
    viz_water_delivery_time_series_mof(water_del, "KMF-1")
    return


@app.cell
def _(mofs, water_del):
    # wut MOF has the highest average water delivery?
    water_del[[mof + " water delivery [g/g]" for mof in mofs]].mean().sort_values()
    return


@app.cell
def _(mofs, water_del):
    # worst-case analysis
    water_del[[mof + " water delivery [g/g]" for mof in mofs]].min().sort_values()
    return


@app.cell
def _(mofs, water_del):
    # when constraints active
    water_del[[mof + " water delivery [g/g]" for mof in mofs]].iloc[[3, 8, 9]]
    return


@app.cell
def _(mofs, viz_water_delivery_time_series, water_del, weather):
    viz_water_delivery_time_series(water_del, weather, mofs)
    return


@app.cell
def _(viz_water_delivery_time_series, water_del, weather):
    if weather.location == "Socorro":
        viz_water_delivery_time_series(water_del, weather, ["KMF-1", "MOF-303", "MIL-160", "CAU-23"], savetag="_explain_day_10")
        # viz_water_delivery_time_series(water_del, weather, ["KMF-1", "MOF-303", "MIL-160"], savetag="_explain_no_KMF1")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # ::tabler:baseline-density-large:: baseline: pure-MOF water harvester

    how much MOF do we need for a water harvester based on a pure-MOF water harvester?
    """
    )
    return


@app.cell
def _(water_del):
    water_del
    return


@app.cell
def _(pd):
    def mass_water_harvester(mofs, water_del, daily_water_demand):
        pure_mof_harvester = pd.DataFrame(index=mofs)
        for mof in mofs:
            # get MOF needed to meet water demand on each day
            #  based on kg H2O / kg MOF delivered on that day
            #  since this is a hard constraint, we take the max over days.
            pure_mof_harvester.loc[mof, "mass [kg]"] = (daily_water_demand / water_del[mof + " water delivery [g/g]"]).max()
        return pure_mof_harvester
    return (mass_water_harvester,)


@app.cell
def _(daily_water_demand, mass_water_harvester, mofs, water_del):
    pure_mof_harvester = mass_water_harvester(mofs, water_del, daily_water_demand)
    pure_mof_harvester
    return (pure_mof_harvester,)


@app.cell
def _(fig_dir, mof_to_color, mofs, plt, pure_mof_harvester, weather):
    def viz_pure_mof_harvester(pure_mof_harvester, mofs, weather):
        fig = plt.figure(figsize=(6.4, 3.5))
        plt.bar(
            range(len(mofs)), [pure_mof_harvester.loc[mof, "mass [kg]"] for mof in mofs], 
            color=[mof_to_color[mof] for mof in mofs]
        )
        plt.xticks(range(len(mofs)), mofs, rotation=90)
        plt.ylabel("mass [kg]")
        plt.title("minimal-mass pure-MOF harvester")

        lg = plt.legend(title=f"{weather.loc_timespan_title}", prop={'size': 12})
        plt.savefig(fig_dir + f"/pure_MOF_harvester_{weather.loc_timespan_title}.pdf", format="pdf", bbox_inches="tight")
        plt.show()

    viz_pure_mof_harvester(pure_mof_harvester, mofs, weather)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # ::twemoji:control-knobs:: optimizing the water harvester

    💡 minimize the mass of the water harvester by tuning the mass of each MOF used, subject to drinking water constraints on each day.
    """
    )
    return


@app.cell
def _(linprog, np, pd, warnings):
    def optimize_harvester(
        # list of candidate MOFs to comprise the water harvester
        mofs,
        # water delivery data on each day for each MOF
        water_del, 
        # water needed per day [kg]
        daily_water_demand,
        # print stuff?
        verbose=True
    ):
        n_mofs = len(mofs)
        n_days = water_del.shape[0]

        if verbose:
            print(f"optimizing water harvest for {n_mofs} MOFs over {n_days} days...")
            print(f"\tdrinking water demand [kg] : {daily_water_demand}")

        # create W matrix
        #  w[d, m] = water delivery [g/g] on day d by MOF m
        water_del_cols = [mof + " water delivery [g/g]" for mof in mofs]
        W = water_del.loc[:, water_del_cols]

        # create cost vector
        cost_per_mass = np.ones(len(mofs))

        # solve linear program
        # decision variable: mass_of_mofs
        res = linprog(
            # objective to minimize = cost_per_mass * mass_of_mofs
            cost_per_mass,
            # daily drinking water constraint W * mass_of_mofs >= daily_demand 
            #    (negative to flip inequality)
            A_ub=-W, b_ub=[-daily_water_demand for d in range(n_days)],
            #
            bounds=(0, None),
            # solver
            method='highs'
        )

        if not res.success:
            warnings.warn("yikes! failure to solve linear program.")
            return res
        else:
            if verbose:
                print("\toptimization successful.")
                print("\t\tmin mass of water harvester [kg]: ", res.fun)
                print("\t\toptimal composition:")
                for (m, mof) in enumerate(mofs):
                    print(f"\t\t\t{mof}: {res.x[m]} kg")

        opt_info = {
            'active constraints': [d for d, s in enumerate(res.slack) if s == 0],
            'slack': res.slack,
            'marginals': res.ineqlin["marginals"]
        }

        # slack and residuals are the same.
        assert np.all(res.ineqlin["residual"] == res.slack)

        return pd.DataFrame({"mass [kg]": res.x}, index=mofs), res.fun, opt_info
    return (optimize_harvester,)


@app.cell
def _(mofs, optimize_harvester, water_del):
    daily_water_demand = 2.0 # kg
    opt_mass_of_mofs, min_mass, opt_info = optimize_harvester(mofs, water_del, daily_water_demand)
    return daily_water_demand, opt_info, opt_mass_of_mofs


@app.cell
def _(
    fig_dir,
    mof_to_color,
    mofs,
    np,
    opt_mass_of_mofs,
    plt,
    pure_mof_harvester,
    weather,
):
    def viz_optimal_harvester(mofs, opt_mass_of_mofs, pure_mof_harvester, weather, save_tag="", ymax_override=None):
        fig = plt.figure(figsize=(6.4 *0.8, 3.6*.8))
        plt.ylabel("mass [kg MOF]")

        # mass of each MOF
        plt.bar(
            range(len(mofs)), 
            [opt_mass_of_mofs.loc[mof, "mass [kg]"] for mof in mofs], 
            color=[mof_to_color[mof] for mof in mofs], edgecolor="black"
        )

        # baseline of optimal pure-MOF water harvester
        if pure_mof_harvester is not None:
            opt_pure_mof = pure_mof_harvester["mass [kg]"].idxmin()
            print("opt pure MOF: ", opt_pure_mof)

            plt.axhline(
                pure_mof_harvester["mass [kg]"].min(), color="gray", linestyle="--"
            )

            if not ((weather.month == 6) and (weather.location == "Tucson")) and not ("MOF-801G" in pure_mof_harvester.index):
                plt.text(-0.5, pure_mof_harvester["mass [kg]"].min(), 
                         f"optimal single-MOF bed ({opt_pure_mof})", fontsize=12,
                            verticalalignment='center', horizontalalignment="left", 
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.75, edgecolor='none')
                )


        # total mass of device, broken down
        x_pos = len(mofs) + 1.
        bottom = 0
        for mof in opt_mass_of_mofs.sort_values("mass [kg]", ascending=False).index:
            m_mof = opt_mass_of_mofs.loc[mof, "mass [kg]"]
            if m_mof > 0.0:
                plt.bar(x_pos, m_mof, bottom=bottom, color=mof_to_color[mof], edgecolor="k")
                plt.text(x_pos + 0.5, bottom + m_mof / 2, mof, verticalalignment="center", fontsize=10)

                bottom += m_mof

        #  handle x-ticks
        plt.xticks([m for m in range(len(mofs))] + [x_pos], mofs + ["total"], rotation=90)
        plt.gca().get_xticklabels()[-1].set_fontweight('bold')

        # print total mass as legend
        total_mass = np.round(opt_mass_of_mofs["mass [kg]"].sum(), decimals=2)
        lg = plt.legend(title=f"total mass:\n  {total_mass} kg")
        lg.get_title().set_fontsize(14) 
        # x, y limits
        plt.xlim(-0.75, x_pos+2.6)
        if pure_mof_harvester is not None:
            plt.ylim(0, pure_mof_harvester["mass [kg]"].min() * 1.1)

        # highlight second plot
        plt.fill_betweenx(
            [0, 3], len(mofs), x_pos+2.6, alpha=0.05, zorder=0, color="black",
            transform=plt.gca().get_xaxis_transform()
        )

        plt.title(weather.loc_timespan_title)

        if weather.location == "Tucson":
            plt.ylim(0, 7)
        if weather.location == "Socorro":
            plt.ylim(0, 25)
        if ymax_override:
            plt.ylim(0, ymax_override)

        # save
        plt.savefig(
            fig_dir + f"/opt_composition_{weather.loc_timespan_title}" + save_tag + ".pdf", 
            format="pdf", bbox_inches="tight"
        )
        plt.show()

    viz_optimal_harvester(mofs, opt_mass_of_mofs, pure_mof_harvester, weather)
    return (viz_optimal_harvester,)


@app.cell
def _(opt_mass_of_mofs):
    # composition by mass fraction
    opt_mass_of_mofs / opt_mass_of_mofs["mass [kg]"].sum()
    return


@app.cell
def _(opt_mass_of_mofs, pure_mof_harvester):
    print("ratio of mix-to-pure mass harvester: ", opt_mass_of_mofs["mass [kg]"].sum() / pure_mof_harvester["mass [kg]"].min())
    return


@app.cell
def _():
    def get_active_mofs(opt_mass_of_mofs):
        return opt_mass_of_mofs[opt_mass_of_mofs["mass [kg]"] > 0.0].index.values

    def get_nonactive_mofs(opt_mass_of_mofs):
        return opt_mass_of_mofs[opt_mass_of_mofs["mass [kg]"] == 0.0].index.values
    return get_active_mofs, get_nonactive_mofs


@app.cell
def _(fig_dir, get_active_mofs, mof_to_color, opt_mass_of_mofs, plt, weather):
    def viz_optimal_harvester_pie(opt_mass_of_mofs, weather, save_fig=True):
        active_mofs = get_active_mofs(opt_mass_of_mofs)
        ms = [opt_mass_of_mofs.loc[mof, "mass [kg]"] for mof in active_mofs]

        total_mass = opt_mass_of_mofs["mass [kg]"].sum()

        fig, ax = plt.subplots()
        ax.pie(
            ms, labels=active_mofs, 
            colors=[mof_to_color[mof] for mof in active_mofs]
        )
        plt.title(f"total mass: {total_mass:.1f} kg")
        if save_fig:
            plt.savefig(
                fig_dir + f"/opt_composition_pie{weather.loc_timespan_title}.pdf", 
                format="pdf", bbox_inches="tight"
            )
        plt.show()

    viz_optimal_harvester_pie(opt_mass_of_mofs, weather)
    return (viz_optimal_harvester_pie,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""viz the slack (water delivered on each day).""")
    return


@app.cell
def _(np, opt_info, weather):
    # the slack is W m* - d i.e the extra drinking water we got.
    assert len(opt_info["slack"]) == weather.ads_des_conditions.shape[0]

    assert np.all([opt_info["slack"][c] == 0.0 for c in opt_info["active constraints"]])

    opt_info["slack"]
    return


@app.cell
def _(fig_dir, mof_to_color, my_date_format, plt):
    def viz_opt_water_delivery_time_series(water_del, opt_mass_of_mofs, daily_water_demand, weather):
        # list of MOFs comprising water harvester
        mofs = [mof for mof in opt_mass_of_mofs.sort_values("mass [kg]", ascending=False).index 
                if opt_mass_of_mofs.loc[mof, "mass [kg]"] > 0
        ]

        plt.figure(figsize=(6.4 * 0.8, 3.6 * 0.8))
        plt.ylabel("water delivery [kg H$_2$O]")
        plt.xticks(rotation=90, ha='center')
        bottom = [0 for d in water_del["date"]]
        for mof in mofs:
            w_mof = water_del[mof + " water delivery [g/g]"] * opt_mass_of_mofs.loc[mof, "mass [kg]"]
            plt.bar(water_del["date"], w_mof, 
                    color=mof_to_color[mof], label=mof, bottom=bottom, edgecolor="k"
            )
            bottom = bottom + w_mof
        plt.axhline(y=daily_water_demand, color="black", linestyle="--", label="demand")
        plt.legend(
            prop={'size': 10}, ncol=1, 
            loc="upper left" if weather.location == "Socorro" else "center"
        )# bbox_to_anchor=(1.05, 1), )
        plt.title(weather.loc_title)
        plt.gca().xaxis.set_major_formatter(my_date_format)
        if weather.location == "Socorro":
            plt.ylim(0, 5.0)
        plt.savefig(
            fig_dir + f"/opt_water_delivery_{weather.loc_timespan_title}.pdf", 
            format="pdf", bbox_inches="tight"
        )
        plt.show()
    return (viz_opt_water_delivery_time_series,)


@app.cell
def _(
    daily_water_demand,
    opt_mass_of_mofs,
    viz_opt_water_delivery_time_series,
    water_del,
    weather,
):
    viz_opt_water_delivery_time_series(water_del, opt_mass_of_mofs, daily_water_demand, weather)
    return


@app.cell
def _(opt_info):
    opt_info["slack"]
    return


@app.cell
def _(water_del):
    water_del
    return


@app.cell
def _():
    return


@app.cell
def _(mofs, opt_mass_of_mofs, water_del):
    def build_water_del_opt_harvester_data(water_del, opt_mass_of_mofs):
        water_del_opt_harvester = water_del[["date"]].copy()
        for mof in mofs:
            m_opt_mof = opt_mass_of_mofs.loc[mof, "mass [kg]"]
            if m_opt_mof == 0.0:
                continue
            col = mof + " water delivery [g]"
            water_del_opt_harvester[col] = water_del[mof + " water delivery [g/g]"] * m_opt_mof

        water_del_opt_harvester["total water delivery [g]"] = water_del_opt_harvester.drop("date", axis=1).sum(axis=1)

        for mof in mofs:
            if opt_mass_of_mofs.loc[mof, "mass [kg]"] == 0.0:
                continue
            col = "fraction delivered by " + mof
            water_del_opt_harvester[col] = water_del_opt_harvester[mof + " water delivery [g]"] / water_del_opt_harvester["total water delivery [g]"]

        return water_del_opt_harvester

    water_del_opt_harvester = build_water_del_opt_harvester_data(water_del, opt_mass_of_mofs)
    water_del_opt_harvester
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""the marginals reveal how sensitive the solution is, to the daily drinking water demand changing. these refer to only the active constraints.""")
    return


@app.cell
def _(opt_info):
    opt_info["active constraints"]
    return


@app.cell
def _(opt_info):
    opt_info["marginals"]
    return


@app.cell
def _(fig_dir, my_date_format, opt_info, plt, weather):
    def viz_marginals(opt_info, weather):
        plt.figure(figsize=(6.4 * 0.8, 3.6))
        plt.bar(weather.ads_des_conditions["date"], -opt_info["marginals"])
        plt.xticks(rotation=90)
        plt.ylabel("shadow price\n[kg MOF / kg H$_2$O]")
        plt.gca().xaxis.set_major_formatter(my_date_format)
        plt.title(f"{weather.loc_timespan_title}")
        plt.tight_layout()
        plt.savefig(fig_dir + f"/shadow_prices_{weather.loc_timespan_title}.pdf", format="pdf")
        plt.show()

    viz_marginals(opt_info, weather)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### sensitivity analysis

    what happens if we perturb the predicted water delivery?
    """
    )
    return


@app.cell
def _(optimize_harvester):
    def design_under_perturbed_water_del(mofs, water_del, daily_water_demand, mof, x):
        # perturb water delivery of MOF x by 10%
        perturbed_water_del = water_del.copy()
        perturbed_water_del[mof + " water delivery [g/g]"] *= (1.0 + x)

        opt_mass_of_mofs, min_mass, opt_info = optimize_harvester(
            mofs, perturbed_water_del, daily_water_demand, verbose=False
        )

        return opt_mass_of_mofs, min_mass, opt_info
    return (design_under_perturbed_water_del,)


@app.cell
def _(design_under_perturbed_water_del, get_active_mofs, get_nonactive_mofs):
    def sensitivity_analysis(opt_mass_of_mofs, water_del, daily_water_demand, x=0.1):
        old_opt_mass_of_mofs = opt_mass_of_mofs["mass [kg]"].sum()
        print("old mass = ", old_opt_mass_of_mofs)
        active_mofs = get_active_mofs(opt_mass_of_mofs)
        nonactive_mofs = get_nonactive_mofs(opt_mass_of_mofs)

        # increase water delivery of non-active MOFs by 10%. 
        #   does the set of active MOFs change?
        for mof in nonactive_mofs:
            new_opt_mass_of_mofs, _, _ = design_under_perturbed_water_del(
                opt_mass_of_mofs.index, water_del, daily_water_demand, mof, x
            )
            new_active_mofs = get_active_mofs(new_opt_mass_of_mofs)
            new_mass = new_opt_mass_of_mofs["mass [kg]"].sum()

            if set(new_active_mofs) != set(active_mofs):
                print(f"increasing water delivery of {mof} by {x} changes composition to {new_active_mofs}.")
                print("\tmass % change = ", (new_mass - old_opt_mass_of_mofs) / old_opt_mass_of_mofs)

        # decrease water delivery of active MOFs by 10%. 
        #   does the set of active MOFs change?
        for mof in active_mofs:
            new_opt_mass_of_mofs, _, _ = design_under_perturbed_water_del(
                opt_mass_of_mofs.index, water_del, daily_water_demand, mof, -x
            )
            new_active_mofs = get_active_mofs(new_opt_mass_of_mofs)
            new_mass = new_opt_mass_of_mofs["mass [kg]"].sum()

            if set(new_active_mofs) != set(active_mofs):
                print(f"decreasing water delivery of {mof} by {x} changes composition to {new_active_mofs}.")
                print("\tmass % change = ", (new_mass - old_opt_mass_of_mofs) / old_opt_mass_of_mofs)
    return (sensitivity_analysis,)


@app.cell
def _(daily_water_demand, opt_mass_of_mofs, sensitivity_analysis, water_del):
    sensitivity_analysis(opt_mass_of_mofs, water_del, daily_water_demand, x=0.1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""how about perturbing the weather?""")
    return


@app.cell(hide_code=True)
def _(mo):
    checkbox = mo.ui.checkbox(label="run sensitivity")
    checkbox
    return (checkbox,)


@app.cell
def _(
    checkbox,
    copy,
    daily_water_demand,
    mof_water_ads,
    mofs,
    optimize_harvester,
    predict_water_delivery,
    truncnorm,
    viz_optimal_harvester,
    weather,
):
    def design_under_perturbed_weather(weather, mof_water_ads, daily_water_demand, sigma):
        ###
        # modify weather
        ###
        new_weather = copy.deepcopy(weather)
        n_days = new_weather.ads_des_conditions.shape[0]

        # lower- and upper-bounds on variables
        lb = {"T [°C]": -10.0, "P/P0": 0.0} # for truncation
        ub = {"T [°C]": 100.0, "P/P0": 1.0}

        for ads_des in ["ads ", "des "]:
            for var in ["T [°C]", "P/P0"]:
                # truncated normal distribution
                new_weather.ads_des_conditions[ads_des + var] = truncnorm.rvs(
                    lb[var], ub[var], scale=sigma[var], size=n_days, loc=new_weather.ads_des_conditions[ads_des + var]
                )

        ###
        # predict water deliveries under modified weather
        ###
        new_water_del = predict_water_delivery(new_weather, {mof: mof_water_ads[mof] for mof in mofs})

        ###
        # design adsorbent bed
        ###
        opt_mass_of_mofs, min_mass, opt_info = optimize_harvester(mofs, new_water_del, daily_water_demand, verbose=False)

        return opt_mass_of_mofs

    sigma = {"T [°C]": 2.0, "P/P0": 0.02}
    n_weather_designs = 12

    if checkbox.value:
        perturbed_weather_designs = [
            design_under_perturbed_weather(weather, mof_water_ads, daily_water_demand, sigma) for _d in range(n_weather_designs)
        ]

        for _d in range(n_weather_designs):
            viz_optimal_harvester(mofs, perturbed_weather_designs[_d], None, weather, save_tag=f"modified_weather_{_d}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # 2D toy example for visualizing the constraints

    suppose there are two candidate MOFs to be carried over three days.
    """
    )
    return


@app.cell
def _(
    Weather,
    daily_water_demand,
    fig_dir,
    mass_water_harvester,
    mof_water_ads,
    np,
    optimize_harvester,
    plt,
    predict_water_delivery,
):
    def draw_2D_toy_example(toy_mode):
        weather = Weather(6, 2024, "Socorro", day_min=8, day_max=10)

        mofs = ["MOF-303", "CAU-23"]

        water_del = predict_water_delivery(weather, {mof: mof_water_ads[mof] for mof in mofs})

        opt_mass_of_mofs, min_mass, opt_info = optimize_harvester(mofs, water_del, daily_water_demand)

        pure_mof_harvester = mass_water_harvester(mofs, water_del, daily_water_demand)

        _start_date = weather.ads_des_conditions["date"].min().date()
        _end_data = weather.ads_des_conditions["date"].max().date()

        plt.figure(figsize=(6.4 *0.8, 4.8*.8))
        plt.xlabel(f"mass of {mofs[0]} [kg]")
        plt.ylabel(f"mass of {mofs[1]} [kg]")

        # plot optimal composition
        if not toy_mode:
            plt.scatter(
                opt_mass_of_mofs.loc[mofs[0], "mass [kg]"], opt_mass_of_mofs.loc[mofs[1], "mass [kg]"], 
                s=250, marker="*", clip_on=False, zorder=25, color="C1", edgecolor="black", label="optimal\ncomposition"
            )

        max_mass = 30.0 if toy_mode else 50.0 # _opt_mass_of_mofs["mass [kg]"].max() * 2.5

        plt.xlim(0, max_mass)
        plt.ylim(0, max_mass)

        # plot water delivery constraints
        m0s = np.linspace(0.0, max_mass, 500)
        m1s_feasible = np.zeros(len(m0s))
        for d in range(water_del.shape[0]):
            d_0 = water_del.loc[d, f"{mofs[0]} water delivery [g/g]"]
            d_1 = water_del.loc[d, f"{mofs[1]} water delivery [g/g]"]

            m1s = (daily_water_demand - m0s * d_0) / d_1
            if d in opt_info["active constraints"]:
                m1s_feasible = np.maximum(m1s_feasible, m1s)

            plt.plot(
                m0s, m1s, 
                color="black", label="drinking water\ndelivery constraint" if d == 0 else ""
            )

        # plot constant mass
        if not toy_mode:
            plt.plot(m0s, min_mass - m0s, color="C5", linestyle="--", label=f"mass = {min_mass:.2f} kg")

        # shade feasible region (works for two active constraints)
        ids_feasible = m1s_feasible < max_mass
        plt.fill_between(
            m0s[ids_feasible], m1s_feasible[ids_feasible], 
            np.ones(np.sum(ids_feasible)) * max_mass,
            color="C2", label="feasible region"
        )

        # pure-MOF harvester
        if not toy_mode:
            plt.scatter(
                pure_mof_harvester.loc[mofs[0], "mass [kg]"], 0.0, 
                label="optimal\nsingle-MOF bed", color="C0", clip_on=False, s=100, zorder=100, edgecolor="black"
            )
        plt.scatter(
            0.0, pure_mof_harvester.loc[mofs[1], "mass [kg]"], 
            color="C0", clip_on=False, s=100, zorder=100, edgecolor="black"
        )

        plt.gca().set_aspect('equal', 'box')
        plt.xticks([0, 10, 20, 30, 40, 50])
        plt.yticks([0, 10, 20, 30, 40, 50])
        plt.xlim([0, max_mass])
        plt.ylim([0, max_mass])

        # split into two lines
        if not toy_mode:
            _lg_title = weather.loc_title + '\n' + weather.loc_timespan_title.split(".")[1] + '.'
            _lg = plt.legend(title=_lg_title, bbox_to_anchor=(1.05, 0.5), loc='center left', prop={'size':14})
            _lg.get_title().set_fontsize(14) 

        _savetag = "_toy" if toy_mode else "" 
        plt.savefig(
            fig_dir + f"/twoD_linear_program_{weather.loc_timespan_title}" + _savetag + ".pdf", 
            format="pdf", bbox_inches="tight"
        )
        plt.show()

        water_del.loc[opt_info["active constraints"]]
        pure_mof_harvester

    draw_2D_toy_example(False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # comparison with MOF-801 harvester in the field

    on 22 October 2017 in Scottsdale, AZ, USA. 

    [publication link](https://www.science.org/doi/10.1126/sciadv.aat3198).

    > Using 0.825 kg of MOF-801/G, 55 g of water was collected
    """
    )
    return


@app.cell
def _():
    field_mass = 0.825 # kg
    return (field_mass,)


@app.cell
def _():
    field_daily_water_demand = 55 / 1000 # kg water, reality
    # field_daily_water_demand = 78 / 1000 # kg water, lab
    return (field_daily_water_demand,)


@app.cell
def _(Weather):
    field_weather = Weather(10, 2017, "Tucson", day_min=22, day_max=22, time_to_hour={'day': 14, 'night': 5})
    field_weather.daynight_wdata
    return (field_weather,)


@app.cell
def _(field_weather):
    field_weather.viz_timeseries(save=True, incl_legend=True, legend_dy=-0.15)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    eyeballing Fig. 3B in the paper and it says:

    > 5% RH at 35° to 40°C during the day
    > 40% RH at 10° to 15°C during the night

    reasonably matches below except Tucson during the day appears a bit cooler.

    they are able to heat the MOF to almost 90 degrees!
    """
    )
    return


@app.cell
def _(MOFWaterAds):
    field_mof_water_ads = {"MOF-801G": MOFWaterAds("MOF-801G", [25])}
    return (field_mof_water_ads,)


@app.cell
def _(field_mof_water_ads):
    field_mof_water_ads["MOF-801G"].viz_adsorption_isotherms(save=True)
    return


@app.cell
def _(field_mof_water_ads, field_weather, predict_water_delivery):
    field_water_del = predict_water_delivery(
        field_weather, field_mof_water_ads
    )
    field_water_del
    return (field_water_del,)


@app.cell
def _(field_mass, field_water_del):
    # predicted water delivered [g]
    field_mass * field_water_del["MOF-801G water delivery [g/g]"].values * 1000
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""get MOF-801/G adsorption data""")
    return


@app.cell
def _(field_daily_water_demand, field_water_del, mass_water_harvester):
    mass_water_harvester(["MOF-801G"], field_water_del, field_daily_water_demand)
    return


@app.cell
def _(field_daily_water_demand, field_water_del, optimize_harvester):
    field_opt_mass_mofs = optimize_harvester(["MOF-801G"], field_water_del, field_daily_water_demand)
    return (field_opt_mass_mofs,)


@app.cell
def _(field_daily_water_demand, field_water_del, mass_water_harvester):
    field_pure_mof_harvester = mass_water_harvester(["MOF-801G"], field_water_del, field_daily_water_demand)
    field_pure_mof_harvester
    return (field_pure_mof_harvester,)


@app.cell
def _(field_mof_water_ads, field_water_del, field_weather, viz_water_delivery):
    viz_water_delivery(field_water_del, "MOF-801G", 0, field_mof_water_ads, field_weather)
    return


@app.cell
def _(field_daily_water_demand):
    # best case adsorb at 100% RH and desorb ALL
    _water_del_max = 0.15 # g H20 / g MOF
    field_daily_water_demand / _water_del_max
    # so there's a huge loss of efficiency for their water harvester in real life...
    return


@app.cell
def _(
    field_opt_mass_mofs,
    field_pure_mof_harvester,
    field_weather,
    viz_optimal_harvester,
):
    viz_optimal_harvester(["MOF-801G"], field_opt_mass_mofs[0], field_pure_mof_harvester, field_weather, ymax_override=0.5)
    return


@app.cell
def _(field_mass, field_opt_mass_mofs):
    field_mass / field_opt_mass_mofs[0].values
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# 🪡 fitting alternative water adsorption models""")
    return


@app.cell(hide_code=True)
def _(mo):
    checkbox_alt_models = mo.ui.checkbox(label="fit alt models")
    checkbox_alt_models
    return (checkbox_alt_models,)


@app.cell
def _(mof_water_ads, water_vapor_presssure):
    def assemble_all_ads_data(mof_water_ads, mof):
        T_list = []
        P_list = []
        n_list = []
        A_list = []
        for temperature in mof_water_ads[mof].data_temperatures:
            # read ads isotherm data
            data = mof_water_ads[mof]._read_ads_data(temperature)

            P0 = water_vapor_presssure(temperature)   
            T_list += [temperature + 273.15] * data.shape[0]          
            P_list += list(data['P/P_0'] * P0)
            n_list += list(data['Water Uptake [kg kg-1]'])
            A_list += list(data['A [kJ/mol]'])
        return T_list, P_list, n_list, A_list

    T_list, P_list, n_list, A_list = assemble_all_ads_data(mof_water_ads, "MOF-801")
    return A_list, P_list, T_list, n_list


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## parametric Polanyi model""")
    return


@app.cell
def _(np):
    def polanyi_model(params, A):
        m, E, b = params
        return m * np.exp(-(A / E) ** b)
    return (polanyi_model,)


@app.cell
def _(A_list, n_list, np, polanyi_model):
    def polanyi_objective(params):
        n_pred = np.array([polanyi_model(params, A) for A in A_list])
        return np.sum((n_list - n_pred)**2)
    return (polanyi_objective,)


@app.cell
def _(differential_evolution):
    def fit_polanyi(objective, seed=100):
        bounds = [
            (0, 1),       # m [kg/kg]
            (0.05, 100),  # E (kJ/mol)
            (0.05, 10)    # b
        ]     
        result = differential_evolution(
            objective, bounds, maxiter=10000, tol=1e-4, atol=1e-6, seed=seed, popsize=100
        )
        return result.x, result.fun
    return (fit_polanyi,)


@app.cell
def _(T_to_color, axis_labels, np, plt, polanyi_model):
    def viz_parametric_polanyi_fit(mof_water_ads, mof, params, RSS):
        plt.figure()
        plt.xlabel(axis_labels['potential'])
        plt.ylabel(axis_labels['adsorption'])

        m, E, b = params
        A_max = -1.0 # for determining axis limits
        for temperature in mof_water_ads[mof].data_temperatures:
            # read ads isotherm data
            data = mof_water_ads[mof]._read_ads_data(temperature)

            # draw data
            plt.scatter(
                data['A [kJ/mol]'], data['Water Uptake [kg kg-1]'], 
                clip_on=False, color=T_to_color(temperature), label="{}°C (exp't)".format(temperature)
            )

            # track A_max
            if data['A [kJ/mol]'].max() > A_max:
                A_max = data['A [kJ/mol]'].max()

        As = np.linspace(0, A_max)
        plt.plot(
            As, [polanyi_model(params, A) for A in As], 
            color="green", label="fit DA model",
            linewidth=3
        )

        print("fit params:", params)
        print("RSS:", RSS)
        plt.ylim(ymin=0)
        plt.xlim(xmin=0)
        plt.title(mof)
        plt.legend(prop={'size': 14})
        plt.tight_layout()
        plt.savefig(f"da_fit_{mof}.pdf", format="pdf")
        plt.show()
    return (viz_parametric_polanyi_fit,)


@app.cell
def _(
    checkbox_alt_models,
    fit_polanyi,
    mof_water_ads,
    polanyi_objective,
    viz_parametric_polanyi_fit,
):
    if checkbox_alt_models.value:
        polanyi_params, polanyi_rss = fit_polanyi(polanyi_objective)
        viz_parametric_polanyi_fit(mof_water_ads, "MOF-801", polanyi_params, polanyi_rss)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## ❌ dual site Langmuir model""")
    return


@app.cell
def _(np):
    def lf(temperature, pressure, m, k, a, v):
        K = k * np.exp(a / temperature)
        return (m * K * pressure**v) / (1 + K * pressure**v)
    return (lf,)


@app.cell
def _(lf):
    def dsfl(params, temperature_pressure):     
        temperature, pressure = temperature_pressure
        m1, k1, a1, v1, m2, k2, a2, v2 = params
        return lf(temperature, pressure, m1, k1, a1, v1) + lf(temperature, pressure, m2, k2, a2, v2)
    return (dsfl,)


@app.cell
def _(P_list, T_list, dsfl, n_list, np):
    def dsfl_objective(params):
        n_pred = np.array([dsfl(params, (temperature, pressure)) 
                           for temperature, pressure in zip(T_list, P_list)])
        return np.sum((n_list - n_pred)**2)
    return (dsfl_objective,)


@app.cell
def _(differential_evolution):
    def fit_dsfl(objective, seed=100):
        bounds = [
            (0,   1),     # m1
            (0, 0.1),     # k1
            (0, 8e4),     # a1  
            (0,   2),     # v1 
            (0,   1),     # m2
            (0, 0.1),     # k2
            (0, 8e4),     # a2
            (0,   5)      # v2
        ]
        result = differential_evolution(objective, bounds, maxiter=10000, tol=1e-4, atol=1e-6, seed=seed, popsize=100)
        return result.x, result.fun
    return (fit_dsfl,)


@app.cell
def _(T_to_color, axis_labels, dsfl, np, plt, water_vapor_presssure):
    def viz_dslf_fit(mof_water_ads, mof, params, RSS):
        plt.figure()
        plt.title('water adsorption isotherms')
        plt.xlabel(axis_labels['pressure'])
        plt.ylabel(axis_labels['adsorption'])

        m1, k1, a1, v1, m2, k2, a2, v2 = params

        for temperature in mof_water_ads[mof].data_temperatures:
            P0 = water_vapor_presssure(temperature) 
            # read ads isotherm data
            data = mof_water_ads[mof]._read_ads_data(temperature)

            # draw data
            plt.scatter(
                np.array(data['P/P_0']), data['Water Uptake [kg kg-1]'], 
                marker="s",
                clip_on=False, color=T_to_color(temperature), label="{}°C (exp't)".format(temperature)
            )        

            p_ovr_p0s = np.linspace(0, 1, 100)[1:]
            plt.plot(
                    p_ovr_p0s, [dsfl(params, (temperature+273.15, p_ovr_p0*P0)) 
                                for p_ovr_p0 in p_ovr_p0s], 
                    color=T_to_color(temperature), label=str(temperature)+"°C DSLF fit"
            )
        print("fit params:", params)
        print("RSS:", RSS)
        plt.ylim(ymin=0)
        plt.xlim(xmin=0)
        plt.title(mof)
        plt.legend(prop={'size': 14})
        plt.tight_layout()
        plt.savefig(f"dslf_fit_{mof}.pdf", format="pdf")
        plt.show()
    return (viz_dslf_fit,)


@app.cell
def _(
    checkbox_alt_models,
    dsfl_objective,
    fit_dsfl,
    mof_water_ads,
    viz_dslf_fit,
):
    if checkbox_alt_models.value:
        dsfl_params, dsfl_rss = fit_dsfl(dsfl_objective)
        viz_dslf_fit(mof_water_ads, "MOF-801", dsfl_params, dsfl_rss)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# toy problem for workflow figure""")
    return


@app.cell
def _(
    Weather,
    daily_water_demand,
    mass_water_harvester,
    mof_water_ads,
    optimize_harvester,
    predict_water_delivery,
):
    toy_weather = Weather(6, 2024, "Socorro", day_min=7, day_max=10)

    toy_mofs = ["MOF-303", "CAU-23", "MOF-801"]

    toy_water_del = predict_water_delivery(toy_weather, {mof: mof_water_ads[mof] for mof in toy_mofs})

    toy_opt_mass_of_mofs, toy_min_mass, toy_opt_info = optimize_harvester(toy_mofs, toy_water_del, daily_water_demand)

    toy_pure_mof_harvester = mass_water_harvester(toy_mofs, toy_water_del, daily_water_demand)
    return toy_mofs, toy_opt_mass_of_mofs, toy_water_del, toy_weather


@app.cell
def _(mpl, toy_weather):
    with mpl.rc_context({'figure.figsize': (3, 4.2)}):
        toy_weather.viz_timeseries(toy=True, save=True)
    return


@app.cell
def _(mof_water_ads, mpl, toy_mofs, viz_all_measured_adsorption_isotherms):
    with mpl.rc_context({'figure.figsize': (3, 3)}):
        viz_all_measured_adsorption_isotherms(mof_water_ads, toy_mofs, save_tag="toy")
    return


@app.cell
def _(
    mpl,
    toy_mofs,
    toy_water_del,
    toy_weather,
    viz_water_delivery_time_series,
):
    with mpl.rc_context({'figure.figsize': (3, 3)}):
        viz_water_delivery_time_series(toy_water_del, toy_weather, toy_mofs, toy=True)
    return


@app.cell
def _(mpl, toy_opt_mass_of_mofs, toy_weather, viz_optimal_harvester_pie):
    with mpl.rc_context({'figure.figsize': (3, 3)}):
        viz_optimal_harvester_pie(toy_opt_mass_of_mofs, toy_weather)
    return


if __name__ == "__main__":
    app.run()
