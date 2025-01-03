import marimo

__generated_with = "0.10.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import interpolate
    from scipy.optimize import linprog
    from dataclasses import dataclass
    import os

    # matplotlib styles
    from aquarel import load_theme
    theme = load_theme("scientific").set_font(size=18)
    theme.apply()
    return (
        dataclass,
        interpolate,
        linprog,
        load_theme,
        mo,
        mpl,
        np,
        os,
        pd,
        plt,
        theme,
    )


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

        | MOF | original reference | data extraction method | confirmed data fidelity | notes |
        | -- | -- | -- | -- | -- |
        | MOF-801 | [link](https://doi.org/10.1038/s41467-018-03162-7) | plot digitized from SI Fig. 6a | ✅ | |
        | KMF-1 | [link](https://www.nature.com/articles/s41467-020-18968-7) | plot digitized from Fig. 2B | ✅ |  |
        | CAU-23 | [link](https://www.nature.com/articles/s41467-019-10960-0)| plot digitized from Fig 2 | |
        | MIL-160 | [link](https://onlinelibrary.wiley.com/doi/10.1002/adma.201502418) | plot digitized from SI page 7 | |
        | Y-shp-MOF-5 | [link](https://pubs.acs.org/doi/10.1021/jacs.7b04132) | plot digitized from Fig. 2 | |
        | MOF-303 | [link](https://www.science.org/doi/10.1126/science.abj0890) | plot digitized from Fig. 1 A | |
        | CAU-10H | [link](https://pubs.rsc.org/en/content/articlelanding/2014/dt/c4dt02264e)| plot digitized from Fig. 2 | |
        | Al-Fum | [link](https://pubs.rsc.org/en/content/articlelanding/2014/ra/c4ra03794d) | plot digitized from Fig. 3 | |

        we extracted all water adsorption data from plots in the papers using [plot digitizer](https://www.graphreader.com/v2). we took only the _adsorption_ branch, neglecting hysteresis.

        below, our class `MOFWaterAds` aims to:

        * read in the raw adsorption data
        * visualize the raw adsorption data
        * employ Polanyi potential theory to predict adsorption in MOFs at any temperature and pressure.

        Cory's notes: 
        """
    )
    return


@app.cell
def _():
    # list of MOFs
    mofs = ["MOF-801", "KMF-1"]

    # maps MOF to the temperatures at which we possess adsorption data
    mof_to_data_temperatures = {
        "MOF-801": [15, 25, 45, 65, 85],
        "KMF-1": [25]
    }
    return mof_to_data_temperatures, mofs


@app.cell
def _():
    R = 8.314 / 1000.0 # universal gas constant [kJ/(mol-K)]
    return (R,)


@app.cell
def _(mpl):
    # stuff for data viz

    # commonly-used plot labels
    axis_labels = {
        'pressure': 'relative humidity, $P/P_0$',
        'adsorption': 'water uptake, $w$ [kg/kg]',
        'potential': 'Polanyi Potential, $A(T, P/P_0)$ [kJ/mol]'
    }

    # mapping temperature to color
    temperature_cmap = mpl.colormaps["inferno"]
    temperature_cmap_norm = mpl.colors.Normalize(vmin=15.0, vmax=90.0)

    def T_to_color(temperature):
        if temperature < temperature_cmap_norm.vmin or temperature > temperature_cmap_norm.vmax:
            raise Exception("out of temperature normalization range.")

        return temperature_cmap(temperature_cmap_norm(temperature))

    temperature_cmap
    return T_to_color, axis_labels, temperature_cmap, temperature_cmap_norm


@app.cell
def _(R, T_to_color, axis_labels, interpolate, np, pd, plt):
    class MOFWaterAds:
        def __init__(self, mof, fit_temperature, data_temperatures):
            """
            mof (string): name of MOF. e.g. "MOF-801"
            data_temperatures: list of data temperatures in degrees C.
            """
            if not fit_temperature in data_temperatures:
                raise Exception("fit temp not in data temps!")

            self.mof = mof
            self.fit_temperature = fit_temperature
            self.data_temperatures = data_temperatures

            # fit char. curve
            self.fit_characteristic_curve()

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

            # Polanyi adsorption potential for every P/P_0 in the data set
            ads_data["A [kJ/mol]"] = -R * (temperature + 273.15) * np.log(ads_data['P/P_0'])

            return ads_data

        def fit_characteristic_curve(self):
            # read in adsorption isotherm data at fit temperature
            data = self._read_ads_data(self.fit_temperature)

            # sort rows by A values
            data = data.sort_values('A [kJ/mol]')

            # monotonic interpolation of water ads. as a function of Polanyi potential A.
            self.ads_of_A = interpolate.PchipInterpolator(
                data['A [kJ/mol]'].values, data['Water Uptake [kg kg-1]'].values
            )

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

            # compute water adsorption at this A on the char. curve
            return self.ads_of_A(A).item() # kg/kg

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

            plt.legend(title="T = {}$^\circ$C".format(temperature))
            plt.ylim(ymin=0)
            plt.xlim(xmin=0)
            plt.title(self.mof)
            plt.show()

        def viz_adsorption_isotherms(self, incl_predictions=True):
            plt.figure()
            plt.title('water adsorption isotherms')
            plt.xlabel(axis_labels['pressure'])
            plt.ylabel(axis_labels['adsorption'])
            for temperature in self.data_temperatures:
                # read ads isotherm data
                data = self._read_ads_data(temperature)

                # draw data
                plt.scatter(
                    data['P/P_0'], data['Water Uptake [kg kg-1]'], 
                    clip_on=False, color=T_to_color(temperature), label="{}$^\circ$C".format(temperature)
                )        
                if incl_predictions:
                    p_ovr_p0s = np.linspace(0, 1, 100)[1:]
                    plt.plot(
                        p_ovr_p0s, [self.predict_water_adsorption(temperature, p_ovr_p0) for p_ovr_p0 in p_ovr_p0s], 
                        color=T_to_color(temperature)
                    )
            plt.ylim(ymin=0)
            plt.xlim(xmin=0)
            plt.title(self.mof)
            plt.legend(prop={'size': 8})
            plt.show()

        def plot_characteristic_curves(self, incl_model=True):
            plt.figure()
            plt.xlabel(axis_labels['potential'])
            plt.ylabel(axis_labels['adsorption'])

            A_max = -1.0 # for determining axis limits
            for temperature in self.data_temperatures:
                # read ads isotherm data
                data = self._read_ads_data(temperature)

                # draw data
                plt.scatter(
                    data['A [kJ/mol]'], data['Water Uptake [kg kg-1]'], 
                    clip_on=False, color=T_to_color(temperature), label="{}$^\circ$C".format(temperature)
                )

                # track A_max
                if data['A [kJ/mol]'].max() > A_max:
                    A_max = data['A [kJ/mol]'].max()

            if incl_model:
                As = np.linspace(0, A_max)
                plt.plot(As, self.ads_of_A(As), color=T_to_color(self.fit_temperature))

            plt.ylim(ymin=0)
            plt.xlim(xmin=0)
            plt.title(self.mof)
            plt.legend(prop={'size': 8})
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
        # temperature [°C]
        45, 
        # list of temperatures for which we have data [°C]
        mof_to_data_temperatures["MOF-801"]
    )
    return (mof,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""retreive the raw data from an adsorption isotherm measurement.""")
    return


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
        # RH (fraction)
        0.2
    ) # g H20 / g MOF
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""visualize the adsorption isotherm data (dots) and Polanyi theory fit (lines).""")
    return


@app.cell
def _(mof):
    mof.viz_adsorption_isotherm(45)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""visualize all measured adsorption data.""")
    return


@app.cell
def _(mof):
    mof.viz_adsorption_isotherms()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""plot the characteristic curves (both data and theory).""")
    return


@app.cell
def _(mof):
    mof.plot_characteristic_curves()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### KMF-1""")
    return


@app.cell
def _(MOFWaterAds, mof_to_data_temperatures):
    _mof = MOFWaterAds(
        # name of MOF crystal structure
        "KMF-1", 
        # temperature [°C]
        25, 
        # list of temperatures for which we have data [°C]
        mof_to_data_temperatures["KMF-1"]
    )
    _mof.viz_adsorption_isotherms()
    return


@app.cell
def _(mo):
    mo.md(r"""### CAU-23""")
    return


@app.cell
def _(MOFWaterAds, mof_to_data_temperatures):
    _mof = MOFWaterAds(
        # name of MOF crystal structure
        "CAU-23", 
        # temperature [°C]
        20, 
        # list of temperatures for which we have data [°C]
        mof_to_data_temperatures["CAU-23"]
    )
    _mof.viz_adsorption_isotherms()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # ⛅ weather data

        📍 Pheonix, Arizona. Sky Harbor International Airport.

        want weather data frame to look like:

        | date | T_night | RH_night | T_day | RH_day | solar_flux day |
        | ---  | --- | --- | ---  | --- |--- |

        where each row is a unique day.

        how is night and day determined? a given time? or the min/max? let's go with min/max.

        weather data needs more processing if you want to calculate solar flux, as file changes
        """
    )
    return


@app.cell
def _(pd, self):
    class Weather:
        def __init__(self, month):
            self.month = month

        def read_raw_weather_data(self):
            weather_filename = 'data/Weather_noclouds/PHX_{}_2023.csv'.format(self.month)
            raw_data = pd.read_csv(weather_filename)
            self.raw_weather_data = raw_data
        def process_weather_data(self):

            raw_data = self.read_raw_weather_data()
            processing_data = raw_data

            processing_data['Date'] = pd.to_datetime(processing_data['DateTime']).dt.date
            processing_data['Time'] = pd.to_datetime(processing_data['DateTime']).dt.time
            processing_data = processing_data.drop('DateTime',axis=1)
            processing_data['Temperature'] = (processing_data['Temperature'] - 32) * 5/9
            processing_data['Relative Humidity'] = processing_data['Relative Humidity'] / 100
            processing_data = processing_data.drop('Dew Point',axis=1)

            processed_data = processing_data
            return processed_data


        def night_conditions(self, day):

            date = '2023-{}-{}'.format(self.month,day)

            month_data = self.process_weather_data()
            month_data["Date"] = month_data["Date"].astype(str)
            date_data = month_data[month_data["Date"] == date]
            date_data = date_data.reset_index(drop=True)

            night_conditions = date_data[date_data["Temperature"] == date_data["Temperature"].max()]
            temp = night_conditions['Temperature'].astype(float)
            RH = night_conditions["Relative Humidity"].astype(float)
            time = night_conditions['Time']

            return temp, RH, time



        def day_conditions(self, day):
            # uses solar flux somehow

            date = '2023-{}-{}'.format(self.month,day)

            month_data = self.process_weather_data()
            month_data["Date"] = month_data["Date"].astype(str)
            date_data = month_data[month_data["Date"] == date]
            date_data = date_data.reset_index(drop=True)

            day_conditions = date_data[date_data["Temperature"] == date_data["Temperature"].max()]
            day_conditions = day_conditions.reset_index(drop=True)
            temp = day_conditions['Temperature']
            temp = temp[0]
            RH = day_conditions["Relative Humidity"]
            RH = RH[0]
            time = day_conditions['Time']
            time = time[0]

            return print(type(time))

        def monthly_weather():

            month_data = self.process_weather_data()
            day_list = month_data['Date'].unique()

            results = [
                {"day": day, "temperature": day_temp, "humidity": day_RH, "time": day_time}
                for day in day_list
                for day_temp, day_RH, day_time in [self.day_conditions(day)]]

            monthly_conditions = pd.DataFrame(results)

            return monthly_conditions
    return (Weather,)


@app.cell
def _(Weather):
    weather = Weather('06')
    weather.day_conditions('05')
    return (weather,)


@app.cell
def _(weather):
    weather.day_conditions('05')
    return


@app.cell
def _():
    return


app._unparsable_cell(
    r"""
    def weather_data(month):
        # reads in raw weather data
        # processes it so each row is a unique day.
    """,
    name="_"
)


@app.cell
def _(mo):
    mo.md(r"""# 🚿 modeling water delivery of each MOF on each day""")
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""# optimizing the water harvester""")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
