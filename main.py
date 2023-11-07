import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt
import os

target_star = "Kepler-8"

# Step 1: Search for target pixel file observation data
search_result = lk.search_targetpixelfile(target_star, cadence="long")

def retrieveBestData(search_result):
    if len(search_result) > 0:
        best_result = search_result[0]
        longest_exptime = best_result.exptime

        for result in search_result:
            exptime = result.exptime
            if exptime > longest_exptime:
                best_result = result
                longest_exptime = exptime

        # Step 3: Download the target pixel file
        print(f"Best observation data: {best_result.mission[0]} {best_result.target_name[0]} ({best_result.exptime[0]})\n")
        return best_result.download()
    else:
        print(f"No search results found for {target_star}")
        os._exit(1)

# Step 2: Filter results and select the one with the longest exposure time (exptime)
tpf = retrieveBestData(search_result)

tpf.plot(frame=0).set_title(target_star)

# Step 4: Convert the target pixel file to a lightcurve
lc = tpf.to_lightcurve(aperture_mask=tpf.pipeline_mask)
lc.plot().set_title("Lightcurve")

# Step 5: Remove NaN values from the lightcurve
lc = lc.remove_nans()

# Step 6: Flatten the lightcurve
flat_lc = lc.flatten()
flat_lc.plot().set_title("Flattened")

# Step 7: Generate a periodogram using the Box Least Squares (BLS) method
period = np.linspace(1, 5, 10000)
bls = lc.to_periodogram(method='bls', period=period, frequency_factor=500)
planet_x_period = bls.period_at_max_power
planet_x_t0 = bls.transit_time_at_max_power
planet_x_dur = bls.duration_at_max_power

# Step 8: Fold the flattened lightcurve with the detected period
folded_lc = flat_lc.fold(period=planet_x_period)
folded_lc.plot().set_title("Folded")

# Step 9: Plot the folded lightcurve with a limited x-axis range
ax = lc.fold(period=planet_x_period, epoch_time=planet_x_t0).scatter()
ax.set_xlim(-2, 2)
ax.set_title("Scattered Folded")

# Step 10: Print the detected planet's period, transit time, and duration
print(f"Detected planet period: {planet_x_period} days")
print(f"Transit time of the detected planet: {planet_x_t0} days")
print(f"Duration of the detected planet transit: {planet_x_dur} days")

plt.show(block=True)
