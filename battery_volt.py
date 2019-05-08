#!/usr/bin/env python3

# Contributed by BrendanSimon

# import rcpy libraries
import rcpy
import rcpy.adc as adc
import time

def adc_test():

    # Read DC Jack and Battery voltages via function calls.
    dc_jack_voltage = adc.get_dc_jack_voltage()
    battery_voltage = adc.get_battery_voltage()
    print("dc-jack : voltage={:+6.2f} || battery : voltage={:+6.2f}".format(dc_jack_voltage, battery_voltage))
        
if __name__ == "__main__":
    while(True):
        adc_test()
        time.sleep(0.1)
        print(chr(27)+"[2J")


