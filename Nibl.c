//
// Created by Jason Kaufmann on 2019-04-14.
//

#include <stdio.h>
#include <math.h>
#include <robotcontrol.h>

int main() {
    while(1) {
        float voltage = rc_adc_read_volt(0);
        printf("ADC Voltage: %6.3f", voltage);
    }
    return 0;
}
