//Nibl - ESE350 Final Project
//Authors: Bhaskar Abhiraman and Jason Kaufmann
//Date: 5/1/2019

#include "mbed.h"
#include "Servo.h"
#include <PwmIn.h>
#include <stdlib.h>

// Serial port for showing RX data.
Serial pc(USBTX, USBRX);

//Two line communication with the BeagleBone
DigitalOut finishedAction(p30);
DigitalIn startAction(p29);


//Bottom servo
PwmOut bottom(p26);
PwmIn bottom_feedback(p22);

//Top servo for camera angle
Servo top(p24);

//Sent back to Beaglebone to verify that it finished moving
DigitalOut verify(p23);

//done taking picture and can move now
DigitalIn advanceAngleSignal(p25);

//IR Sensor
AK9753 IR(p28, p27, 0x64);

//UART communication between the Beaglebone and the Mbed
RawSerial beagbed(p9, p10);


//Variables for the Logitech C920 camera
int degreeChange = 45;
int hfov = 70;
int vfov = 43;
int hmid = hfov/2;
int vmid = vfov/2;

//State variable that indicates which mode the Mbed is in
enum doingThings {
    NOTHING,
    SCANNING,
    TRACKING,
    VOICEMODE,
};

doingThings state = NOTHING; //initialize it to doing nothing until it gets an action


int map(int x, int in_min, int in_max, int out_min, int out_max) {
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

/*
 * Sets the velocity of the bottom servo by mapping an rpm to a pulsewidth
 */
void setVelocity(float rpm, PwmOut* servo) {
    int pulse;
    if (rpm > 0) {
        pulse = map(rpm, 0, 140, 1480, 1280);
    } else if(rpm < 0) {
        pulse = map(rpm, 0, -140, 1520, 1720);
    } else {
        pulse = 100000;
    }
    servo->pulsewidth_us(pulse);
}

/*
 * Gets the current angle of the bottom servo using the PwmIn input
 */
float getAngle(PwmIn *servo) {
    float dc = (1.0-servo->dutycycle()) * 100; //we had to invert the signal so get the compliment of the duty cycle being read
    float angle = ((dc-2.5)*360)/(96.2-2.5+1); //formula from https://www.pololu.com/file/0J1395/900-00360-Feedback-360-HS-Servo-v1.2.pdf
    return angle;
}

void servoGoAngle(int angleDes, PwmOut* servo, PwmIn* servo_fb) {
    if (angleDes < 0) {
        angleDes+= 360;
    } else if (angleDes > 360) {
        angleDes%=360;
    }
    float angleCurr = getAngle(servo_fb);
    float Kp = 0.55;
    float Kd = 0;
    float Ki = 0;
    float error = angleDes-angleCurr;
    float errorPrev = 0;
    if (abs(error) > 180) {
            if (error < 0) {
                error += 360;
            }
            else if (error > 0) {
                error -=360;    
            }    
    }
    float errorCum = 0;
    float dt = 0.01;
    while(abs(error) > 4) {
        errorCum+=error;
        float rpm = Kp * error + Kd*(error-errorPrev)/dt + Ki*errorCum ;
        if (abs(rpm) < 10) {
            if(rpm > 0) {
                rpm = 10;
            } else {
                rpm=-10;
            }
        }
        setVelocity(static_cast<int>(rpm), servo);
        angleCurr = getAngle(servo_fb);
        errorPrev = error;
        wait_ms(10);
        error = angleDes-angleCurr;
        if (abs(error) > 180) {
            if (error < 0) {
                error += 360;
            }
            else if (error > 0) {
                error -=360;    
            }    
        }
    }
    setVelocity(0, servo);
}

void servoAdvanceAngle(int advanceAmount,PwmOut* servo, PwmIn* servo_fb) {
    float angleCurr = getAngle(servo_fb);
    int angleDes = static_cast<int>(angleCurr) + advanceAmount;
    servoGoAngle(angleDes, servo, servo_fb); 
}


//We were gonna use a stepper motor for shooting, but not needed anymore
/*void stepperGo(int numPulses, int speed) {
    for(int i =0; i<numPulses; i++){
        bottom_2 = 1;
        wait_us(speed);
        bottom_2 = 0;
        wait_us(speed);
    }
}

void stepperGoAngle(int angle, int speed) {
    int numPulses = map_int(angle, 0, 360, 0, 200);
    pc.printf("Num Pulses: %d \n", numPulses);
    stepperGo(numPulses, speed);
}*/


//------------------UNUSED FUNCTIONS---------------------//
//They were used to communicate with the IR and TOF sensors, but we ended up not using those
void setupTOFSensor(VL53L1X* tof){
    tof->softReset();
}

int getTOFValue(VL53L1X* tof) {
    int distance;
    tof->startMeasurement();
    while(tof->newDataReady() != true) {
    }
    distance = static_cast<int>(tof->getDistance());
    return distance;
}
void setupIRSensor(AK9753* ir) {
    ir->software_reset();
    ir->setEINTEN(0x01);
}

float getIRValue(AK9753* ir) {
    float irValues[4];
    ir->setECNTL1(0xAA);
    while(ir->dataReady() != true) {
    }
    irValues[0] = ir->getIR1();
    irValues[1] = ir->getIR2();
    irValues[2] = ir->getIR3();
    irValues[3] = ir->getIR4();
    ir->dataOverRun();

    pc.printf("%f", irValues[0]); pc.printf(" ");
    pc.printf("%f\n", irValues[2]); 

    
    float average = 0;
    float sum = 0;
    for(int i= 0; i<4; i++) {
        sum+=irValues[i];
    }
    average = sum/sizeof(irValues);  
    return average;
}

float getIndividualIRValue(AK9753* ir, int sensor) {
    ir->setECNTL1(0xAA);
    while(ir->dataReady() != true) {
    }
    
    float irValue = 0;
    switch(sensor) {
        case 1:
            irValue = ir->getIR1();
            break;
        case 2:
            irValue = ir->getIR2();
            break;
        case 3:
            irValue = ir->getIR3();
            break;
        case 4:
            irValue = ir->getIR4();
            break;
        }

    ir->dataOverRun();
    return irValue;
}
//----------------------------------------------------//

/*
void tracking(PwmOut* servo, PwmIn* servo_fb, AK9753* ir) {

    Servo top(p24);
    top.calibrate(0.00085, 90);
    setupIRSensor(ir);
    float errorHorz = getIndividualIRValue(ir, 3)-getIndividualIRValue(ir, 1);
    float errorVert = getIndividualIRValue(ir, 4)-getIndividualIRValue(ir, 2);
    float KpH = 0.04;
    float KpV = 0.05;
    float vertAngle = 0;
    while(true) {
        errorHorz = (getIndividualIRValue(ir, 3)+50)-getIndividualIRValue(ir, 1);
        errorVert = (getIndividualIRValue(ir, 4)+50)-getIndividualIRValue(ir, 2);
            if (abs(errorHorz) > 50) {
                float rpm = KpH * errorHorz;
                setVelocity(static_cast<int>(rpm), servo);
            }
            else {
                setVelocity(0, servo);   
            }
            
            if (abs(errorVert) > 40) {
                float rpmVert = KpV * errorVert;
                vertAngle += rpmVert;
                if(vertAngle < 0) {
                    top.position(0);
                } else if (vertAngle > 60) {
                    top.position(60);
                } else {
                    top.position(vertAngle);
                }
            }
            
            errorVert = (getIndividualIRValue(ir, 4)+50)-getIndividualIRValue(ir, 2);
            errorHorz = (getIndividualIRValue(ir, 3)+50)-getIndividualIRValue(ir, 1);
       // }
        //setVelocity(0, servo);
    }
    
}*/

/*
 * main function that scans the room and then goes to the angle with the max Cosmo and Wanda
 */
void scan(PwmOut* servo, PwmIn* servo_fb, AK9753* ir) {
    beagbed.getc();
    beagbed.getc();
    servoGoAngle(15, &bottom, &bottom_feedback);
    top.position(0);
    wait_ms(100);
    
    float angleCurr = getAngle(servo_fb);
    
    float angleMaxHorz = 0;
    float angleMaxVert = 0;
    int mass;
    int massMax = 0;
    int horzPercent;
    int vertPercent;
    int numOfMoves = 0;
    while(numOfMoves<8) {
        while(!advanceAngleSignal.read()) {
            wait_ms(1);
        }
        mass = beagbed.getc();
        horzPercent = beagbed.getc();
        vertPercent = beagbed.getc();
        pc.printf("%d kg \n", mass);
        pc.printf("%d \n", horzPercent);
        pc.printf("%d \n", vertPercent);
        pc.printf("\n");
        angleCurr = getAngle(servo_fb);
        if (mass > massMax) {
            massMax = mass;
            angleMaxHorz = angleCurr+(hfov*(horzPercent/100.0f)-hmid);
            angleMaxVert = vfov*(1- vertPercent/100.0f)-vmid;
        }
        
        servoGoAngle(degreeChange*(numOfMoves+1)+15, &bottom, &bottom_feedback);
        verify.write(1);
        wait_ms(25);
        verify.write(0);
        numOfMoves++;
    }

    setVelocity(0, servo);
    if (angleMaxHorz < 0) {
        angleMaxHorz+= 360;
    }
    
    int goPosition = static_cast<int>(angleMaxHorz);
    pc.printf("CURRENT ANGLE: %f\n", angleCurr);
    pc.printf("COLOR MAX HORZ ANGLE: %d\n", goPosition);
    pc.printf("COLOR MAX VERT ANGLE: %f\n", angleMaxVert);
    
    servoGoAngle(goPosition, &bottom, &bottom_feedback);
    top.position(angleMaxVert);
    verify.write(1);
    wait_ms(25);
    verify.write(0);
}

/*
 * main tracking function that takes runs for 10 iterations (10 pictures) and moves to the max location
 * each time.
 */
void tracking(PwmOut* servo, PwmIn* servo_fb) {
    pc.printf("IN TRACKING!\n");
    top.position(0);
    beagbed.getc();
    beagbed.getc();
    wait_ms(100);
    float deltaH = 0;
    int mass;
    int horzPercent;
    int vertPercent;
    float angleCurr = getAngle(servo_fb);
    float angleMaxVert = 0;
    int i = 0;

    while(i<10) {
        while(!advanceAngleSignal.read()) {
        }
        mass = beagbed.getc();
        horzPercent = beagbed.getc();
        vertPercent = beagbed.getc();
        pc.printf("%d kg \n", mass);
        pc.printf("%d \n", horzPercent);
        pc.printf("%d \n", vertPercent);
        pc.printf("\n");
        angleCurr = getAngle(servo_fb);
        if (mass > 2 && horzPercent < 101 &&  vertPercent < 101) {
            deltaH = (hfov*(horzPercent/100.0f)-hmid);
            pc.printf("deltaH: %f", deltaH);
            servoGoAngle(angleCurr + deltaH, &bottom, &bottom_feedback);
            angleMaxVert = (vfov*(1- vertPercent/100.0f)-vmid) + angleMaxVert;
            pc.printf("deltaV: %f", angleMaxVert);
            top.position(angleMaxVert);
        }
        pc.printf("TAKE ANOTHER PIC!\n");
        verify.write(1);
        wait_ms(25);
        verify.write(0);
        i++;
    }
}

/*
 * main voice_mode function that gets voice commands from the Beaglebone and executes until the user says "Hit me"
 */
void voice_mode(PwmOut* servo, PwmIn* servo_fb) {
    pc.printf("IN VOICEMODE!\n");
    top.position(0);
    beagbed.getc();
    beagbed.getc();
    wait_ms(100);
    
    int command = -1;
    float currHorzAngle = -1;
    float currVertAngle = 0;
    int horzMove = 45;
    int vertMove = 20;
    float angleDes = -1;
    while(command != 5) {
        while(!advanceAngleSignal.read()) {
        }
        currHorzAngle = getAngle(servo_fb);
        command = beagbed.getc();
        beagbed.getc();
        beagbed.getc();
        switch (command) {
            case 1:
                pc.printf("RIGHT\n");
                servoGoAngle(currHorzAngle + horzMove, &bottom, &bottom_feedback);
                break;
            case 2:
                pc.printf("LEFT\n");
                servoGoAngle(currHorzAngle - horzMove, &bottom, &bottom_feedback);
                break;
            case 3:
                pc.printf("UP\n");
                angleDes = currVertAngle+vertMove;
                if (angleDes >= 75) {
                    angleDes = 75;
                } else if (angleDes <= -40) {
                    angleDes = -40;
                }
                currVertAngle = angleDes;
                top.position(angleDes);
                break;
            case 4:
                pc.printf("DOWN\n");
                angleDes = currVertAngle-vertMove;
                if (angleDes >= 75) {
                    angleDes = 75;
                } else if (angleDes <= -40) {
                    angleDes = -40;
                }
                currVertAngle = angleDes;
                top.position(angleDes);
                break;
        }
        pc.printf("GET ANOTHER VOICE COMMAND!\n");
        pc.printf("\n");
        wait_ms(1000);
        verify.write(1);
        wait_ms(25);
        verify.write(0);
    }
}

int main() {
    pc.baud(115200);
    beagbed.baud(9600);
    
    bottom.period_ms(20);

    AK9753* ir = &IR;
    while(true) {
        if(state == NOTHING) {
            //get commands from the Beaglebone
            char command = beagbed.getc(); //get an 8bit number from the UART interface
            if (command == 255) { //this is -1 in signed 8bit binary
                state = SCANNING;
            } else if (command == 254) {  //this is -2 in signed 8bit binary
                state = TRACKING;
            } else if (command == 253) {  //this is -3 in signed 8bit binary
                state = VOICEMODE;
            }
        } else if (state == SCANNING){
            pc.printf("SCAN ACTION RECIEVED! \n");
            scan(&bottom, &bottom_feedback, ir);
            wait(1);
            state = NOTHING;
            pc.printf("FINISHED ACTION! \n");
        } else if (state == TRACKING) {
            pc.printf("TRACK ACTION RECIEVED! \n");
            tracking(&bottom, &bottom_feedback);
            wait(1);
            state = NOTHING;
            pc.printf("FINISHED ACTION! \n");
        } else if (state == VOICEMODE) {
            pc.printf("VOICE MODE ACTION RECIEVED! \n");
            voice_mode(&bottom, &bottom_feedback);
            wait(1);
            state = NOTHING;
            pc.printf("FINISHED ACTION! \n");
        }
        wait_ms(50);
    } 
}