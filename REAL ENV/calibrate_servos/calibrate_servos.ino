#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();   // Initiates library.

#define SERVOMIN  500// Minimum pulse length count out of 4096.
#define SERVOMAX  2500 // Maximum pulse length count out of 4096.

int servoNo = 0; // Defines a counter for servos.


void setup() 
{
  Serial.begin(9600);       // Starts serial connecton at 9600 baud rate.
  pwm.begin();         // Sends PWM signals.
  pwm.setOscillatorFrequency(27000000);
  
  pwm.setPWMFreq(50);  // Makes servos run at 60 Hz rate.
  delay(20);
}


void loop() 
{   
  
  for (int pulselen = SERVOMIN; pulselen < SERVOMAX; pulselen++){ // Drives each servo one at a time first                                                                                                
    pwm.writeMicroseconds(0, 500);
    pwm.writeMicroseconds(1, 500);                                // to maximum pulse length then to minimum pulse length.
  }
  delay(600);
  
  for (int pulselen = SERVOMAX; pulselen > SERVOMIN; pulselen--){
    pwm.writeMicroseconds(0, 2500);
    pwm.writeMicroseconds(1, 2500);
  }
  delay(600);

  pwm.writeMicroseconds(0, (1500.0));
  pwm.writeMicroseconds(1, (1500.0));                           // to maximum pulse length then to minimum pulse length.
  delay(2000);
} 
