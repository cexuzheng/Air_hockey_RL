#include <Wire.h>
#include <math.h>
#include <Adafruit_PWMServoDriver.h>
#include <ArduinoJson.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();
//Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40, Wire);


// Depending on your servo make, the pulse width min and max may vary, you 
// want these to be as small/large as possible without hitting the hard stop
// for max range. You'll have to tweak them as necessary to match the servos you
const int USMIN = 500;            // This is the rounded 'minimum' microsecond length based on the minimum pulse of 150
const int USMAX = 2500;           // This is the rounded 'maximum' microsecond length based on the maximum pulse of 600
const float MIN_ANGL = -2.35619;  // Minimum angle  
const float MAX_ANGL = 2.35619;   // Maximum angle
const int SERVO_FREQ = 50;        // Servo frequency

// ROBOT DATA
const uint8_t SERVO_q1 = 0;       // Q1 servo channel
const uint8_t SERVO_q2 = 1;       // Q2 servo channel
const float a1 = 0.2;             // Shoulder link length
const float a2 = 0.2;             // Elbow link length
const float MIN_LIM_q1 =  MIN_ANGL; // Physical limits
const float MAX_LIM_q1 =  MAX_ANGL; // Physical limits
const float MIN_LIM_q2 =  MIN_ANGL; // Physical limits
const float MAX_LIM_q2 =  MAX_ANGL; // Physical limits

//GLOBAL VARIABLES
float real_x, real_y, real_q1, real_q2; //REAL ROBOT STATE
DynamicJsonDocument doc(256);


float mapfloat(float x, float in_min, float in_max, float out_min, float out_max)
{
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}


int angle2microsec(float angle){
  return mapfloat(angle, MIN_ANGL, MAX_ANGL, USMAX, USMIN); // Regla de la mano derecha
}

bool inverseKinematics2DOF(float x, float y, float &q1, float &q2){

  //Elbow DOWN option
  q2 = acos( (pow(x,2) + pow(y,2) - pow(a1,2) - pow(a2,2))/(2*a1*a2) );
  q1 = atan2(y,x) - atan2( (a2*sin(q2)), (a1 + a2*cos(q2)) );

  if( (q1>MIN_LIM_q1 and q1<MAX_LIM_q1 and q2>MIN_LIM_q2 and q2<MAX_LIM_q2) == false ){
    q2 = -acos( (pow(x,2) + pow(y,2) - pow(a1,2) - pow(a2,2))/(2*a1*a2) );
    q1 = atan2(y,x) + atan2( (a2*sin(q2)), (a1 + a2*cos(q2)) );
  }
    return (q1>MIN_LIM_q1 and q1<MAX_LIM_q1 and q2>MIN_LIM_q2 and q2<MAX_LIM_q2);
}



////////////////////////////////////
////////// SETUP FUNCTION //////////
////////////////////////////////////
void setup() {
  Serial.begin(2000000);
  Serial.setTimeout(1);
  pwm.begin();
  /* In theory the internal oscillator (clock) is 25MHz but it really isn't
   * that precise. You can 'calibrate' this by tweaking the setOscillatorFrequency 
   * until you get the PWM update frequency you're expecting!
   */
  pwm.setOscillatorFrequency(27000000);
  pwm.setPWMFreq(SERVO_FREQ);  
  delay(10);


  //Move the robot to known position
  real_x=0.2;
  real_y=0.2;
  inverseKinematics2DOF(real_x, real_y, real_q1, real_q2);
  pwm.writeMicroseconds(SERVO_q1, angle2microsec(real_q1));  
  pwm.writeMicroseconds(SERVO_q2, angle2microsec(real_q2));
  delay(500);
  //Serial.println("**********************");
  //Serial.print(real_x);Serial.print("  ");Serial.println(real_y);
  //Serial.print(real_q1);Serial.print("  ");Serial.println(real_q2);
  //Serial.print(angle2microsec(real_q1));Serial.print("  ");Serial.println(angle2microsec(real_q2));
  
}



////////////////////////////////////
////////// LOOP FUNCTION ///////////
////////////////////////////////////


void loop() {

  /*real_x=0.2;
  real_y=0.2;
  inverseKinematics2DOF(real_x, real_y, real_q1, real_q2);
  Serial.println("**********************");
  Serial.print(real_x);Serial.print("  ");Serial.println(real_y);
  Serial.print(real_q1);Serial.print("  ");Serial.println(real_q2);
  Serial.print(angle2microsec(real_q1));Serial.print("  ");Serial.println(angle2microsec(real_q2));
  pwm.writeMicroseconds(SERVO_q1, angle2microsec(real_q1));  
  pwm.writeMicroseconds(SERVO_q2, angle2microsec(real_q2));
  delay(500);*/
  
  
  while(Serial.available()){
    String input_msg = Serial.readString();
    
    if(input_msg =="state"){
      String state_json = String("{\"x\":") + String(real_x) + String(", \"y\":") + String(real_y) + String("}");
      Serial.println(state_json);
    }else{
      float timer = millis();
      deserializeJson(doc, input_msg);

      if(doc.containsKey("x") and doc.containsKey("y")){
        float next_q1, next_q2, next_x, next_y;
        next_x=doc["x"];
        next_y=doc["y"];
        
        if(inverseKinematics2DOF(next_x, next_y, next_q1, next_q2)){
          int time_up_q1, time_up_q2;
          time_up_q1 = angle2microsec(next_q1);
          time_up_q2 = angle2microsec(next_q2);
          pwm.writeMicroseconds(SERVO_q1, time_up_q1);  
          pwm.writeMicroseconds(SERVO_q2, time_up_q2);
      
          
          //UPDATE ROBOT STATE
          real_x = next_x;
          real_y = next_y; 
          real_q1 = next_q1;
          real_q2 = next_q2;
        }
      }
      
    }
  
  }


}
