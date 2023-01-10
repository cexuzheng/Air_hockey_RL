# Importing Libraries
import serial
import time
import json


arduino = serial.Serial(port='/dev/ttyUSB0', baudrate=2000000, timeout=.1)

def read_arduino():
    data = arduino.readline()
    return data.decode()
    
def write_arduino(x):
    arduino.write(bytes(x, 'utf-8'))
    time.sleep(0.01)
    
while True:
    write_arduino("{\"x\":0.22, \"y\":0.50 }")
    time.sleep(0.05)
    write_arduino("state")

    #value = read_arduino()
    #time.sleep(0.05)
    #print(value) # printing the value
    
    value = read_arduino()
    json_is_valid=True
    try:
        data = json.loads(value)
    except ValueError as e:
        json_is_valid = False  
    time.sleep(0.05)
    
    if(json_is_valid):
        print("x: ", data["x"]) 
        print("y: ", data["y"]) 
        print()
    
