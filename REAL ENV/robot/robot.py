# Importing Libraries
import serial
import time
arduino = serial.Serial(port='/dev/ttyUSB1', baudrate=2000000, timeout=.1)

def read_arduino():
    data = arduino.readline()
    return data.decode()
    
def write_arduino(x):
    arduino.write(bytes(x, 'utf-8'))
    time.sleep(0.01)
    
while True:
    #write_arduino("state")
    write_arduino("{\"x\":0.15, \"y\":-0.15 }")
    value = read_arduino()
    print(value) # printing the value
