from networktables import NetworkTable
from visual import *


cam = box(pos=(0,0,0), size=(1,1,1), color=color.red)
while True:
    table = NetworkTable.getTable("ComVision")
    if table.containsKey("CamRotation"):
        key = table.getValue("CamRotation")
        cam.pos = (key[0], key[1], key[2])
