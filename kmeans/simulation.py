import csv
import queue
import math
import random

X_RANGE = 6673
Y_RANGE = 4449


DRIVERS = 400
MOVESPEED = 5
WANDERSPEED = 4
orders = queue.Queue()
WaitedOrders = set()
OrderedCars = queue.PriorityQueue()
WanderingCars = []



def calc_angle(x1,y1,x2,y2):
    angle=0
    dy= y2-y1
    dx= x2-x1
    if dx==0 and dy>0:
        angle = 0
    if dx==0 and dy<0:
        angle = 180
    if dy==0 and dx>0:
        angle = 90
    if dy==0 and dx<0:
        angle = 270
    if dx>0 and dy>0:
       angle = math.atan(dx/dy)*180/math.pi
    elif dx<0 and dy>0:
       angle = 360 + math.atan(dx/dy)*180/math.pi
    elif dx<0 and dy<0:
       angle = 180 + math.atan(dx/dy)*180/math.pi
    elif dx>0 and dy<0:
       angle = 180 + math.atan(dx/dy)*180/math.pi
    return angle


class Order:
    def __init__(self,x1,y1,t1,x2,y2,t2):
        self.begin = (x1,y1)
        self.end = (x2,y2)
        self.time = t1
        self.length = t2 - t1

class Car:
    def __init__(self):
        self.position = (random.random()*X_RANGE, random.random()*Y_RANGE)
        self.time = 0
        self.diret = random.random() * math.pi * 2

    def move(self,t):
        self.position = ((self.position[0] + (t - self.time) * WANDERSPEED * math.cos(self.diret)) % X_RANGE,
                         (self.position[1] + (t - self.time) * WANDERSPEED * math.sin(self.diret)) % Y_RANGE)
        self.time = t
        self.diret = random.random()* math.pi * 2

def can_match(order, car):
    return (abs(car.position[0] - order.begin[0]) < 500 and abs(car.position[1] - order.begin[1]) < 500)

def car_order_distance(order, car):
    return math.sqrt((order.begin[0] - car.position[0])**2 + (order.begin[1] - car.position[1])**2 )

with open('simulation.csv','r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        row = [int(i) for i in row]
        orders.put(Order(row[1],row[2],row[0],row[4],row[5],row[3]))

tick = 0

WanderingCars = [Car() for i in range(DRIVERS)]


while not orders.empty():
    current_order = orders.get()
    WaitedOrders.add(current_order)
    newWaitedOrders = set()
    for order in WaitedOrders:
        car_to_choose = []
        for car in WanderingCars:
            if can_match(current_order,car):
                WanderingCars.remove(car)
                car_to_choose.append(car)
        if car_to_choose: # car matched
            car_to_choose.sort(key=lambda cur_car: car_order_distance(order,cur_car))
            car_to_choose[0].position = order.end
            car_to_choose[0].time = current_order.time + order.length + car_order_distance(order, car_to_choose[0]) / MOVESPEED
            OrderedCars.put((car_to_choose[0].time,car_to_choose[0]))
            WanderingCars += car_to_choose[1:]
        else:
            newWaitedOrders.add(order)
    WaitedOrders = newWaitedOrders
    tick = current_order.time
    if not OrderedCars.empty():
        while OrderedCars.queue[0][0] < tick:
            WanderingCars.append(OrderedCars.get()[1])
        for car in WanderingCars:
            car.move(tick)
    print (tick, len(WanderingCars), OrderedCars.qsize(), len(WaitedOrders))