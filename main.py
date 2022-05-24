from flask import Flask, request, render_template
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

Verification = Flask(__name__)

with open("./model.pickle", "rb") as f:
    rfc = pickle.load(f)

@Verification.route("/")
def root():
    return render_template("CodeDemo.html")

@Verification.route("/predict", methods = ["post",])
def predict():
    info = request.form["info"]
    info = info.split("\n")
    final_gap = eval(info[-1].strip())
    
    trace = []
    for i in range(0, len(info) - 1):
        time = info[i].split(":")[0].strip()
        pos =  info[i].split(":")[1].strip()
        timegap = time
        x = pos.split(" ")[0].strip()
        y = pos.split(" ")[1].strip()
        trace.append({"timegap": timegap, "x": x, "y": y})
    
    tmp = []#去重,去掉那些完全一致的轨迹坐标
    for j in range(0, len(trace)):
        if trace[j] not in tmp:
            if len(tmp) > 0:
                if tmp[-1]["x"] != trace[j]["x"] or tmp[-1]["y"] != trace[j]["y"]:
                    tmp.append(trace[j])
            else:
                tmp.append(trace[j])
    trace = tmp
  
    avg_y = 0.0#计算y坐标的平均值
    for j in range(0, len(trace)):#对于轨迹的每一个位置
        trace[j]["timegap"] = eval(trace[j]["timegap"])#进行数值转换
        trace[j]["x"] = eval(trace[j]["x"])
        trace[j]["y"] = eval(trace[j]["y"])
        avg_y += trace[j]["y"]
           
    avg_y = avg_y / len(trace)#y坐标的平均值
    total_time = trace[len(trace) - 1]["timegap"] - trace[0]["timegap"]
    
    speed = []
    speed_sum = 0.0
    variance_y = 0.0
    for j in range(0, len(trace)):#第二轮遍历,计算速度,y坐标的方差,x方向的平均速度
        spd = 0.0
        variance_y += (trace[j]["y"] - avg_y) ** 2.0#累加方差
        if j == 0:#第一个点的速度肯定是0
            speed.append(spd)
        else:
            #这里其实不太严谨,选取的速度是当前点与上一个点这段位移的平均速度,严格来说并不是当前点的速度
            #另外,这里算出来的是绝对速度而不是矢量速度
            dist = math.sqrt((trace[j]["x"] - trace[j - 1]["x"]) ** 2.0 + (trace[j]["y"] - trace[j - 1]["y"]) ** 2.0)
            time = trace[j]["timegap"] - trace[j - 1]["timegap"]
            if time == 0.0:#有时候会出现这种记录,说明用户在一毫秒以内完成了这段位移,这种数据不能作废,当作用户是1ms完成这段位移即可
                time = 1.0
            spd = dist / time
            speed.append(spd)
            
        speed_sum += spd
        
    #存储计算所得的速度数据
    if len(speed) == 1:
        speed.append(0.0)#有的数据就一个位置信息,补一个数值避免下面speed切片出错
    avg_speed = speed_sum / len(trace)
    min_speed = min(speed[1: ])
    y_variance = variance_y / len(trace)#记录y坐标方差
    
    acceleration = []
    acceleration_sum = 0.0
    for j in range(0, len(trace)):#第三轮遍历,计算加速度
        acc = 0.0
        if j == 0:
            acceleration.append(acc)
        else:
            #同样的,用的是当前点与上一个点算加速度,所以严格来说并不是当前点的加速度
            #不是矢量加速度,算出来的是标量
            delta_speed = speed[j] - speed[j - 1]
            time = trace[j]["timegap"] - trace[j - 1]["timegap"]
            if time == 0.0:#依旧,对于1ms内的位移当作1ms处理
                time = 1.0
            acc = delta_speed / time
            acceleration.append(acc)
        acceleration_sum += acc
        
    min_acceleration = min(acceleration)
    
    feature = [[y_variance, avg_speed, min_acceleration, min_speed, total_time, float(final_gap)]]
    print(feature)
    ss = MinMaxScaler()
    feature = ss.fit_transform(feature)
    res = rfc.predict(feature)[0]
    print(res)
    return str(res)

if __name__ == '__main__':
    Verification.run(port = 80, debug = True)