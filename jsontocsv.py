# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 19:27:42 2020

@author: Nimisha
"""

import json
import csv
#from pprint import pprint

with open('twwetexp2.json',encoding="utf8") as f:
    data = json.loads("[" + 
        f.read().replace("}\n{", "},\n{") + 
    "]")

    #print(data)
with open("exp.csv", "w",encoding="utf8") as file:
    csv_file = csv.writer(file)
    for item in data:
        csv_file.writerow([item['id'], item['tweet']])
