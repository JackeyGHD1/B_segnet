#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 deep <deep@deep-All-Series>
#
# Distributed under terms of the MIT license.

"""
b-segnetのpredict.pyで読み込むためのcsvファイルを作成
"""

import glob
import numpy as np

path = "data/CamVid/test/"

f_list = glob.glob(path + "*")
f_sorted = sorted(f_list)

f = open(path + "../test.txt", "w")
for line in f_sorted:
    f.write(line + "," + line.replace("/test/", "/testannot/"))
    f.write("\n")


