ADL is a public dataset, which can be found from
https://archive.ics.uci.edu/ml/datasets/Dataset+for+ADL+Recognition+with+Wrist-worn+Accelerometer

ADL Dataset: It is a public wrist-worn accelerometer
dataset, which records totally 16 volunteers performing different
daily activities. Data are collected by a single 3-axis
accelerometer at sampling rate 32Hz. There are 689 raw data
samples used in our experiments. Including 102 climbing
stairs, 96 drinking water, 101 getting up bed, 100 pouring
water, 96 sitting down, 95 standing up and 99 walking.

We encode 3-axis signals(x, y, z) as 3-channel images using a recurrence
plot (RP) and train a tiny neural network to do image
classification.

we provide a method that encodes x−, y− and z−axis of signals to red, green and blue channel of images.

We resort RP to encoding 3-axis signals as RGB channels of
images so that their correlation information can be exploited.