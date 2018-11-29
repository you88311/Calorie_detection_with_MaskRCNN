volume = [160,
280,
210,
260,
150,
200,
220,
270,
220,
260,
250,
200,
190,
160,
130

]
weight=  [220.5,
197.9,
202.5,
218,
138.9,
179.3,
184.8,
232.5,
209.5,
234,
222,
177.3,
162.3,
167.5,
179.7
]
density = 0.0
for i in range(len(volume)):
    density += float(weight[i]/volume[i])
density /= float(len(volume))
print(density)
