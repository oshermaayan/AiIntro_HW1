import numpy as np
from matplotlib import pyplot as plt

X = np.array([400, 450, 900, 390, 550])

T=np.linspace(0.01, 5, 100)
min = np.amin(X)

#squared=np.array(map(lambda x:)
P = np.zeros((len(T),len(X)))

for i in range(len(X)):
    x_array = np.zeros(len(T))
    for temp_ind, temp in enumerate(T):
        exp = -1/temp
        squared = np.array(list(map(lambda x: (x/min)**exp, X)))
        squared_sum = np.sum(squared)
        top = (X[i]/min)**exp
        x_array[temp_ind] = (top / squared_sum)

    P[:, i] = x_array


print(P)

for i in range(len(X)):
    plt.plot(T, P[:, i], label=str(X[i]))

plt.xlabel("T")
plt.ylabel("P")
plt.title("Probability as a function of the temperature")
plt.legend()
plt.grid()
plt.show()
exit()
