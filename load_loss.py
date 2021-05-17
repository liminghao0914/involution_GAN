import os
import json
import matplotlib.pyplot as plt
import numpy as np

subpath = 'celeb_igan_3.0'
file = os.path.join("..", "Baseline GAN", "logs", subpath, 'lossv1.log')
with open(file, 'r') as f:
    s = f.readline()
s = s.strip()

log = json.loads(s)

# print(len(log['lossG']))
# print(len(log['lossD']))
x_g = np.arange(1, len(log['lossg'][:5000]) * 10, 10)
x_d = np.arange(1, len(log['lossd'][:5000]) * 10, 10)
x_gp = np.arange(1, len(log['gp'][:5000]) * 10, 10)
# lossg, = plt.plot(x_g, np.array(log['lossg']), color="red")
# lossd, = plt.plot(x_d, np.array(log['lossd']), color="blue")
# gp, = plt.plot(x_gp, np.array(log['gp']), color="green")

plt.axes(yscale="log")
loss = plt.plot(x_d, -np.array(log['lossd'][:5000]) - np.array(log['gp'][:5000]))
# lossg,=plt.plot(np.array(log['lossg']), color="red")
# lossd,=plt.plot(np.array(log['lossd']), color="blue")
# gp,=plt.plot(np.array(log['gp']), color="yellow")
# for i in range(30, 1000):
#     if log['lossD'][i] > -5:
#         log['lossD'][i] = 0.6*log['lossD'][i]+0.4*log['lossD'][i-1]
plt.xlabel('Training Steps')
plt.ylabel('EMD Loss')
# plt.legend(['loss G, loss D, penalty'])
# plt.legend(labels=['lossg', 'lossd', 'gp'])
# plt.ylim([-200000, -10])

# plt.plot(np.array(log['lossD'])[150:], color='r')
# plt.plot(log['lossG'], color='g')
# plt.yscale('symlog')
pic = os.path.join("..", "Baseline GAN", "logs", subpath, subpath + '.png')
plt.savefig(pic)
plt.show()
