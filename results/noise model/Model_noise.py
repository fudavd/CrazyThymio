import matplotlib.pyplot as plt
import numpy as np

heading_data = np.load("./results/noise model/log_neg_headings.npy", allow_pickle=True)
dist_data = np.load("./results/noise model/log_quad_dist.npy", allow_pickle=True)
fault_ind = ~(dist_data==2.0)
fig, ax = plt.subplots(2)
ax[0].hist(np.unwrap(heading_data[fault_ind]-np.pi))
ax[1].hist(dist_data[fault_ind])
plt.show()
# Missing value frequency 26/1339
fault_ratio = np.sum(fault_ind)/len(dist_data)
# Heading noise 0.004624489441151099
heading_std = np.unwrap(heading_data[fault_ind]-np.pi).std()
# Distance noise 0.043254121974921116
dist_std = dist_data[fault_ind].std()
