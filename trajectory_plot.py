from random import choice,randint
import matplotlib.pyplot as plt
import numpy as np
#np.set_printoptions(threshold=np.inf)

x_values = np.load('trajectory_result/xs.npy', allow_pickle=True)
y_values = np.load('trajectory_result/ys.npy', allow_pickle=True)
all_obs_xvalues = np.load('trajectory_result/obs_xs.npy', allow_pickle=True)
all_obs_yvalues = np.load('trajectory_result/obs_ys.npy', allow_pickle=True)

x_values_qp = np.load('trajectory_result/xs_qp.npy', allow_pickle=True)
y_values_qp = np.load('trajectory_result/ys_qp.npy', allow_pickle=True)
print(x_values_qp[11],y_values_qp[11])
all_obs_xvalues_qp = np.load('trajectory_result/obs_xs_qp.npy', allow_pickle=True)
all_obs_yvalues_qp = np.load('trajectory_result/obs_ys_qp.npy', allow_pickle=True)

safe_obs_xvalues = np.load('trajectory_result/safe_obs_xs.npy', allow_pickle=True)
safe_obs_yvalues = np.load('trajectory_result/safe_obs_ys.npy', allow_pickle=True)


for i in range(len(all_obs_xvalues)):
	if (all_obs_xvalues[i] > 1.0):
		all_obs_xvalues[i] = -1 + (all_obs_xvalues[i] - 1)
	elif (all_obs_xvalues[i] < -1.0):
		all_obs_xvalues[i] = 1 + (all_obs_xvalues[i] + 1)
for i in range(len(all_obs_xvalues_qp)):
	if (all_obs_xvalues_qp[i] > 1.0):
		all_obs_xvalues_qp[i] = -1 + (all_obs_xvalues_qp[i] - 1)
	elif (all_obs_xvalues_qp[i] < -1.0):
		all_obs_xvalues_qp[i] = 1 + (all_obs_xvalues_qp[i] + 1)

#绘制运动的轨迹图，且颜色由浅入深
point_numbers = np.array(range(len(x_values)))
obs_point_numbers = np.array(range(len(all_obs_xvalues)))
plt.scatter(x_values, y_values, c=point_numbers, cmap=plt.cm.Greens, edgecolors='none', s=15)
plt.scatter(all_obs_xvalues, all_obs_yvalues, c=obs_point_numbers, cmap=plt.cm.Reds, s=40)
plt.scatter(safe_obs_xvalues, safe_obs_yvalues, c='red', s=1)
#将起点和终点高亮显示，s=100代表绘制的点的大小
plt.scatter(x_values[0], y_values[0], c='green', s=100)
plt.scatter(x_values[-1], y_values[-1], c='Black', s=100)

point_numbers = np.array(range(len(x_values_qp)))
obs_point_numbers = np.array(range(len(all_obs_xvalues_qp)))
plt.scatter(x_values_qp, y_values_qp, c=point_numbers, cmap=plt.cm.Blues, edgecolors='none', s=15)
plt.scatter(all_obs_xvalues_qp, all_obs_yvalues_qp, c=obs_point_numbers, cmap=plt.cm.Reds, s=40)
#将起点和终点高亮显示，s=100代表绘制的点的大小
plt.scatter(x_values_qp[0], y_values_qp[0], c='green', s=100)
plt.scatter(x_values_qp[-1], y_values_qp[-1], c='Black', s=100)

plt.scatter(x_values[16], y_values[15], c='Black', s=70, marker='*')
plt.annotate("adapted SSA", xy=(-0.3, -0.45), xytext=(-0.1, -0.75), arrowprops=dict(arrowstyle="->"), c = 'blue')
plt.annotate("vanilla SSA", xy=(-0.75, -0.80), xytext=(-0.4, -0.85), arrowprops=dict(arrowstyle="->"), c = 'green')
plt.axhline(y=1.0, xmin=-1, xmax=1, color='g',linewidth=3., linestyle='-')
plt.axvline(x=1.0, ymin=0.06, ymax=0.95, color='g',linewidth=3., linestyle='-')
plt.axvline(x=-1.0, ymin=0.06, ymax=0.95, color='g',linewidth=3., linestyle='-')
#plt.fill_between(np.array([-1,1]), np.array([1,1]), np.array([1.05,1.05]))
# 隐藏x、y轴
plt.axes().get_xaxis().set_visible(True)
plt.axes().get_yaxis().set_visible(True)
#显示运动轨迹图
plt.show()