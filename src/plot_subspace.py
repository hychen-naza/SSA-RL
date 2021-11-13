import matplotlib.pyplot as plt 
import matplotlib
import numpy as np    
font = {'weight' : 'bold',
        'size'   : 13}
smallfont = {'weight' : 'bold',
        'size'   : 10}
matplotlib.rc('font', **font)
#[-0.04576116, 1.16521658], [-0.00499961, -0.00500713], 'red', 
L_gs = [[0.50669673, -1.1083459], [-0.98246777, -0.02553507]]
us = [[-0.00629471, -0.00216765], [0.00873846, -0.00464291]]
i=0
colors = ['black', 'blue']
plt.axes().set_xlim(-0.04, 0.04)
plt.axes().set_ylim(-0.04, 0.04)
plt.axes().get_xaxis().set_visible(False)
plt.axes().get_yaxis().set_visible(False)
for u, L_g in zip(us, L_gs):
	orthogoal_line = [-L_g[1], L_g[0]]
	slope = orthogoal_line[1]/orthogoal_line[0]
	intercept = u[1] - slope * u[0]
	axes = plt.gca()
	x_vals = np.array(axes.get_xlim())
	y_vals = intercept + slope * x_vals
	print(f"intercept {intercept}, slope {slope}")
	plt.plot(x_vals, y_vals, '--', color=colors[i])
	
	L_g_norm = np.linalg.norm(L_g)
	arrow = [0.01*L_g[0]/L_g_norm, 0.01*L_g[1]/L_g_norm]
	print(np.dot(orthogoal_line, arrow))
	#plt.annotate("", xy=(arrow[0], arrow[1]), xytext=(0, 0),arrowprops=dict(arrowstyle="->"), color=colors[i])
	plt.arrow(0,0,-arrow[0], -arrow[1], width = 0.0005, color=colors[i])
	i += 1
plt.plot(0,0,'ro',color='blue') 
plt.text(-0.01, -0.035, 'Control Space')
plt.text(0, 0.005, r'$L_g\phi$')
x = [0.04,0.04,0.007578157477451398,0.008498354159046052]
y = [0.01899666176701994,0.04,0.04,0.004595218194909942]
plt.fill(x,y,color="green",alpha=0.5)
plt.text(0.011, 0.025, "Safe Control Subspace",**smallfont)
plt.savefig("/home/naza/Desktop/safecontrol.eps", format='eps')
plt.show()
#abline(slope, intercept)