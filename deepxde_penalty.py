import torch
import deepxde as dde
import meshio
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
pc = dde.geometry.Rectangle([0,0],[2,0.41])


points=np.random.rand(500,2)
points[:,0]=points[:,0]*2
points[:,1]=points[:,1]*0.41
u_true=6*points[:,1]*(0.41-points[:,1])/(0.41)**2
v_true=0*points[:,1]
u_true=u_true.reshape(-1)
v_true=v_true.reshape(-1)
epsilon=0.01

def Navier_Stokes_Equation(x, y):
    u= y[:,0:1]
    u = y[:,0:1]
    v = y[:,1:2]
    p = y[:,2:3]
    du_x = dde.grad.jacobian(y, x, i=0, j=0)
    dv_y = dde.grad.jacobian(y, x, i=1, j=1)
    du_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    du_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)
    dv_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
    dv_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)
    continuity = du_x + dv_y
    continuity_x=dde.grad.jacobian(continuity, x, i=0, j=0)
    continuity_y=dde.grad.jacobian(continuity, x, i=0, j=1)
    x_momentum = 1/epsilon*continuity_x + (du_xx + du_yy)
    y_momentum = 1/epsilon*continuity_y + (dv_xx + dv_yy)
    return [x_momentum, y_momentum]

def transform(input, output):    
    u=output[:,0].unsqueeze(-1)
    v=output[:,1].unsqueeze(-1)
    x=input[:,0].unsqueeze(-1)
    y=input[:,1].unsqueeze(-1)
    u_new=(0.41-y)*y*x*(2-x)*u+(6*(y)*(0.41-y))/(0.41*0.41)
    v_new=((0.41-y)*y*x*v)
    return torch.concat((u_new,v_new),axis=1)

net = dde.nn.FNN(
[2] + [500] * 4 + [2], "sin", "Glorot uniform"
)


net.apply_output_transform(transform)

data = dde.data.PDE(
    pc,
    Navier_Stokes_Equation,
    [],
    num_domain=500,
    num_boundary=0)


model = dde.Model(data, net)
model.compile("adam", lr=0.0001)
losshistory, train_state = model.train(iterations=100,display_every=1)
with torch.no_grad():
    tmp=model.predict(points)
    u_pred=tmp[:,0].reshape(-1)
    v_pred=tmp[:,1].reshape(-1)
    print(1/2*np.linalg.norm(u_true-u_pred)/np.linalg.norm(u_true+1))
    print(1/2*np.linalg.norm(v_true-v_pred)/np.linalg.norm(v_true+1))
