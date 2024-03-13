# PINN and the Penalty Method

This is a repo for studyng the stability of the PINN training following steady stokes equation

$$\begin{cases}
\Delta u(x,y)-\nabla p(x,y)=0 & (x,y)\in (0,2)\times (0,0.41)  \\
\nabla \cdot u(x,y) = 0  & (x,y)\in (0,2)\times (0,0.41)\\
u(x,0)=u(x,0.41)=0 & x \in [0,2]\\
u(0,y)=u(2,y)=6y\frac{(0.41-y)}{(0.41)^{2}} & x \in [0,2]\\
\end{cases}$$

This system has an unique solution:
$$u(x,y)=[\frac{6y(0.41-y)}{(0.41)^{2}},0]$$
$$p(x,y)=-\frac{12}{(0.41)^{2}}x$$

The neural networks are initialized as follows:
-  the x component of the velocity is initialized with glorout uniform + the true solution. As a consequence, at the start of the training it is approximately the true solution
- the y component of the velocity is initialized with glorout uniform. As a consequence, at the start of the training it is approximately the true solution
- the pressure is inizialized with glorout uniform plus a percentage (in the code $\epsilon$) of the true solution.

From the figures it can clearly be seen that the system is unstable in $\epsilon$, and that this instability does not depend from the number of epochs.
The theoretical motivations are the following:
It can be proven that the system above is equivalent to:

$$\begin{cases}
\Delta \bar{u}(x,y)-\nabla p(x,y)=-\frac{12}{(0.41)^{2}}xe_{1} & (x,y)\in (0,2)\times (0,0.41)  \\
\nabla \cdot \bar{u}(x,y) = 0  \\
\bar{u}(x,0)=\bar{u}(x,0.41)=0 & x \in [0,2]\\
\bar{u}(0,y)=\bar{u}(2,y)=0 & x \in [0,2]
\end{cases}$$

with $$\bar{u}=u-[\frac{6y(0.41-y)}{(0.41)^{2}},0]$$
It can be proven that $\bar{u}$ is the solution of
the functional minimization problem

$$\min \int(|\nabla u|^2-2 (\nabla f \cdot \nabla u)) d x d y \quad \text{subject to} div(u)=0$$
and so that the stokes equations are exactly the Lagrange Multiplier equations associated to the minimization problem. 

So, from the Lagrange Multiplier equations we know that the solution is a saddle point problem, which can be hard to reach with local optimization methods if the initial point is on a "maximization curve".

As explained in [this paper](https://www.researchgate.net/publication/357302175_A_Variational_Principle_for_Navier-Stokes_Equations_Fluid_Mechanics_as_a_Minimization_Problem) this results also extends to general unsteady incompressible navier stokes equation.

Another perspective can be given by finite element theory. It can be proven that the solution of the stoke equation above is given by the following problem

$$(u,p)=\arg\min_{v}\arg \max_{q} L(v,q)$$ 
where


$$L(v,q)= \int (v\cdot v- q \cdot div(v)-f\cdot v)dxdy $$


A (partial) solution is to use [penalty methods](https://www.jstor.org/stable/2158403), i.e to solve

$$\begin{cases}
\Delta u_{\epsilon}(x,y)+\frac{1}{\epsilon}\nabla div(u_{\epsilon}) =0 & (x,y)\in (0,2)\times (0,0.41)  \\
u(x,0)=u(x,0.41)=0 & x \in [0,2]\\
u(0,y)=u(2,y)=6y\frac{(0.41-y)}{(0.41)^{2}} & x \in [0,2]\\
\end{cases}$$
