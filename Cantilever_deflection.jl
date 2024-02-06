using NeuralPDE, Lux, ModelingToolkit
using Optimization, OptimizationOptimJL, OptimizationOptimisers
import ModelingToolkit: Interval, infimum, supremum

@parameters x
@variables u(..)
P=2000
Modulus=70e9
Inertia=8.3e-10

Dxx = Differential(x)^2
Dx = Differential(x)
# ODE
eq = Dxx(u(x)) ~ (-(P * x)/(Modulus*Inertia))

# Initial and boundary conditions
bcs = [u(0.0) ~ 0.0,
    Dx(u(0.0)) ~ 0.0]

# Space and time domains
domains = [x ∈ Interval(0.0, 1.0)]

# Neural network
chain = Lux.Chain(Dense(1, 20, Lux.σ), Dense(20, 20,Lux.σ),Dense(20,1))

discretization = PhysicsInformedNN(chain, QuasiRandomTraining(20))
@named pde_system = PDESystem(eq, bcs, domains, [x], [u(x)])
prob = discretize(pde_system, discretization)

callback = function (p, l)
    println("Current loss is: $l")
    return false
end

res = Optimization.solve(prob, ADAM(0.01); callback = callback, maxiters = 2000)
phi = discretization.phi


using Plots
dx = 0.05
xs = [infimum(d.domain):(dx / 10):supremum(d.domain) for d in domains][1]
u_predict = [first(phi(x, res.u)) for x in xs]

x_plot = collect(xs)
plot!(x_plot, u_predict, title = "predict")
