# control-theory-presentation

Code for "Linear Algebra in Control Theory" presentation.

Code is mostly for generating plots of dynamic systems using their state-space representation and testing methods I discuss in the presentation (calculating eigenvalues)

The various `test#.py` files all approximately do the same thing (plot a spring mass system) but I gradually understood what I was doing more so the quality improves. 

`test_pole_placement.py` tests an implementation I derived for a specific case of pole placement. 

`main.py` puts all of this together to create code that's documented, decently re-usable, and (hopefully) readable. 

The final code contains code to: 
* create a system (represented by its state-space equation)
* create a simulation for that system (given its initial conditions) that updates over a small timestep given an input
* method to record data about the system over a given time
* various methods to plot data about the system