# Spin-glass transformers in JAX

Exploring connections between transformer neural networks and (non-)equilibrium statistical mechanics in JAX.

**Blog post (at some point): [Spin-Glass Transformers: A Physics-Inspired Class of Transformer Modules](https://mcbal.github.io/) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)**


## Summary
In a series of previous [blog posts](https://mcbal.github.io), we have tried to connect the forward pass of a transformer neural-network module to computing mean magnetizations in Ising-like vector-spin models with parametrized couplings and magnetic fields. In this picture, the forward pass of a transformer module computes statistical observables given the couplings and magnetic fields and the backward pass nudges the parametrized couplings and magnetic fields to better respond to the demands of the training loss. However, both the TAP-style mean-field approach of [Deep Implicit Attention: A Mean-Field Theory Perspective on Attention Mechanisms](https://mcbal.github.io/post/deep-implicit-attention-a-mean-field-theory-perspective-on-attention-mechanisms/) and the saddle-point approach of [Transformers from Spin Models: Approximate Free Energy Minimization](https://mcbal.github.io/post/transformers-from-spin-models-approximate-free-energy-minimization/) are only well-defined for models with symmetric coupling matrices, whose stochastic dynamics obey detailed balance and converge to a steady-state equilibrium characterized by the Boltzmann distribution. 

To capture models with asymmetric coupling matrices (like softmax attention in transformer modules), we need to consider non-equilibrium systems, whose non-equilibrium steady state lacks detailed
balance and is not described by a Boltzmann distribution. These conditions induce a time-reversal asymmetry in dynamical trajectories, leading to positive entropy production and energy dissipation [2]. Several self-consistent mean-field approaches have been developed for the binary kinetic Ising model based on expansions around a non-interacting ansatz [1-3]. It would be interesting to generalize these results to vector spins and compare the resulting magnetization equations to the forward pass of a transformer module. Intuitively, we expect to find an explicit expression for whatever it is the feed-forward network is trying to approximate, but without introducing additional parameters.


## References

### Non-equilibrium

[1] M. Aguilera, S.A. Moosavi, and H. Shimazaki, **A unifying framework for mean-field theories of asymmetric kinetic Ising systems.** *Nat Commun* **12**, 1197 (2021). https://arxiv.org/abs/2002.04309

[2] H.C. Nguyen, R. Zecchina, and J. Berg, **Inverse statistical problems: from the inverse Ising problem to data science**, *Advances in Physics*, 66:3, 197-261 (2017). https://arxiv.org/abs/1702.01522

[3] H.J. Kappen and J.J. Spanjers, **Mean field theory for asymmetric neural networks**, *Phys. Rev. E* **61**, 5658 (2000)


### Equilibrium
- 
