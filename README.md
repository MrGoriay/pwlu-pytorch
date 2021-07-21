# Piecewise Linear Unit
Proposed by Yucong Zhou,Zezhou Zhu,Zhao Zhong in a paper:[Learning specialized activation functions with the Piecewise Linear Unit](https://arxiv.org/abs/2104.03693), Piecewise Linear Unit is a variant of dynamic activation function, that has several good properties:
- Most of the flexibilities lie in a bounded region, which
can maximize the utilization of learnable parameters
- As a universal approximator, PWLU can closely approximate any contiguous, bounded scalar functions
- Thanks to the uniform division of intervals, PWLU is
efficient in computation, especially for inference

# WARNING
Pay attention to the _mode_ argument, you need to set it to __1__ to get a dynamic behaviour, otherwise PWLA will act as ReLU. In theory, you can always pass one, but authors got their best results with __heavily modified__ training pipeline, check the paper.

### Notes on this project
This is an implementation that I am going to use in my own projects, which means several things:
- Project will be upgrading
- **NO** extensive testing
- **Contributions and feedback are welcome**
## Roadmap
- [ ] Tested on several real datasets
- [ ] Efficiency improved
- [ ] Replicate in JAX-based libraries?
