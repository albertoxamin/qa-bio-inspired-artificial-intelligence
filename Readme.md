# Check Points Bio-Inspired Artificial Intelligence 

Dr Giovanni Iacca giovanni.iacca@unitn.it 

October 29, 2018 

This is a series of check points (in the form of questions) that you can use to check if you understood the main points of each lecture. If you are not able to answer some of these questions, first check the slides and lecture notes. If you still do not understand the question, send an e-mail to giovanni.iacca@unitn.it. 

## 1 Introduction

* What is a tentative definition of Artificial Intelligence?
* 
* What are the classic and modern paradigms of Artificial Intelligence?
* What are the main features of Computational Intelligence algorithms?
* What are the main approaches for global and local optimization?
* What are the main difficulties that can be met in optimization problems?
* What is a metaheuristic?
* What is the main idea of the No Free Lunch Theorem?
* What are the 4 pillars of evolution?
* What is phenotype and genotype?
* What’s the difference between Darwinian and Lamarckian evolution?
* What is the fitness landscape?
* What are the similarities and differences between natural and Artificial Evolution?
* What are the key elements of Evolutionary Algorithms?
* Why and how do Evolutionary Algorithms work?
* What are the main principles of Swarm Intelligence?
* Can you mention some natural examples of emergent behavior?
* What are the main principles and challenges of (computational) Swarm Intelligence?* What are Reynolds’ flocking rules? 

## 2 Evolutionary Algorithms I 

* What are the main steps of an EA generation cycle?
* What is a discrete representation?
* How can you represent a real value with a binary representation?
* How can you represent a sequence with a discrete representation?
* What are real-valued representations and when may be used?
* What are tree-based representations?
* What are the main methods to create the initial population in EAs?* How does fitness-proportionate selection work?
* When does fitness-proportionate selection fail and why?
* How does rank-based selection work?
* How does truncated rank-based selection work?
* How does tournament selection work?
* What is elitism?
* How does crossover work for different representations?
* How does mutation works for different representations?
* How can you monitor the performance of an evolutionary algorithm?* Why is it important to monitor diversity?
* What’s the main idea of the schemata theory? 

## 3 Evolutionary Algorithms II 

* What is the advantage of using an adaptive mutation step-size in Evolution Strategies?* What are the three self-adaptive mutation strategies used in Evolution Strategies?
* What is it useful to use correlated mutations in Evolution Strategies?
* How can the pairwise dependency between n variables be represented? 

* What’s the difference between (μ, λ)-ES and (μ + λ)-ES?
* What are the main advantages of CMA-ES?
* What are the deterministic and stochastic selection schemes in Evolutionary Programming?* What are the main operators of Differential Evolution?
* What are the main parameters of Differential Evolution? 

* What’s the difference between exponential and binomial crossover used in DE?* Why and how does Differential Evolution?
* What are the differences between classic Evolutionary Algorithms and EDAs?* How does PBIL work? 

## 4 Multi-Objective Evolutionary Algorithms 

* Can you mention some examples of Multi-Objective Optimization problems?
* What are some of the drawbacks of combining criteria into a single fitness function?
* How does lexicographic ordering work?
* How does the ε-constraint method work?
* What does it mean for one solution to Pareto-dominate another solution?
* What is the Pareto-optimal front?
* What’s the difference between local and global Pareto front?
* Why is it useful to find a Pareto-optimal front?
* What’s the difference between a priori and a posteriori Multi-Objective Optimization?* What are the important aspects to take into account for Multi-Objective EAs?
* How does VEGA work?
* What is the “Pareto rank” that is used in NSGA-II, and how is it used?
* What is the “crowding distance” that is computed in NSGA-II, and how is used? 

## 5 Constrained Evolutionary Algorithms 

* What kind of constraints can an optimization problem have?
* What does it mean for a solution to be infeasible?
* What are the main kinds of penalty functions that can be used in EAs?
* What are the main issues of penalty-based approaches?
* How does ASCHEA work?
* How does stochastic ranking work?
* How does constraint domination (Pareto ranking) work?
* How can the notion of Pareto-dominance be adapted to problems with constraints?* What’s the rationale behind repair mechanisms?
* How can repaired solutions be used?
* What’s the main idea of the Ensemble of Constraint-Handling Techniques?
* What’s the main idea of Viability Evolution? 

## 6 Swarm Intelligence I 

* What is the biological inspiration of Particle Swarm Optimization?
* What are the three strategies adopted by birds in a flock?
* How can a “particle” be defined and what kind of information does it have?
* What are the main neighborhood levels and social structures that can be used in PSO?* What’s the idea behind growing neighborhoods?
* What are the main steps of (standard) Particle Swarm Optimization?
* What are the velocity component and velocity models in PSO?
* What is the effect of the acceleration coefficient parametrization?
* What are the similarities and differences between EAs and PSO?
* Can you mention some examples of dynamic inertia rules?
* What is velocity clamping and why is it needed?
* What is the main idea of Multi-Start PSO?
* What’s the main idea of Comprehensive Learning Particle Swarm Optimization?
* What’s the main idea of Cooperatively Coevolving Particle Swarms?
* Can you mention some hybrid variants of PSO? 

## 7 Swarm Intelligence II 

* What is stigmergy?
* How is stigmergy used by ants?
* What are the main principles of Ant Colony Optimization?
* What kind of problems can Ant Colony Optimization solve? Why?
* How does the transition rule used in ACO work?
* What’s the influence of the transition rule’s parameters α and β?
* How does the classic pheromone update rule used in ACO work?
* How do the local/global pheromone update rules used in ACO work?
* How can you apply Ant Colony Optimization for shortest-path problems and TSP?* Can you mention some of the main variants of ACO?
* What are the main steps of Continuous Ant Colony Optimization?
* What are the two main methods for Multi-Objective Ant Colony Optimization?
* How does the multi-pheromone approach work?
* How does the multi-colony approach work? 

## 8 Neuro-evolution 

* What are the advantages of nervous systems?
* What are the two main neuron communication models?
* What is firing time and firing rate?
* What are main elements of the McCulloch-Pitts model?
* Can you mention some types of activation functions?
* What is the difference between Feed-Forward and Recurrent Neural Networks?
* Why is linear separability important and how is it handled by Neural Networks?
* What is the difference between Single and Multi-Layer Perceptron?
* What are the main areas of applications of Artificial Neural Networks?
* What is learning in an artificial neural network?
* What is supervised learning?
* What is unsupervised learning?
* What is reinforcement learning?
* What is an error function?
* How is possible to prevent over-fitting?
* How does the Back-Propagation algorithm work?
* What are the main advantages of Neuro-evolution?
* How can genotype encode in Neuro-evolution?
* What are some problems with conventional Neuro-evolution?
* How does the Enforced Sub-Population (ESP) Algorithm work?
* How does ESP overcome some of the problems with conventional Neuro-evolution?* What is CoSyNE and how does it differ from ESP?
* What are TWEANNs?
* What are the main challenges when creating a TWEANN algorithm?
* How are genomes initialized in NEAT? Why?
* How are networks encoded in NEAT?
* What is the purpose of historical markings in NEAT?
* How does crossover work in NEAT?
* How does speciation work in NEAT and why is it used?
* What are CPPNs?
* What is HyperNEAT? How does it differ from NEAT?
* What’s the main idea of Compressed Network Search? 

## 9 Swarm and Evolutionary Robotics 

* What are the goals of Swarm Robotics?
* Can you mention some examples of tasks solved by Swarm Intelligence in nature?
* Can you describe some examples of Swarm Robotics applications?
* What are reconfigurable robots and what are their main kinds?
* What are the motivations behind Evolutionary Robotics?
* How is it possible to design a fitness function for evolving collision-free navigation?
* How is it possible to design a fitness function for evolving homing for battery charge?* How is it possible to design a fitness function for evolving autonomous car driving?
* Can you think of other examples of Evolutionary Robotics applications? 

## 10 Competitive and Cooperative Co-Evolution 

* What is competitive co-evolution?
* What is the difference between formal and computational models of competitive co-evolution?* How can you apply competitive co-evolution to evolve sorting algorithms?
* What is the recycling problem and how can it be limited?
* What is the problem of dynamic fitness landscape?
* Does competitive co-evolution lead to progress?
* How can you measure co-evolutionary progress with CIAO graphs?
* How can you measure co-evolutionary progress with Master Tournaments?
* What is the Hall of Fame and why is it useful?
* How is it possible to evolve a robotic prey-predator scenario?
* What’s the difference between inter-species and intra-species cooperation?
* What are the two main kinds of cooperation?
* Can you mention some examples of cooperation observed in nature?
* Why is it difficult to explain the evolution of altruistic cooperation?
* What is the main idea of kin selection? What is genetic relatedness?
* What is group selection and how does it differ from kin selection?
* What kind of computational models (team composition and selection) can be used?
* What are some possible applications of cooperative co-evolution?
* Can you describe the robotic foraging task experiments and their main results?
* What is the best evolutionary algorithm for evolving cooperative control systems?
* How can we use artificial competitive/co-evolutionary systems to understand biology? 

## 11 Genetic Programming 

* What are the main applications of Genetic Programming?
* What are the main strength and weakness of Genetic Programming?
* Why does GP use trees for encoding solutions?
* How can you encode a Boolean expression with a tree?
* How can you encode an arithmetic expression with a tree?
* How can you encode a computer program with a tree?
* What kind of fitness functions can be used in Genetic Programming?
* How do the full, grow and ramped half-and-half initialization methods work?* How does parent ad survivor selection work in Genetic Programming?
* How is it possible to apply crossover on trees?
* What kinds of mutation can be applied on trees?
* What is bloat in Genetic Programming?
* What are the main countermeasures to prevent bloat? 

## 12 Applications & Recent Trends 

* Can you describe some recent examples of applications of EAs?
* How can you use EAs to find bugs or validate systems, rather than optimizing them?* What’s the difference between parameter tuning and parameter control?
* How can the different parameter setting strategies used in EAs be categorized?
* What are Memetic Algorithms and how do they work?
* Why is diversity important in evolution?
* What are the main implicit and explicit approaches for diversity preservation?
* How do island models work?
* How does the diffusion model work?
* How is it possible to implement speciation in EAs?
* What’s the main idea behind fitness sharing and crowding distance?
* What is Interactive Evolutionary Computation and how does it work?
* What is the main motivation behind fitness-free Evolutionary Algorithms?
* How does Novelty Search work?
* How does MAP-Elites work?
* How is Quality Diversity computed? 