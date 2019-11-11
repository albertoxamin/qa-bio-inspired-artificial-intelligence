# Check Points Bio-Inspired Artificial Intelligence 

This is a series of check points (in the form of questions) that you can use to check if you understood the main points of each lecture.

- [Table of Contents](#check-points-bio-inspired-artificial-intelligence)
  * [1 Introduction](#1-introduction)

  * [2 Evolutionary Algorithms I](#2-evolutionary-algorithms-i)

  * [3 Evolutionary Algorithms II](#3-evolutionary-algorithms-ii)

  * [4 Multi-Objective Evolutionary Algorithms](#4-multi-objective-evolutionary-algorithms)

  * [5 Constrained Evolutionary Algorithms](#5-constrained-evolutionary-algorithms)

  * [6 Swarm Intelligence I](#6-swarm-intelligence-i)

  * [7 Swarm Intelligence II](#7-swarm-intelligence-ii)

  * [8 Neuro-evolution](#8-neuro-evolution)

  * [9 Swarm and Evolutionary Robotics](#9-swarm-and-evolutionary-robotics)

  * [10 Competitive and Cooperative Co-Evolution](#10-competitive-and-cooperative-co-evolution)

  * [11 Genetic Programming](#11-genetic-programming)

  * [12 Applications & Recent Trends](#12-applications---recent-trends)



## 1 Introduction

#### What is a tentative definition of Artificial Intelligence?

The ability to perform a choice from a set of options in order to achieve a certain objective

#### What are the classic and modern paradigms of Artificial Intelligence?

The goal was to achieve human cognition, now the goal is to possibly go beyond human capabilities 

#### What are the main features of Computational Intelligence algorithms?

Learn or adapt to new situations, generalise, abstract, discover, associate

#### What are the main approaches for global and local optimization?

For **local** optimization we have gradient based methods (Gradient descent, Newton methods, Quasi-Newton methods) and heuristic methods (Rosenbrock, Nelder-Mead, Hooke-Jeeves, etc.)
For **global** optimization we can have unimodal or multimodal functions and we have 2 approaches: Deterministic and Stochastic

#### What are the main difficulties that can be met in optimization problems?

High non-linearities, high multimodality, noisy object function, approximated object function

#### What is a metaheuristic?

A metaheuristic is an algorithm that doesn't require any assumption on the objective function. In short any stochastic optimization algorith.

#### What is the main idea of the No Free Lunch Theorem?

Every problem shoud be solved with a proper algorithm that is tailored arounds its features. 

#### What are the 4 pillars of evolution?

**Population**, **Diversity** (through mutation), **Heredity**, **Selection**

#### What is phenotype and genotype?

**Genotype** is the genetic material of the organism, is transmitted during reproduction and the mutation and the crossover affect it.
**Phenotype** is the manifestation of an organism, affected by the enviroment, development, selection operates on it.

#### What’s the difference between Darwinian and Lamarckian evolution?

**Lamark**: all acquired characteristics are transmitted to the offspring
**Darwin**: only useful variations are transferred, survival of the fittest

#### What is the fitness landscape?

In evolutionary biology, fitness landscapes or adaptive landscapes are used to visualize the relationship between genotypes and reproductive success. It is assumed that every genotype has a well-defined replication rate. This fitness is the "height" of the landscape. Genotypes which are similar are said to be "close" to each other, while those that are very different are "far" from each other. The set of all possible genotypes, their degree of similarity, and their related fitness values is then called a fitness landscape.

#### What are the similarities and differences between natural and Artificial Evolution?

**Similarities:**

* *Individual*: a candidate solution for a given problem: phenotype+genotype
* *Population*: a set of individuals
* *Diversity*: A measure of how individuals in a population differ
* *Selection*: A mechanism to select which individuals survive and reproduce
* *Inheritance*: A mechanism to partly transmit the properties of a solution to another individual.

**Differences:**

* Fitness is a measure of how good a solution is, **not** the number of offspring
* Selection is based on fitness, **not** on competition and interactions
* Generations are not overlapping, parents and offspring **don't** exist at the same time
* We expect an improvement between the initial and final solution, **natural evolution is not an optimization process**

#### What are the key elements of Evolutionary Algorithms?

There are two key elements in EAs: An **individual** that encodes a potential solution for a given problem and at each individual is assigned a **fitness** that is a metric of how good that solution is for a specific problem.

#### Why and how do Evolutionary Algorithms work?

Classical algorithms work on a single solution at the time, aren't parallelizable and aren't good for exploring the fitness landscape. But EAs are highly parallelisable, are good at exploitation and exploration, every solution can be perturbed differently and solutions can interact with crossover.

#### What are the main principles of Swarm Intelligence?

Swarm intelligence is a the propery of a system where the collective behaviour of multiple agents interacting causes an emergent behaviour.

* The agents perceive and act based only on local information
* Agents cooperate by means of local information
* Information propagates through the entire swarm
* The result is distributed problem solving

#### Can you mention some natural examples of emergent behavior?

  Without leader: Termite nests, with learder: V flock formation

#### What are the main principles and challenges of (computational) Swarm Intelligence?

  Principles:

  * Unity is strength
  * Resilience, individuals are disposable 
  * Locality: individuals have simple abilities and the only have theri local sensory information, they also perform simple actions

  Challenges:

  * Find individual behavioural rules that result in the desired swarm behaviour by reverse engineering. (They can be obtained automatically by EAs)
  * Make sure the emergent behaviour is stable.

#### What are Reynolds’ flocking (BOIDS) rules? 

1. Separation
2. Cohesion
3. Alignment 

## 2 Evolutionary Algorithms I 

#### What are the main steps of an EA generation cycle?

  1. Find a genetic representation 
  2. Build a population
  3. Design a fitness function
  4. Choose selection method
  5. Choose replacement method
  6. Choose crossover and mutation
  7. Choose data analysis method
  8. Repeat until:
     * a maxium fitness value is found
     * a solution is good enough
     * a time limit
     * a certain convergence condition is met

#### What is a discrete representation?

A discrete representation is our way of storing the genotype which can be mapped into different phenotypes depending on the optimization problem

#### How can you represent a real value with a binary representation?

With the [IEE754](https://en.wikipedia.org/wiki/IEEE_754) or others, it depends on the precision we want

#### How can you represent a sequence with a discrete representation?

By using the gene position to identify an object of the sequence and the value as value

#### What are real-valued representations and when may be used?

The genotype is sequences of real values that represent the problem parameters.
Used when high precision parameter optimization is required 

#### What are tree-based representations?

The genotype describes a tree with branching points and terminals.

#### What are the main methods to create the initial population in EAs?

First we should have a sufficiently large population to cover the search space, but also small for evaluation.
Then we create each individual from an uniform sample of the search space.

#### How does fitness-proportionate selection work?

The probability that an individual makes an offspring is proportional to how good his fitness is wrt the population fitness. This biases the selection towards the most fit indivials. 

#### When does fitness-proportionate selection fail and why?

The fitness must be positive, with uniform fitness values it becomes random selection, with few high fitness individuals the low fitness ones have almost zero chance of reproducing.

#### How does rank-based selection work?

Individuals are sorted on their fitness value. The index on the list is the rank. Selection pressure wrt fitness proportionate selection `p(i)=1-r(i)/∑r`

#### How does truncated rank-based selection work?

  Only the best **x** individuals are allowed to make offsprings and each of them makes the same number of offsprings: **N/x** where N is the population size.  

#### How does tournament selection work?

For every offspring to be generated:

1. Pick randomly k individuals from the population, where k is the tournament size (< N)
2. Choose the individual with the highest fitness and make a copy
3. Put all individuals back in the population

#### What is elitism?

Mantain the best *n* individuals from previous generation to prevent loss of best individuals by effects of mutation or sub-optimal fitness evaluation

#### How does crossover work for different representations?

* One point: decide an index where to spli the genome
* Unimform: for each value randomly choose on of the two parents
* Arithmetic: arithmetic average of the two genomes
* Sequences: create a new sequence with some elements in the same order of the parent
* Trees: Cut and paste

#### How does mutation works for different representations?

* Binary: random mutate some bits
* Real valued: increase or decrease some elements
* Sequence: randomly swap elements
* Trees: randomly change

#### How can you monitor the performance of an evolutionary algorithm?

* By tracking the bes/worst population average fitness of each generation
* Multiple runs are necessary 
* Fitness graphs are meaningful only if the fitness function doesn't change over time, these plots can be used to detect if the algorithm stagnated or coverged.

#### Why is it important to monitor diversity?

Diversity tells where there is potential for further evolution.

#### What’s the main idea of the schemata theory?

**DEF**: a schema is a set of individuals with some common genes.

**DEF**: The order of a schema is the number of common genes

**DEF**: The defining length of a schema is the longest between two defined positions, including the initial one

**Building Block Hypothesis**: a GA seeks near optimal performance through the closeness of short, low order, high performance schemata called the building blocks 

## 3 Evolutionary Algorithms II

#### What is the advantage of using an adaptive mutation step-size in Evolution Strategies?

It allows us to have a more explorative behaviour at the beginning in order to ignore local optimums, and to have a more exploitative behaviour at the end.
#### What are the three self-adaptive mutation strategies used in Evolution Strategies?

* Uncorrelate mutations with one mutation step size `(x1,…,xn,σ)`
* Uncorrelated mutations with multiple step size `(x1,…,xn,σ1,…,σn)`
* Correlated mutations individuals are represented as `(x1,…,xn,σ1,…,σn,ɑ1,…,ɑn)`

#### Why is it useful to use correlated mutations in Evolution Strategies?



#### How can the pairwise dependency between n variables be represented?

With a covariance matrix

#### What’s the difference between (μ, λ)-ES and (μ + λ)-ES?

 

#### What are the main advantages of CMA-ES?
#### What are the deterministic and stochastic selection schemes in Evolutionary Programming?
#### What are the main operators of Differential Evolution?
#### What are the main parameters of Differential Evolution? 
#### What’s the difference between exponential and binomial crossover used in DE?
#### Why and how does Differential Evolution?
#### What are the differences between classic Evolutionary Algorithms and EDAs?
#### How does PBIL work? 

## 4 Multi-Objective Evolutionary Algorithms 

#### Can you mention some examples of Multi-Objective Optimization problems?

Buying a car: comfort vs price
Engineering design: lighness vs strength

#### What are some of the drawbacks of combining criteria into a single fitness function?

To find different tradeoff solutions we must re-optimise the weights.

#### How does lexicographic ordering work?

The objectives are ranked in a user-defined order of importance, so we optimise each objective, restricting every time.

#### How does the ε-constraint method work?

There are k single objective optimization problems that are solved separately, imposing for every problem k-1contraints that correspond to the other problems.

#### What does it mean for one solution to Pareto-dominate another solution?

A solution *i* is said to "Pareto-dominate" a solution *j* if *i* is **not worse** than *j* on every objectives, **AND** if *i* is **better** than *j* **on at least one objective**.

#### What is the Pareto-optimal front?

A pareto-optimal front is **a set** that contains **all** the **solutions that are not dominated** by any other.

#### What’s the difference between local and global Pareto front?

The global pareto front is the pareto-optimal front in all the fitness function, a local pareto front is the optimal in a subset of the fitness function.

#### Why is it useful to find a Pareto-optimal front?

When the decision about a problem are made after the optimization.

#### What’s the difference between a priori and a posteriori Multi-Objective Optimization?

In "a priori" methods the decisions about a problem are before the optimisation, whereas in "a posteriori" methods the decisions are made after the optimisation.

#### What are the important aspects to take into account for Multi-Objective EAs?

* Fitness assignments, by means of:
  * Weighted sum, the weights are adapted during evolution
  * Separate evaluation of each objective by a subset of the population
  * Pareto-dominance
* Diversity preservation
  * Fitness sharing
  * Crowding distance sorting
* Memory of all the non-dominated points 
  * an archive of non-dominated points (to expand the current population)
  * elitist selection

#### How does VEGA work?

First it evaluates each solution for every objective function separately, then for each objective selects the best N/k individuals and puts the bests into a single final population

#### What is the “Pareto rank” that is used in NSGA-II, and how is it used?

Solutions can be ranked according to their level of dominance: non dominated solutions have rank 1, then we get rank 2... recursively

#### What is the “crowding distance” that is computed in NSGA-II, and how is used? 

It is used as a diversity preservation mechanism

## 5 Constrained Evolutionary Algorithms 

#### What kind of constraints can an optimization problem have?

**Hard-constraints** which set conditions that are requred to be satisfied and **soft-constraints** which have some variables values that are penalised in the objective function.
#### What does it mean for a solution to be infeasible?

That some hard-constraints are not satisfied.
#### What are the main kinds of penalty functions that can be used in EAs?

* Death Penalty: when a solution violates a constraint it is rejected
* Static Penalty: the penalty functions remains constant during the evolution process
* Dynamic Penalty: the penalty function changes depending on the generation
* Adaptive penalty: the penalty function takes a feedback from the search process

#### What are the main issues of penalty-based approaches?

* The penalty factors are highly problem dependent.
* If the penalty is too high or low the optimization process may not be efficient

#### How does ASCHEA work?

It has 3 main components:

* An adaptive penalty function
* A recombination operator guided by the constraints, that mixes an infeasable solution with a feasible one
* A "Segregational selection operator": Aims to define the ratio of feasible solutions in the next population

#### How does stochastic ranking work?

It consists of multi-membered Evolution strategies that use a penalty function and a selection based on a ranking process. It tries to balance the influence of the objective function and the penalty function when assigning fitness to a solution. It doesn't requre the definition of a penalty factor

#### How does constraint domination (Pareto ranking) work?
#### How can the notion of Pareto-dominance be adapted to problems with constraints?
#### What’s the rationale behind repair mechanisms?

It is an approach to turn unfeasible solutoins into feasible ones.
This can be done through greedy algorithms, random or custom herustics.

#### How can repaired solutions be used?

Repaired solutions can be used for firness evaluation only or to replace the corrisponding unfeasible solutions in the population (always or with some propbability).

#### What’s the main idea of the Ensemble of Constraint-Handling Techniques?

To have several constraint handling techniques, each with its own popultaion and parameters. Each population produces its offspring and evaluates them. However, the offspring compete with all the offsprings from all the populations. This aims to automate the selection of the best constraint-handling technique for a certain problem (no free lunch theorem)

#### What’s the main idea of Viability Evolution?

Progressively shrinking the boundaries of unfeasible solutions.

## 6 Swarm Intelligence I

#### What is the biological inspiration of Particle Swarm Optimization?

A flock of birds that wants to find the area with the highest concentration of food. Birds don't know where the area is, but each bird can tell its neighbors how many insects are at its location, also each birds remenbers its own location where it found the highest concetration of food so far.

#### What are the three strategies adopted by birds in a flock?

1. Brave: keep flying in the same direction
2. Conservative: fly back towards the best position
3. Swarm: move towards the best neighbor

#### How can a “particle” be defined and what kind of information does it have?

A particle can be defined as position, velocity and performance and it has perception on the performance of neighbors, the best particle among its neighbors and its position, its best position.

#### What are the main neighborhood levels and social structures that can be used in PSO?

Global neighborhood, distance based neighborood, list based neighborhood, ring, star, von-neumann,pyramid, clusters, wheel...

#### What’s the idea behind growing neighborhoods?

Start with a ring topology with least connectivity, then grow towards a star topology. Particles are added to the neighborhood according to a scaled euclidiean distance smaller than an increading radius.

#### What are the main steps of (standard) Particle Swarm Optimization?

* Each particle is initialized with random position and velocity
* At each step, each particle updates first the velocity: $$v'=w*v+\phi_1U_1\cdot(y-x)+\phi_2U_2\cdot(z-x)$$, where **x** is the current position and **v** the current velocity. **y** and **z** are the personal and social best position. **w** is inertia $$\phi_1$$ and $$\phi_2$$ are the cognitive and social coefficients. U1 and U2 are uniform random numbers
* Then each particles updates its position: $x'=x+v'$
* In case of improvement update y and z
* Do all the above until a stopping criteria is met.

#### What are the velocity component and velocity models in PSO?

**Components:**

* Inertia velocity: the momentum
* Cognitive velocity: the nostalgia
* Social velocity: the envy

**Models:**

* Cognition only model: **Independent hill climbers**, local search by each particle
* Social-only model: the swarm is **one stochastic hill climber**

#### What is the effect of the acceleration coefficient parametrization?

ɸ1= cognitive ɸ2 = social

* ɸ1=ɸ2=0 then the particles moves only based on its inertia
* ɸ10, ɸ2=0 cognition only model
* ɸ1=0, ɸ20 social only model
* ɸ1=ɸ20 particles are attraccted towards the average of personal and social best
* ɸ1<ɸ2 better for unimodal
* ɸ1ɸ2 better for multimodal
* low ɸ1 and ɸ2 smoother trajectories
* high ɸ1 and ɸ2 higher acceleration, abrupt movemnts

#### What are the similarities and differences between EAs and PSO?

* in PSO we have the equivalent of two populations instead of one, because we have personal best solutions and the current positions
* PSO can be see as a population based algorithm where the particle position evolve over time
* The main difference is in the logic related to the selection:
  * EAs: fitness based and take into account the entire population
  * PSO: considers the solution before and after the perturbations, one to one spawning

#### Can you mention some examples of dynamic inertia rules?

#### What is velocity clamping and why is it needed?

#### What is the main idea of Multi-Start PSO?

#### What’s the main idea of Comprehensive Learning Particle Swarm Optimization?

#### What’s the main idea of Cooperatively Coevolving Particle Swarms?

#### Can you mention some hybrid variants of PSO? 

## 7 Swarm Intelligence II 

#### What is stigmergy?

Agents can communicate through changes in the enviroment.

#### How is stigmergy used by ants?

Ants leave a pheromone trail, ants follow the path with the highest pheromone concentration, in absence of pheromone random path.

#### What are the main principles of Ant Colony Optimization?

Assumes an optimization problem represented as a graph.
We build solutions at runtime.

There are two rules, the transition rule and the pheromone update rule.

When an ant steps on a node leaves pheromones that evaporate over time.

#### What kind of problems can Ant Colony Optimization solve? Why?

  Combinatorial problems over networks and any TSP problem

#### How does the transition rule used in ACO work?

It is used to encourage the choice of edges with high pheromone levels and short distances.

#### What’s the influence of the transition rule’s parameters α and β?

ɑ decides how much the ants will be looking for pheromones, β defines how important is the distance of the nodes.

#### How does the classic pheromone update rule used in ACO work?

The pheromone on an edge linearly decreases with time, but is increased by the transition of all ants on that edge.

#### How do the local/global pheromone update rules used in ACO work?

Local: The pheromone level of each edge visited by an ant is decreased by a fraction of its current level and increased by a fraction of the initial level.

Global: Whel all ants have completed their paths, the length of the shortest path is found and only the pheromone levels of the shortest path are updated.

#### How can you apply Ant Colony Optimization for shortest-path problems and TSP?

* Distribute m ants on random nodes
* Initial pheromone levels are equal for all edges and inversely proportional to the number of nodes times the estimated length of the optimal path.

#### Can you mention some of the main variants of ACO?

Ant Colony Systems (ACS), Max-Min AS (MMAS), Elitist Pheromone Updates, Fast Ant System (FANT), Antabu, AS-Rank

#### What are the main steps of Continuous Ant Colony Optimization?

CACO performs a bi-level search: local (exploit good regions) and global (explore bad regions) 

```pseudocode
Create n_r regisons and set tau_i(0)=1, i=1,...,n_r
repeat
	Evaluate fitness, f(x_i), of each region
	Sort regions in descending order of fitness
	send 90% of n_g global ants for mutation
	send 10% of n_g global ants for trail diffusion
	update pheromone and age of n_g weak regions
	do loacl search to exploit good region
until stopping condition is true
return region x_i with the best fitness as solution
```

#### What are the two main methods for Multi-Objective Ant Colony Optimization?

Multi-pheromone approach with one pheromone for each objective

Multi-colony approach with one colony for each objective

#### How does the multi-pheromone approach work?



#### How does the multi-colony approach work? 




## 8 Neuro-evolution 

#### What are the advantages of nervous systems?

Selective transimmions of signals across distant areas which results in more complex bodies

Complex adaptation that results in survival inside changing enviroments 

#### What are the two main neuron communication models?

Spiking neurons and McCulloch-Pitts 

#### What is firing time and firing rate?

Firing time is used in spiking neurons, firing rate (signal strength) is the base of mcculloch

#### What are main elements of the McCulloch-Pitts model?

* A neural network is a black-box system that communicates with an external environment through input units and output units. All other elements are called internal or hidden units. Units are usually referred to also as nodes. 

* Nodes are linked by connections

* A connection is characterized by a weight that multiplies the input

* Each node computes the weighted sum of its inputs and applies an activation function ɸ

#### Can you mention some types of activation functions?

Identity, step, sigmoid

#### What is the difference between Feed-Forward and Recurrent Neural Networks?

In Feed-Forward Neural Networks (**FFNN**) information flows one way only

* Perceptron: input and output nodes only, no hidden learning
* Multi-Layer Perceptron (MLP): when there are one ore more hidden layers

In Recurrent Neural Networs information flows both ways, "enables" the memory capability

#### Why is linear separability important and how is it handled by Neural Networks?

The separation line is defined by the synaptic weights: w1x1+w2x2-θ=0 → x2=θ/w2-w1/w2x1 

#### What is the difference between Single and Multi-Layer Perceptron?

Single layer perceptron can solve only problems whose input/output space is linearly separable.

MLP can solve problems that are not linearly separable.

#### What are the main areas of applications of Artificial Neural Networks?

Pattern recognition, content generation, regression, classification, black box control systems

Applications: Self driving cars, network efficiency, cybersecurity, amrketing, fin-tech

#### What is learning in an artificial neural network?

The netork adjusts the weights to achieve a desired output, there are 2 types of learning supervised learning, unsupervised learning and reinforment learning. 

#### What is supervised learning?

Correct outputs are known, the goal is to minimize the error, usually with *back-propagation*

#### What is unsupervised learning?

The correct outputs aren't known, the goal is to find the correlations in data or compress data or feature extraction, usually with hebbian learning

#### What is reinforcement learning?

The correct outputs aren't known the goal is to discover a mapping from states to actions

#### What is an error function?

#### How is possible to prevent over-fitting?

1. Divide the available data into:  
   * training set (for weight update) 
   * validation set (for error monitoring)

2. Stop training when the error on the validation set starts growing 

#### How does the Back-Propagation algorithm work?



#### What are the main advantages of Neuro-evolution?

Large/continuos state and output spaces are easier to handle than with classical training algorithms

#### How can genotype encode in Neuro-evolution?

1. **Weights**, when we have a pre defined network topology, fixed length genotype, replace the backpropagation entirely
2. **Topology**, variable length genotype encodes teh presence and type of neurons and their connectivity, keeps the backpropagation
3. **Topology and Weights**
4. **Learning rules**, change how the weights are updated

#### What are some problems with conventional Neuro-evolution?

* Premature convergence, diversity is lost and progress stagnates
* Large networks may have too many parameters to be optimized simultaneously 
* How do we use crossover here?

#### How does the Enforced Sub-Population (ESP) Algorithm work?

*h* sub-populations of *n* neurons, foreach fitness evalutaion one member of ech sub-pop is chosen to be in the network, the fitness of each neuron is the average across the network that it was part of

* Each hidden neuron is in a separate subpopulation 
  * One neuron from each sub-pop is active at a given time 

  * Selection/reproduction happens within a given sub-pop 

* Populations learn compatible subtasks 

  * Allows each sub-pop to specialize 

* Evolution encourages diversity automatically
  * “Good” networks require different kinds of neurons 

* Evolution discourages competing conventions
  * Cooperative co-evolution: neurons optimized for compatible roles 

#### How does ESP overcome some of the problems with conventional Neuro-evolution?

#### What is CoSyNE and how does it differ from ESP?

Extends to synapses rather than neurons

#### What are TWEANNs?

Topology and Weight Evolving Artificial Neural Nets 

#### What are the main challenges when creating a TWEANN algorithm?

* Initial population randomization
  * non functional networks
  * Unnecessarily high-dimensional search space
* competing conventions problem worsen when evolving topologies
* loss of innovative structure
  * more complex networks can't compete in the short run
  * need to protect innovation

#### How are genomes initialized in NEAT? Why?

Nodes and connections

Nodes can be either sensors, output or hidden

Connections have input, output, weight, status (enabled/disabled), innov

#### How are networks encoded in NEAT?

In the Phenotype 

#### What is the purpose of historical markings in NEAT?

Allows matching networks of different topologies, and solves the problems that rises where same genes are in different positions or different genes that have the same position.

This by marking the gene with an historical mark

#### How does crossover work in NEAT?

#### How does speciation work in NEAT and why is it used?

NEATS divides the population into species, organism are grouped by similarity

Keeps incompatible networks apart, mating only happens within a specie

Allows the optimization without having to compete with all the other networks

Fitness sharing preserves diversity

#### What are CPPNs?

COMPOSITIONAL PATTERN PRODUCING NETWORKS.

A CPPN is similar to a neural network, but acts as indirect encoding for other objects of interest.

#### What is HyperNEAT? How does it differ from NEAT?

Hyperneat uses CPPNs to encode neural networks and by compactly encoding connectivity patterns, HyperNEAT has been demonstrated to successfully evolve neural networks of several million connections 

#### What’s the main idea of Compressed Network Search?

## 9 Swarm and Evolutionary Robotics

#### What are the goals of Swarm Robotics?

#### Can you mention some examples of tasks solved by Swarm Intelligence in nature?

#### Can you describe some examples of Swarm Robotics applications?

#### What are reconfigurable robots and what are their main kinds?

#### What are the motivations behind Evolutionary Robotics?

#### How is it possible to design a fitness function for evolving collision-free navigation?

#### How is it possible to design a fitness function for evolving homing for battery charge?

#### How is it possible to design a fitness function for evolving autonomous car driving?

#### Can you think of other examples of Evolutionary Robotics applications?

## 10 Competitive and Cooperative Co-Evolution

#### What is competitive co-evolution?

#### What is the difference between formal and computational models of competitive co-evolution?

#### How can you apply competitive co-evolution to evolve sorting algorithms?

#### What is the recycling problem and how can it be limited?

#### What is the problem of dynamic fitness landscape?

#### Does competitive co-evolution lead to progress?

#### How can you measure co-evolutionary progress with CIAO graphs?

#### How can you measure co-evolutionary progress with Master Tournaments?

#### What is the Hall of Fame and why is it useful?

#### How is it possible to evolve a robotic prey-predator scenario?

#### What’s the difference between inter-species and intra-species cooperation?

#### What are the two main kinds of cooperation?

#### Can you mention some examples of cooperation observed in nature?

#### Why is it difficult to explain the evolution of altruistic cooperation?

#### What is the main idea of kin selection? What is genetic relatedness?

#### What is group selection and how does it differ from kin selection?

#### What kind of computational models (team composition and selection) can be used?

#### What are some possible applications of cooperative co-evolution?

#### Can you describe the robotic foraging task experiments and their main results?

#### What is the best evolutionary algorithm for evolving cooperative control systems?

#### How can we use artificial competitive/co-evolutionary systems to understand biology?

## 11 Genetic Programming

#### What are the main applications of Genetic Programming?

#### What are the main strength and weakness of Genetic Programming?

#### Why does GP use trees for encoding solutions?

#### How can you encode a Boolean expression with a tree?

#### How can you encode an arithmetic expression with a tree?

#### How can you encode a computer program with a tree?

#### What kind of fitness functions can be used in Genetic Programming?

#### How do the full, grow and ramped half-and-half initialization methods work?

#### How does parent ad survivor selection work in Genetic Programming?

#### How is it possible to apply crossover on trees?

#### What kinds of mutation can be applied on trees?

#### What is bloat in Genetic Programming?

#### What are the main countermeasures to prevent bloat?

## 12 Applications & Recent Trends

#### Can you describe some recent examples of applications of EAs?

#### How can you use EAs to find bugs or validate systems, rather than optimizing them?

#### What’s the difference between parameter tuning and parameter control?

#### How can the different parameter setting strategies used in EAs be categorized?

#### What are Memetic Algorithms and how do they work?

#### Why is diversity important in evolution?

#### What are the main implicit and explicit approaches for diversity preservation?

#### How do island models work?

#### How does the diffusion model work?

#### How is it possible to implement speciation in EAs?

#### What’s the main idea behind fitness sharing and crowding distance?

#### What is Interactive Evolutionary Computation and how does it work?

#### What is the main motivation behind fitness-free Evolutionary Algorithms?

#### How does Novelty Search work?

#### How does MAP-Elites work?

#### How is Quality Diversity computed? 