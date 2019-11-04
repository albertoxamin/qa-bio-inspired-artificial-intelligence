# Check Points Bio-Inspired Artificial Intelligence 

This is a series of check points (in the form of questions) that you can use to check if you understood the main points of each lecture.

## 1 Introduction

* What is a tentative definition of Artificial Intelligence?

  > The ability to perform a choice from a set of options in order to achieve a certain objective

* What are the classic and modern paradigms of Artificial Intelligence?

  > The goal was to achieve human cognition, now the goal is to possibly go beyond human capabilities 

* What are the main features of Computational Intelligence algorithms?

  > Learn or adapt to new situations, generalise, abstract, discover, associate

* What are the main approaches for global and local optimization?

  >For **local** optimization we have gradient based methods (Gradient descent, Newton methods, Quasi-Newton methods) and heuristic methods (Rosenbrock, Nelder-Mead, Hooke-Jeeves, etc.)
  >For **global** optimization we can have unimodal or multimodal functions and we have 2 approaches: Deterministic and Stochastic

* What are the main difficulties that can be met in optimization problems?

  > High non-linearities, high multimodality, noisy object function, approximated object function

* What is a metaheuristic?

  > A metaheuristic is an algorithm that doesn't require any assumption on the objective function. In short any stochastic optimization algorith.

* What is the main idea of the No Free Lunch Theorem?

  > Every problem shoud be solved with a proper algorithm that is tailored arounds its features. 

* What are the 4 pillars of evolution?

  > **Population**, **Diversity** (through mutation), **Heredity**, **Selection**

* What is phenotype and genotype?

  > **Genotype** is the genetic material of the organism, is transmitted during reproduction and the mutation and the crossover affect it.
  > **Phenotype** is the manifestation of an organism, affected by the enviroment, development, selection operates on it.

* What’s the difference between Darwinian and Lamarckian evolution?

  > **Lamark**: all acquired characteristics are transmitted to the offspring
  > **Darwin**: only useful variations are transferred, survival of the fittest

* What is the fitness landscape?

  > In evolutionary biology, fitness landscapes or adaptive landscapes are used to visualize the relationship between genotypes and reproductive success. It is assumed that every genotype has a well-defined replication rate. This fitness is the "height" of the landscape. Genotypes which are similar are said to be "close" to each other, while those that are very different are "far" from each other. The set of all possible genotypes, their degree of similarity, and their related fitness values is then called a fitness landscape.

* What are the similarities and differences between natural and Artificial Evolution?

  > **Similarities:**
  >
  > * *Individual*: a candidate solution for a given problem: phenotype+genotype
  > * *Population*: a set of individuals
  > * *Diversity*: A measure of how individuals in a population differ
  > * *Selection*: A mechanism to select which individuals survive and reproduce
  > * *Inheritance*: A mechanism to partly transmit the properties of a solution to another individual.
  >
  > **Differences:**
  >
  > * Fitness is a measure of how good a solution is, **not** the number of offspring
  > * Selection is based on fitness, **not** on competition and interactions
  > * Generations are not overlapping, parents and offspring **don't** exist at the same time
  > * We expect an improvement between the initial and final solution, **natural evolution is not an optimization process**

* What are the key elements of Evolutionary Algorithms?

  > There are two key elements in EAs: An **individual** that encodes a potential solution for a given problem and at each individual is assigned a **fitness** that is a metric of how good that solution is for a specific problem.

* Why and how do Evolutionary Algorithms work?

  > Classical algorithms work on a single solution at the time, aren't parallelizable and aren't good for exploring the fitness landscape. But EAs are highly parallelisable, are good at exploitation and exploration, every solution can be perturbed differently and solutions can interact with crossover.

* What are the main principles of Swarm Intelligence?

  > Swarm intelligence is a the propery of a system where the collective behaviour of multiple agents interacting causes an emergent behaviour.
  >
  > * The agents perceive and act based only on local information
  > * Agents cooperate by means of local information
  > * Information propagates through the entire swarm
  > * The result is distributed problem solving

* Can you mention some natural examples of emergent behavior?

  >Without leader: Termite nests, with learder: V flock formation

* What are the main principles and challenges of (computational) Swarm Intelligence?

  >Principles:
  >
  >* Unity is strength
  >* Resilience, individuals are disposable 
  >* Locality: individuals have simple abilities and the only have theri local sensory information, they also perform simple actions
  >
  >Challenges:
  >
  >* Find individual behavioural rules that result in the desired swarm behaviour by reverse engineering. (They can be obtained automatically by EAs)
  >* Make sure the emergent behaviour is stable.

* What are Reynolds’ flocking (BOIDS) rules? 

  > 1. Separation
  > 2. Cohesion
  > 3. Alignment 

## 2 Evolutionary Algorithms I 

* What are the main steps of an EA generation cycle?

  >1. Find a genetic representation 
  >2. Build a population
  >3. Design a fitness function
  >4. Choose selection method
  >5. Choose replacement method
  >6. Choose crossover and mutation
  >7. Choose data analysis method
  >8. Repeat until:
  >   * a maxium fitness value is found
  >   * a solution is good enough
  >   * a time limit
  >   * a certain convergence condition is met

* What is a discrete representation?

  > A discrete representation is our way of storing the genotype which can be mapped into different phenotypes depending on the optimization problem

* How can you represent a real value with a binary representation?

  > With the [IEE754](https://en.wikipedia.org/wiki/IEEE_754) or others, it depends on the precision we want

* How can you represent a sequence with a discrete representation?

  > By using the gene position to identify an object of the sequence and the value as value

* What are real-valued representations and when may be used?

  > The genotype is sequences of real values that represent the problem parameters.
  > Used when high precision parameter optimization is required 

* What are tree-based representations?

  > The genotype describes a tree with branching points and terminals.

* What are the main methods to create the initial population in EAs?

  > First we should have a sufficiently large population to cover the search space, but also small for evaluation.
  > Then we create each individual from an uniform sample of the search space.

* How does fitness-proportionate selection work?

  > The probability that an individual makes an offspring is proportional to how good his fitness is wrt the population fitness. This biases the selection towards the most fit indivials. 

* When does fitness-proportionate selection fail and why?

  > The fitness must be positive, with uniform fitness values it becomes random selection, with few high fitness individuals the low fitness ones have almost zero chance of reproducing.

* How does rank-based selection work?

  > Individuals are sorted on their fitness value. The index on the list is the rank. Selection pressure wrt fitness proportionate selection `p(i)=1-r(i)/∑r`

* How does truncated rank-based selection work?

  >Only the best **x** individuals are allowed to make offsprings and each of them makes the same number of offsprings: **N/x** where N is the population size.  

* How does tournament selection work?

  > For every offspring to be generated:
  >
  > 1. Pick randomly k individuals from the population, where k is the tournament size (< N)
  > 2. Choose the individual with the highest fitness and make a copy
  > 3. Put all individuals back in the population

* What is elitism?

  > Mantain the best *n* individuals from previous generation to prevent loss of best individuals by effects of mutation or sub-optimal fitness evaluation

* How does crossover work for different representations?

  > * One point: decide an index where to spli the genome
  > * Unimform: for each value randomly choose on of the two parents
  > * Arithmetic: arithmetic average of the two genomes
  > * Sequences: create a new sequence with some elements in the same order of the parent
  > * Trees: Cut and paste

* How does mutation works for different representations?

  > * Binary: random mutate some bits
  > * Real valued: increase or decrease some elements
  > * Sequence: randomly swap elements
  > * Trees: randomly change

* How can you monitor the performance of an evolutionary algorithm?

  > * By tracking the bes/worst population average fitness of each generation
  > * Multiple runs are necessary 
  > * Fitness graphs are meaningful only if the fitness function doesn't change over time, these plots can be used to detect if the algorithm stagnated or coverged.

* Why is it important to monitor diversity?

  > Diversity tells where there is potential for further evolution.

* What’s the main idea of the schemata theory?

  > **DEF**: a schema is a set of individuals with some common genes.
  >
  > **DEF**: The order of a schema is the number of common genes
  >
  > **DEF**: The defining length of a schema is the longest between two defined positions, including the initial one
  >
  > **Building Block Hypothesis**: a GA seeks near optimal performance through the closeness of short, low order, high performance schemata called the building blocks 

## 3 Evolutionary Algorithms II

* What is the advantage of using an adaptive mutation step-size in Evolution Strategies?

  > It allows us to have a more explorative behaviour at the beginning in order to ignore local optimums, and to have a more exploitative behaviour at the end.
* What are the three self-adaptive mutation strategies used in Evolution Strategies?

  > * Uncorrelate mutations with one mutation step size `(x1,…,xn,σ)`
  > * Uncorrelated mutations with multiple step size `(x1,…,xn,σ1,…,σn)`
  > * Correlated mutations individuals are represented as `(x1,…,xn,σ1,…,σn,ɑ1,…,ɑn)`
* What is it useful to use correlated mutations in Evolution Strategies?
* How can the pairwise dependency between n variables be represented? 
* What’s the difference between (μ, λ)-ES and (μ + λ)-ES?
* What are the main advantages of CMA-ES?
* What are the deterministic and stochastic selection schemes in Evolutionary Programming?
* What are the main operators of Differential Evolution?
* What are the main parameters of Differential Evolution? 
* What’s the difference between exponential and binomial crossover used in DE?
* Why and how does Differential Evolution?
* What are the differences between classic Evolutionary Algorithms and EDAs?
* How does PBIL work? 

## 4 Multi-Objective Evolutionary Algorithms 

* Can you mention some examples of Multi-Objective Optimization problems?

  > Buying a car: comfort vs price
  > Engineering design: lighness vs strength
  
* What are some of the drawbacks of combining criteria into a single fitness function?

  > To find different tradeoff solutions we must re-optimise the weights.
  
* How does lexicographic ordering work?

  > The objectives are ranked in a user-defined order of importance, so we optimise each objective, restricting every time.

* How does the ε-constraint method work?

  > There are k single objective optimization problems that are solved separately, imposing for every problem k-1contraints that correspond to the other problems.

* What does it mean for one solution to Pareto-dominate another solution?

  > A solution *i* is said to "Pareto-dominate" a solution *j* if *i* is **not worse** than *j* on every objectives, **AND** if *i* is **better** than *j* **on at least one objective**.

* What is the Pareto-optimal front?

  > A pareto-optimal front is **a set** that contains **all** the **solutions that are not dominated** by any other.

* What’s the difference between local and global Pareto front?

  > The global pareto front is the pareto-optimal front in all the fitness function, a local pareto front is the optimal in a subset of the fitness function.

* Why is it useful to find a Pareto-optimal front?

  > When the decision about a problem are made after the optimization.

* What’s the difference between a priori and a posteriori Multi-Objective Optimization?

  > In "a priori" methods the decisions about a problem are before the optimisation, whereas in "a posteriori" methods the decisions are made after the optimisation.

* What are the important aspects to take into account for Multi-Objective EAs?

  > * Fitness assignments, by means of:
  >   * Weighted sum, the weights are adapted during evolution
  >   * Separate evaluation of each objective by a subset of the population
  >   * Pareto-dominance
  > * Diversity preservation
  >   * Fitness sharing
  >   * Crowding distance sorting
  > * Memory of all the non-dominated points 
  >   * an archive of non-dominated points (to expand the current population)
  >   * elitist selection

* How does VEGA work?

  > First it evaluates each solution for every objective function separately, then for each objective selects the best N/k individuals and puts the bests into a single final population

* What is the “Pareto rank” that is used in NSGA-II, and how is it used?

  > Solutions can be ranked according to their level of dominance: non dominated solutions have rank 1, then we get rank 2... recursively

* What is the “crowding distance” that is computed in NSGA-II, and how is used? 

  > It is used as a diversity preservation mechanism

## 5 Constrained Evolutionary Algorithms 

* What kind of constraints can an optimization problem have?
* What does it mean for a solution to be infeasible?
* What are the main kinds of penalty functions that can be used in EAs?
* What are the main issues of penalty-based approaches?
* How does ASCHEA work?
* How does stochastic ranking work?
* How does constraint domination (Pareto ranking) work?
* How can the notion of Pareto-dominance be adapted to problems with constraints?
* What’s the rationale behind repair mechanisms?
* How can repaired solutions be used?
* What’s the main idea of the Ensemble of Constraint-Handling Techniques?
* What’s the main idea of Viability Evolution? 

## 6 Swarm Intelligence I 

* What is the biological inspiration of Particle Swarm Optimization?
* What are the three strategies adopted by birds in a flock?
* How can a “particle” be defined and what kind of information does it have?
* What are the main neighborhood levels and social structures that can be used in PSO?
* What’s the idea behind growing neighborhoods?
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
* How can you apply Ant Colony Optimization for shortest-path problems and TSP?
* Can you mention some of the main variants of ACO?
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
* How does ESP overcome some of the problems with conventional Neuro-evolution?
* What is CoSyNE and how does it differ from ESP?
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
* How is it possible to design a fitness function for evolving homing for battery charge?
* How is it possible to design a fitness function for evolving autonomous car driving?
* Can you think of other examples of Evolutionary Robotics applications? 

## 10 Competitive and Cooperative Co-Evolution 

* What is competitive co-evolution?
* What is the difference between formal and computational models of competitive co-evolution?
* How can you apply competitive co-evolution to evolve sorting algorithms?
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
* How do the full, grow and ramped half-and-half initialization methods work?
* How does parent ad survivor selection work in Genetic Programming?
* How is it possible to apply crossover on trees?
* What kinds of mutation can be applied on trees?
* What is bloat in Genetic Programming?
* What are the main countermeasures to prevent bloat? 

## 12 Applications & Recent Trends 

* Can you describe some recent examples of applications of EAs?
* How can you use EAs to find bugs or validate systems, rather than optimizing them?
* What’s the difference between parameter tuning and parameter control?
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