---
tags:
  - machine-learning
  - final-project
  - neuroscience
  - sna
aliases:
  - Connectomes
  - Connectomics
  - FlyWire
  - Drosophila Connectome
---

A **connectome** is a complete map of the neurons in a nervous system and the synapses connecting them: the brain as a network, with neurons as [[Node|nodes]] and synapses as directed, weighted [[Tie|ties]].

# Major connectomes
- ***C. elegans*** (White et al. 1986): 302 neurons, about 7,000 synapses. The first complete connectome; an early example of a [[Small World Model|small-world]] network (Watts & Strogatz 1998).
- **Drosophila larva** (Winding et al. 2023): about 3,000 neurons.
- **Adult fly hemibrain** (Janelia FlyEM, Scheffer et al. 2020): about 25,000 neurons in the central brain.
- **FlyWire whole adult fly brain** (Dorkenwald et al.; Schlegel et al., *Nature* 2024): about 139,000 neurons and ~50 million synapses, with cell types and predicted neurotransmitters (so edges can be signed excitatory/inhibitory).
- **Mouse visual cortex (MICrONS, 2025)**: a cubic millimeter with ~200,000 cells, functional recordings aligned to structure.

# Structure (in network terms)
- Directed, weighted (synapse counts), signed (excitatory vs inhibitory by neurotransmitter).
- Heavy-tailed degree distributions with hub neurons ([[Degree Distribution]]), a rich club of densely interconnected hubs, high clustering and reciprocity relative to random graphs ([[Clustering Coefficient]], [[Reciprocity]]), and modular organization by brain region ([[Community Detection]]).
- Circuits such as the fly **mushroom body** (learning and memory: Kenyon cells, output neurons, dopaminergic neurons) are well-defined subnetworks that can be extracted as candidate reservoirs.
- These are exactly the SNA measures used on social networks; the [[Social Network Analysis]] toolkit applies directly.

# Using a connectome as a model
- **As a reservoir**: use the synaptic weight matrix as the fixed recurrent weights of an echo state network and train only a readout ([[Reservoir Computing]]; conn2res toolbox).
- **Connectome-constrained training**: keep which neurons connect, but learn the weights by gradient descent (Lappalainen et al., *Nature* 2024, trained a connectome-constrained model of the fly visual system that predicted neural responses) ([[Backpropagation]]).
- **Controls**: compare against random networks matched on size, density, degree sequence and spectral radius, so that performance differences can be attributed to wiring rather than scale ([[Random Graph]]).
- Caveats: connectomes give structure, not dynamics (weights, neuromodulation, plasticity and gap junctions are partly missing); a mushroom body evolved for odor learning has no obvious reason to suit country-month conflict data, which is the hypothesis being tested.

Data access: FlyWire Codex (codex.flywire.ai), neuPrint (FlyEM hemibrain), `fafbseg` / `navis` Python packages.

Related: [[Fly Connectome Conflict Forecasting]] | [[Neural Networks]] | [[Network]]
