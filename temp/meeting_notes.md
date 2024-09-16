## End-Of Summer Meeting
ensure training set is stratafied the same as the validation set
pruning - Look at original PPNet paper
pacmap for dimensionality reduction

## 9/9
### Things to Bring up
* Implemented Pruning
    * Weight based
        * Push, optimize, prune, optimize
    * PPNet based
        * Push, prune, optimize
        * This was chosen to combat background patches
    * Should this only happen at the end, or throughout?
    * Should we learn a soft mask?
* What is our pitch? Is this classification? Is it feature extraction? How is this interesting?
* What we still need to do:
    * Run pruning testing (sorry)
    * Fix Dataset...
    * Train Genetic, Image, and Multimodal PPNets and blackboxes for each node. Compare accuracies.
    * Train using code from BIOSCAN people, compare accuracies
    * Visualizations + Investigation
### Meeting Notes
* We should prune throughout! So we can relearn the prototypes.
* Reach out to Anush?
    * Who, what, when, where? Help!!!
* Probability level, Bayesian, Species Accuracy
* Get accuracy numbers

## 9/16
### Things to bring up
* Do I need to modify my loss to enforce that this is a probability distribution?
