Objective: show that the boundaries created by the NLM does not allow to have a good predictive uncertainty in terms of separating OOD from in-distribution points.
- First step: Neural Linear Model Classifiers are unable to model
OOD uncertainty.
Experiment: use NLM, BNN (with BBVI) and show that they are unable to capture the predictive uncertainty far from the data. How ? Compute the epistemic uncertainty far from the data and show that it is irrelevant (with the decision boundaries)
- Second step: the training of the NLM are unable to predict OOD points or not. 
Train a NLM the usual way. Then, use the feature map produced in order to predict b or not b. Study the AUC metric. 
Compare that to when we try to model the joint distribution y,b|x.
