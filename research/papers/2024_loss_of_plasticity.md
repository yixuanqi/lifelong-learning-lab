# Dohare et al., Nature (2024): Loss of plasticity in deep continual learning

DOI: https://doi.org/10.1038/s41586-024-07711-7

**Forgetting**: old task performance degrades. **Plasticity loss**: ability to learn new tasks degrades. Test these separately.

The paper investigates continual ImageNet, incremental CIFAR-100, reinforcement learning and Online Permuted MNIST (up to 800 tasks with online single-pass training). Internal diagnostics include dormant units, representation effective rank and weight magnitudes.

**Continual Backprop** selectively reinitializes low-utility mature hidden units to inject new representational diversity, while minimizing immediate disruption through zeroing outgoing weights.

Lab 03 starts with 30/100 tasks and may use mini-batches; therefore it is a *scaled-down adaptation*, not a literal reproduction. Compare identical tasks, step budgets, and initial learning curves with matched scratch baselines. A correlation between dormant units and lower accuracy does not establish causation.
