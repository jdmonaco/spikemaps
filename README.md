> **Note: `spikemaps` has not yet been refactored as an independent package. You should have no expectation that it will run or import correctly in its current state.**
>
> **I hope to ameliorate this as soon as I have time; of course, PRs welcome.**

<p>&nbsp;</p>

# spikemaps

The `spikemaps` package supports the creation of adaptive kernel-based maps from neurobehavioral datasets containing spikes and (`x,y`)-position trajectories within a 2D environment (e.g., average firing-rate maps for place-cell recordings with head-position tracking). 

## Origin

This code was used to generate all of the spatial map images presented in this paper:

* Monaco JD, De Guzman RM, Blair HT, and Zhang K. (2019). [Spatial synchronization codes from coupled rate-phase neurons](https://dx.doi.org/10.1371/journal.pcbi.1006741). *PLOS Computational Biology*, **15**(1), e1006741. doi:&nbsp;[10.1371/journal.pcbi.1006741](https://dx.doi.org/10.1371/journal.pcbi.1006741)

The complete code archive for the paper is available on figshare (doi:&nbsp;[10.6084/m9.figshare.6072317.v1](https://doi.org/10.6084/m9.figshare.6072317.v1)) and the dataset is archived on OSF (doi:&nbsp;[10.17605/osf.io/psbcw](https://doi.org/10.17605/osf.io/psbcw)). The `spikemaps` package is based on the `spc.tools` subpackage in that code archive. 

## Dependencies

*Note: This section will be updated as the packaging and dependencies are fixed.*

The nearest-neighbor modeling depends on `scikit-learn` algorithms, which can be installed into an Anaconda environment as:

```bash
conda install scikit-learn
```

Similarly, for `numpy`, `matplotlib`, and `pillow`. 

## Todo

- [ ] Fix dependencies for other packages of mine (e.g., remove or add as submodules)
- [ ] Update the `setup.py` to ensure correct installation, etc.
- [ ] Improve function and class APIs to enhance usabililty and convenience
- [ ] Code style and formatting consistency (e.g., `flake8` validation)
