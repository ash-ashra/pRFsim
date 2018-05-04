# PyPRF
<img src="logo.png" width=200 align="right" />

A free & open source *python package* for *population receptive field (PRF) simulation*. This package is mainly developed for functional magnetic resonance imaging (fMRI) experiments.

## Installation
The `prfsim` package can directly be installed from PyPI, in the following way:

```bash
pip install prfsim
```

## Dependencies
`prfsim` is implemented in [Python 3.6.4]

If you install `pyprf` using `pip` (as described above), all of the following dependencies are installed automatically - you do not have to take care of this yourself. Simply follow the above installation instructions.

| pRFsim dependencies                                   | Tested version |
|-------------------------------------------------------|----------------|
| [NumPy](http://www.numpy.org/)                        | 1.14.0         |
| [Pandas](https://pandas.pydata.org/)                  | 0.22.0         |
| [SciPy](http://www.scipy.org/)                        | 1.0.0          |
| [Seaborn](https://seaborn.pydata.org/)   		| 0.8.1          |

## Contributions

For contributors, we suggest the following procedure:

* Create your own branch (in the web interface, or by `git checkout -b new_branch`)
    * If you create the branch in the web interface, pull changes to your local repository (`git pull`)
* Change to new branch: `git checkout new_branch`
* Make changes
* Commit changes to new branch (`git add .` and `git commit -m`)
* Push changes to new branch (`git push origin new_branch`)
* Create a pull request using the web interface

## References
This application is based on the following work:

[1] Dumoulin, S. O. & Wandell, B. A. (2008). Population receptive field estimates in human visual cortex. NeuroImage 39, 647–660.

## Support
Please use the [github issues](https://github.com/arash-ash/prfsim/issues) for questions or bug reports.

## License
The project is licensed under [GNU General Public License Version 3](http://www.gnu.org/licenses/gpl.html).
