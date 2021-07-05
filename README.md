# generalized weighted social network model

A source code repository for the generalized weighted social network model (GWSN model) and its meta-modeling and sensitivity analysis.

## building simulation codes

Clone the repository with its submodules.

```shell
git clone --recursive git@github.com:yohm/GWSN_metamodeling.git
```

MPI and OpenMP are prerequisites. To install these prerequisites on macOS, run the following.

```shell
brew install libomp
brew install openmpi
```

After installing the prerequisites, build the executables using CMake.

```shell
mkdir -t build
cd build
cmake ..
make
```

It makes two executables, `wsn.out` and `search.out`.

- wsn.out runs a single Monte Carlo simulation for GWSN model.
- search.out runs a number of simulations sampling parameter space in order to prepare training data for meta-modeling.

On supercomputer Fugaku, run the shell script as follows.

```shell
./build_fugaku.sh
```

## executing a single run of GWSN model

To run a single GWSN model, run `wsn.out` after preparing `_input.json` file in the current directory. The format of the `_input.json` file is the following.

```jsonc:_input.jsonc
{
    "net_size":5000,  // the size of the network
    "p_tri":0.2,      // the probability to close the traids during Local Attachment
    "p_r":0.001,      // the probability of forming a link during Global Attachment
    "p_nd":0.0001,    // the probability of Node Deletion
    "p_ld":0.005,     // the probability of Link Deletion
    "aging":0.0001,   // the speed of Link Aging. Larger value corresponds to a faster decay of link weight.
    "w_th":0.5,       // the threshold of removing a link
    "w_r":1.0,        // the amount of link reinforcement during Local Attachment
    "q":3,            // the number of distinct values in each feature
    "F":8,            // the length of feature vector
    "alpha":0.0,      // geographic factor
    "t_max":50000,    // maximum time step
    "_seed":1234567890  // seed of the random number generator
}
```

After you execute the command, you'll find the output file `_output.json`, that contains the network property of the generated network.

```shell
$ vi _input.json               # set parameters
$ ./wsn.out                    # execution of the simulator
$ cat _output.json             # output is stored in _output.json
```

A sample of `_output.json` is like the following:

```json:_output.json
{
  "average_degree": 45.844,
  "average_link_weight": 20.84398215897621,
  "clustering_coefficient": 0.05834940516339395,
  "link_overlap": 0.0280994751715939,
  "pcc_c_k": -0.6182153734185948,
  "pcc_k_knn": -0.003892285713515246,
  "pcc_link_overlap_weight": 0.585569014394968,
  "percolation_fc_ascending": 1.3108518749999734,
  "percolation_fc_descending": 0.9132984374999735,
  "stddev_degree": 10.81091296643507
}
```

To integrate the wsn.out with [OACIS](https://github.com/crest-cassia/oacis), register the following command as a simulation command.

```shell
~/path/to/repo/oacis/run_wsn_all.sh
```

## executing a job for preparing training data sets

To execute a search.out, set `OMP_NUM_THREADS` and run using `mpiexec` command.
The command line arguments are `N_init`, `duration(sec)`, `N_sample`, `seed`. The output is printed in standard output.

```shell
env OMP_NUM_THREADS=2 mpiexec -n 8 ../search.out 3 10 2 1234 > training_data.txt
```

To execute on supercomputer Fugaku, submit a job using `pjsub` command. Sample code is available in `job_search/` directory.

## installing python packages

We use several python packages for regression and sensitivity analyses such as keras and tensorflow.
Install these python pre-requisites using pipenv.

```shell
pipenv install        # install pre-requisites
pipenv shell          # activate the shell on which the packages are available
```

## regression

The code for the regression analysis is in `learning/` directory.

Run `multi_layer_perceptron.py` to conduct regression using multi-layer perceptron having three hidden layers.
You can set the hyper-parameters in `_input.json` file which looks like the following:

```json:_input.json
{
  "ycol": 0,
  "units1": 30,
  "units2": 10,
  "units3": 0,
  "activation": "relu",
  "lr": 0.001,
  "batch_size": 200,
  "epochs": 2000
}
```

- `ycol` specifies which column to read as the training data.
  - specify an integer ranging `[0,7]`. 0~7 respectively indicates average degree, stndard deviation of degree, average weight, assortativity coefficient, clustering coefficient, `rho_{ck}`, link overlap, `rho_{Ow}`
- `units1`, `units2`, `units3` are the number of units in hidden layers
- `activation`: functional form of the activation function. The available options are `relu`, `tanh`, `sigmoid`.
- `lr`: learning rate
- `batch_size`: batch size
- `epochs`: The number of epochs. Note that the learning may terminate earlier when the loss function for the validation set does not improve for a certain amount of steps.

After setting up `_input.json`, run the script giving the path to the training data as follows.

```shell
python multi_layer_perceptron.py training_data.txt
```

The results (the values of the loss function for the training set and the validation set) are shown in `_output.json` file like the following:

```json:_output.json
{"min_val_loss": 0.10936633497476578, "min_loss": 0.11838668584823608}
```

You'll also find `loss.png` file that shows the time evolutions of the error functions. The regression model is saved in `model.h5`.

To integrate it with [OACIS](https://github.com/crest-cassia/oacis), register the command as follows: (change the path according to your environment)

```
env PIPENV_PIPFILE=~/path/to/repo/Pipfile pipenv run python ~/path/to/repo/learning/multi_layer_perceptron.py ~/path/to/training_data.txt
```

## comparing the regression with simulation data

The script `calc_predicted.py` makes plots that compare the regression model and simulations.
Before running this script, run simulations on [OACIS](https://github.com/crest-cassia/oacis). These results are collected when executing this script and the figures are made.

## making an interactive chart

To show the regression results on a web browser, convert h5 file to a JS file:

```
cd chart
tensorflowjs_converter --input_format keras model.h5 converted_k
```

Save converted results for each output into the directories `converted_k`, `converted_kk`, `converted_w`, `converted_cc`, `converted_ck`, `converted_o`, `converted_ow`, `converted_perc_a`, `converted_perc_d`.
Then, launch a web server and open `index.html`.

## obtained meta-models

The meta-models obtained by the regression are available in `docs` directory. Find `model_*.h5` files.

## sensitivity analysis

A scrirpt to conduct global sensitivity analysis is in `sensitivity_analysis/sensitivity_analysis.py`. The sensitivity analysis is conducted against the meta-model (regression model) so make sure to run regression first.
After you prepared the metamodel, run the script like

```shell
python sensitivity_analysis.py
```

# Reference

