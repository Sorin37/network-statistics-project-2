# Files

- `metropolis.py`: Is the implementation of the algorithm depicted in the slides, especially slide 30. 
- 'main.py': just a file to run the experiment
- `convergence_weight_history.html`: plot that shows the evolution of the weight during the initial `initial_convergence` algorithm
- `sample_history.html`: plot that shows the evolution of the weight during the metropolis algorithm, as well as the times when a new candidate was accepted or rejected.

## Metropolis.py functions

- **`initial_convergence`** performs the convergence step of the Metropolis algorithm, and outputs the best weight and grouping found. The weight can be used as a constant for the reliabilities. 
- **`sample_different_groupings`** Takes the result from initial_convergence and continues to sample groupings via random changes that have a higher weight. Whenever a new candidate appears it is tested against the already accepted groupings. A new candidate is accepted if the maximum obtained NMI score against the already accepted groupings is smaller than the `nmi_threshold` parameter (smaller is better if the goal is to have different groupings).

More details about the parameters can be found in the docstring of the functions.

## Differences from the course code

- the `initial_convergence` function follows the exact code as in `MetropolisAlgorithm.py`. The only modification lies in a trick for numerical stability that prevents calculating the probability of acceptance to lead to an overflow (the code was still doing the correct thing, but now it just prevents the error from appearing :D)
- the 'sample_different_groupings' follows the same logic found in the `MetropolisAlgorithm.py` to some degree:
  - in the file, the MI is calculated and picks only the top 10 groupings when the MI has changed. Here I used the NMI score since it's easier to understand with values between [0, 1].  
  - Since our dataset has 235 taxa, making one change at a time seems to be pretty slow, so multiple changes can be done at once to speed things up (even though this hurts the performance of the algorithm theoretically)

## How to run through the terminal
Activate the virtual environment
Navigate to the metropolis_sorin folder
Run the script

```
venv/Scripts/activate
cd metropolis_sorin
python main.py
```

## Reflection

- *Randomness*: the algorithm is *very* sensitive to randomness, every time I ran the algorithm with the same configurations, I got a different number of groupings
- *NMI Score*: I'm not entirely convinced by the usefulness of this metric since it is a statistical metric that takes into account only the grouping of the nodes and no information about the edges. For smaller graphs it yielded a low NMI score even though only one node had a different grouping. Maybe we can look into better ways to sample different groupings.  
