# MLproj1
Machine Learning Project 1

# Structure
- The latex folder contains the .tex file to the report. 
- The src folder contains all the source code, and 2 plot folders, namely, "plots" and "terrain plots". "plots" contains all the figures generated using the Franke function, and "terrain plots" contains the same plots generated using the terrain data provided in the project. 

# Code
- All the code will be run from "main.py" by calling the following functions with the desired variable.
```python
ols()
ridge()
lasso()
```
- The regression models classes are contained in the file "model.py"
- the results are produced in "produce_result.py"
- "plotting.py" contains general functions for plotting both in and outside of the context of this project.
- "generate.py" contains functions which prepare data in the desired format.
- "calculate.py" contains functions that calculates the errors and scores.
- "polyfuncs.py" generates an array of polynomial functions that for the regression.


