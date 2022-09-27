# MLproj1
Machine Learning Project 1

## TODO (code):

- More documentation for methods in model and produce_results
- Other ways of plotting beta (not a line graph)

### Part b
- scaling/centering of the data (for example by subtracting the mean value) (There is functions for this, but it is currently not in use)

## Code structure:

- The functions that compares models for different polynomial degrees run in decreasing order of complexity. This is done to for optimization reasons. It lets us find all features and make the design matrix just once, removing/ignoring the remainder when reducing the complexity.
- The model object saves the features as an array of functions, it also saves integers for number of features, and polynomial degree.
- In addition it also saves the coefficients beta, but the method fit(X_train,z_data) must be use to produce the beta values. When complexity is reduced, the previous beta values are discarded.

## Patch notes for model 2.0
- Model will now take training data (x_train,z_train) in constructor
- Model makes dictionary of {name:design matrix} elements
- The (input) training data from constructor is saved as {"train" : design(x_train)}
- new method 'add_x(x,name)' will take new set of inputs, and a name to make new design matrix
- complexity reduction will now reduce the size of every design matrix to match the new number of features

## Comments on model 2.0
- Model will automatically fit to training data on construction
- Design matrices are now fully enclosed inside of model, so the user won't have to consider it anymore
- Storing matrices is better than storing values, as the design matrix won't have to be calculated more than once
- Matrices automatically match the number of features of the model, and won't need to be checked/changed for every prediction
- Predictions can now be made by passing name of set as input
- Name of sets are given by user. Make sure to not override sets by accident

## Potential plans for model improvement
- sort methods according to usage (I.E. put all bootstrap methods together, with clear indentation for easier readability)
- Include functions from calculate.py into Model methods
* calc-design into Model.design
* find_beta into new method in Model
* get_predict into new method in Model

- Some (or all?) of the functions from poly_funcs.py could also be implemented into model
- Ridge + Lasso
- Cross Validation/KFold
- scaling
