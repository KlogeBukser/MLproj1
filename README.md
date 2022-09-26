# MLproj1
Machine Learning Project 1

## TODO (code):

### Part b
- scaling/centering of the data (for example by subtracting the mean value) (There is functions for this, but it is currently not in use)

## Code structure:

- The functions that compares models for different polynomial degrees run in decreasing order of complexity. This is done to for optimization reasons. It lets us find all features and make the design matrix just once, removing/ignoring the remainder when reducing the complexity.
- The model object saves the features as an array of functions, it also saves integers fro number of features, and polynomial degree.
- In addition it also saves the coefficients beta, but the method fit(X_train,z_data) must be use to produce the beta values. When complexity is reduced, the previous beta values are discarded.
