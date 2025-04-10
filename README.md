# QBVI

Two Scripts for running a logistic and a linear regression respectively.

In \VBLab\VB\QBVI the QBVI.m file contain a class implementing the QBVI optimizer

To apply the script to a custom model the h_function implementation for the model must be provided, which dependes on the specific loglikelihood function at hands.
To this end provide the corresponding function in \VBLab\VB\QBVI

The implementation is based on the [VBLab](https://github.com/VBayesLab/VBLab) project. 