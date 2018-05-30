function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the
%project where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

%values = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
values = [0.1; 0.3; 1];
minError = 999999999;
for i=1:length(values)
  for j=1:length(values)
    model = svmTrain(X, y, values(i), @(x1, x2) gaussianKernel(X(:,1),X(:,2),values(j)) );
    predictions = svmPredict(model, Xval);
    error = mean(double(predictions ~= yval));
    fprintf('error: %f, %f, %f\n', error, values(i), values(j));
    if (error < minError)
      minError = error;
      C = values(i); sigma = values(j);
    end
  end
end
