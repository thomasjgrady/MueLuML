A = csvread('grid_search.csv');
A = reshape(A, [10, 10]);
imagesc(A)