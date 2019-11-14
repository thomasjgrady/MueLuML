files = dir('coords_*');
num_files = length(files);
results = cell(length(files), 1);
for i = 1:num_files
  results{i} = csvread(files(i).name);
end

C = results;
vertcat(C{:});
A = cell2mat(C);

[m, n] = size(A);

f = A(:,1);
A = A(:,2:n);

figure
lambda = eigs(A * A');
plot(lambda)
title('Eigenvalues of parameter space sampled data (AA^T)');

%figure
%gplotmatrix(A);

D = pdist(A);
A = mdscale(D, 2);

x = A(:,1);
y = A(:,2);
tri = delaunay(x,y);

figure
h = trisurf(tri, x, y, f);
axis vis3d