% Load or create your adjacency matrix (adj_matrix) here
function D = dictionary_maken(adj_matrix)
% Calculate degree centrality
N=size(adj_matrix,1);
degree_centrality = sum(adj_matrix);
D=zeros(N,5);
% Calculate betweenness centrality
n = length(adj_matrix);
betweenness_centrality = zeros(n, 1);
G = graph(adj_matrix);
for source = 1:n
    for target = 1:n
        if source ~= target
            shortest_paths = distances(G, source, target);
            num_paths = sum(shortest_paths);
            betweenness_centrality(source) = betweenness_centrality(source) + (num_paths / ((n - 1) * (n - 2)));
        end
    end
end

% Calculate eigenvector centrality using the power iteration method
num_iterations = 100;
eigenvector_centrality = ones(N, 1);
for iter = 1:num_iterations
    eigenvector_centrality = adj_matrix * eigenvector_centrality;
    eigenvector_centrality = eigenvector_centrality / norm(eigenvector_centrality);
end

% Calculate closeness centrality
%closeness_centrality = 1 ./ sum(shortestpath(graph(adj_matrix), 'Method', 'unweighted'), 2);

% Calculate PageRank centrality using the built-in 'pagerank' function
pagerank_centrality = centrality(graph(adj_matrix), 'pagerank');

% Calculate local clustering coefficient
%local_clustering_coefficient = clustering_coef_wu(adj_matrix);

D(:,1)=degree_centrality/(sum(degree_centrality));
D(:,2)=betweenness_centrality/(sum(betweenness_centrality));
D(:,3)=eigenvector_centrality/(sum(eigenvector_centrality));
%D(:,4)=closeness_centrality/sum(closeness_centrality);
D(:,4)=pagerank_centrality/sum(pagerank_centrality);
%D(3,:)=local_clustering_coefficient/sum(local_clustering_coefficient);
D(:,5)=ones(N,1)/N;
end