#include <iostream>
#include <stack>
#include <vector>
#include <cstring>
#include <utility>
#include <chrono>
#include <algorithm>
#include "parser.hpp"

typedef std::vector<std::vector<bool>> Domains;

const bool ENABLE_LOGGING = false;

// N-Queens node
struct Node {
    int depth; // depth in the tree
    Domains domains;
    int index;
    // Initialize the domains for each variable
    Node(size_t N, int *u): depth(0), domains(N), index(0) {
        domains.resize(N);

        for (int i = 0; i < N; i++){

            // Domain of variable i has flags for each value
            // from 0 to upperbound
            domains[i].resize(u[i]+1);
            for (int j = 0; j <= u[i]; j++){
                // At first, every variable can take any value in 
                // the respective domain
                domains[i][j] = true;
            }
        }

    }

    Node(const Node&) = default;
    Node(Node&&) = default;
    Node() = default;
};

void log_info(const std::string &message) {
    if (ENABLE_LOGGING) {
        std::cout << "[INFO] " << message << std::endl;
    }
}

void print_domains(Domains &domains){
    if(!ENABLE_LOGGING)
        return;
        
    for(int i = 0; i < domains.size(); i++){
        std::cout << "Domain of variable " << i << ": ";
        for(int j = 0; j < domains[i].size(); j++){
            std::cout << domains[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

// Update the domains given the constraints
// Returns true if the domains were updated
bool update_domains(Domains &domains, Domains& new_domains, int **C){

    bool updated = false;

    new_domains = Domains(domains);

    // For every variable
    for(int i = 0; i < domains.size(); i++){

        auto d = domains[i];

        // We check the constraints that involve only one variable,
        // because they are the only ones that can reduce the domains
        // of other variables
        if(std::count(d.begin(), d.end(), true) > 1)
            continue;

        // Find the only value that is true in the domain
        int value = std::find(d.begin(), d.end(), true) - d.begin();

        // If there is a constraint between i and j, then I remove
        // the value from the domain of j            
        for(int j = 0; j < domains.size(); j++){

            // If the value is out of bounds, then we skip
            if(value > domains[j].size())
                continue;

            if (C[i][j] == 1){
                if(new_domains[j][value]){
                    new_domains[j][value] = false;    
                    updated = true;
                }
            }
        }
    }

    return updated;
}

std::vector<Node> generate_children(const Node& parent, int variable_i, int **C){
    std::vector<Node> children;

    std::vector<bool> d = parent.domains[variable_i];
    std::vector<int> true_values;
    // Find all the true values in the domain of the variable
    for(int i = 0; i < d.size(); i++){
        if(d[i] == true)
            true_values.push_back(i);
    }

    // Generate children with unitary domains
    for(int i = 0; i < true_values.size(); i++){
        Node child(parent);
        child.domains[variable_i] = std::vector<bool>(child.domains[variable_i].size(), false);
        child.domains[variable_i][true_values[i]] = true;
        child.depth  = parent.depth + 1;
        children.push_back(std::move(child));
    }
    return children;
}

void evaluate_and_branch(Node& parent, std::stack<Node>& stack, int **C, size_t& tree_loc, size_t& num_sol){

    Node updated_parent(parent);
    int curr_depth = parent.depth;
    int n = parent.domains.size(); 

    log_info("Evaluating node at depth " + std::to_string(curr_depth));

    // Update the domains using the constraints
    if(curr_depth == n){
        log_info("Solution found!");
        num_sol++;
        return;
    }
    bool updated;
    int a;
    do{
        log_info("Updating domains...");
        updated = update_domains(updated_parent.domains, updated_parent.domains, C);
        print_domains(updated_parent.domains);
    }while(updated);

    log_info("Updated domains: ");
    print_domains(updated_parent.domains);

    // Generate branches from current variable domain (curr_depth is also current variable index)
    std::vector<Node> children = generate_children(updated_parent, curr_depth, C);

    log_info("Number of children: " + std::to_string(children.size()));

    // If there is only one child (meaning that the variable could not be branched)
    // and we are at the last variable, then we have a solution
    if(curr_depth == n - 1 && children.size() == 1){
        log_info("Solution found!");
        num_sol++;
        return;
    }

    // Else, we branch 
    for(int i = 0; i < children.size(); i++){
        log_info("Pushing child " + std::to_string(i) + " with domain: ");
        print_domains(children[i].domains);
        stack.push(std::move(children[i]));
        tree_loc++;
    }
}

int main(int argc, char *argv[]){

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <filename>" << std::endl;
        return 1;
    }

    std::string filename = argv[1];
    Data data;
    if (data.read_input(filename.c_str())){
        data.print_n();
        data.print_u();
        data.print_C();
    } else {
        std::cerr << "Failed to read input file: " << filename << std::endl;
        return 1;
    }

    // Number of variables
    int n = data.get_n();
    int *u = data.get_u();
    int **C = data.get_C();

    // Nodes tree
    Node root(n, u);
    assert(root.domains.size() == n);

    std::stack<Node> stack;
    stack.push(std::move(root));

    print_domains(stack.top().domains);

    // Counters
    size_t exploredTree = 0;
    size_t exploredSol = 0;

    // Domains for each variable
    while(stack.size() != 0){
        Node current(std::move(stack.top()));
        stack.pop();
        evaluate_and_branch(current, stack, C, exploredTree, exploredSol);
    }

    std::cout << "Number of solutions: " << exploredSol << std::endl;
    std::cout << "Number of nodes explored: " << exploredTree << std::endl;

    return 0;
}
