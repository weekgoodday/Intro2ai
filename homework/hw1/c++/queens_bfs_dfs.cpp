#include <iostream>
#include <ctime>

#include "problem/queens.hpp"

#include "algorithm/depth_first_search.hpp"
#include "algorithm/breadth_first_search.hpp"

int main(){
    std::ios::sync_with_stdio(false);
    for(int i=8;i<=15;i++)
    {
    time_t t0 = time(nullptr);
    
    QueensState state(i);
    BreadthFirstSearch<QueensState> bfs(state);
    bfs.search(true, false);
    time_t t1 = time(nullptr);
    DepthFirstSearch<QueensState> dfs(state);
    dfs.search(true, false);
    
    std::cout << "bfs: "<< t1-t0 << std::endl;
    std::cout << "dfs: "<< time(nullptr) - t1 << std::endl;
    }
    return 0;
}
