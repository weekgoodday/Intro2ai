hex_mcts_test.cpp是六边形棋蒙特卡洛树搜MCTS 与json无关的实现
g++ hex_mcts_test.cpp -o test --std=c++11 -O3
./test
utils/hex.hpp是六边形棋的状态动作定义
algorithm/monte_carlo_tree_search.hpp是MCTS算法实现

hex_main.cpp实现了面向botzone的蒙特卡洛树搜索六边形棋智能体，输入由Botzone给出，为游戏历史，输出符合Botozne格式，为当前玩家的决策。（功能同main.cpp）
main.cpp与hex_main.cpp功能相同，但将#include的本地库（algorithm/interface/problem/utils）中内容展开到同一个文件中，本文件可以直接提交到botzone作为bot运行。

（另两个minmax和alpha-beta剪枝也有实现，可以用井字棋验证，作业不要求）