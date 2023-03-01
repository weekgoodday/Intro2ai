hw1：作业一是全局搜索算法。 任务要求见 hw1/README.txt 分N皇后和最短路径问题。
运行环境：
C++是程序在Ubuntu18.04，gcc9.4.0运行（本身是7.5.0但hsc师兄顺便就配好了）。python是在Ubuntu18.04,python3.8运行（本来都想在ubuntu18.04运行，但ubuntu18.04自带7.5.0版的gcc报错，貌似gcc8以上版本才OK）。

N皇后问题：
python/c++都需要运行，但都直接写好了，注释输出、点个运行就可以（hw1/python/queens_bfs_dfs.py），比较宽度优先、广度优先搜索时间空间就行。
最短路径问题：
我用的python版本。要求完成一致代价搜索（已经实现在hw1/python/algorithm/uniform_cost_search.py）、贪心搜索、A*搜索

N皇后搜索问题建模：
每行为树的一层，上一层有N个位置作为节点，向下发展，广搜即先探索完上一层的状态（可放置的位置），再向下，直到最后一行valid或者已经不行即切断。valid的路径就输出。

评价：皇后问题没什么意思，浪费时间整理。