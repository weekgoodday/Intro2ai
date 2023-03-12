#pragma once

#include <vector>

// 并查集算法
class UnionFindSet{
private:
    
    // 保存元素所属的类别
    std::vector<int> color;

public:

    UnionFindSet(int n){
        for (int i = 0; i < n; ++ i){
            color.push_back(i); //vector尾加入数据 构造函数 加入0-122
        }
    }

    UnionFindSet() = default;  //

    // 查找一个元素所属的类别
    int find(int x){
        return x == color[x] ? x : (color[x] = find(color[x])); //路径压缩的并查集简写，每个节点都直连根节点
    }

    // 合并两个元素所属的类别
    void join(int x, int y){  //没有设置按秩合并，只是把x的根往y的根节点上合并
        int cx = find(x), cy = find(y);
        if (cx != cy){
            color[cx] = cy;
        }
    }
};