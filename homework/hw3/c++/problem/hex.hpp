#pragma once

#include <array>
#include <cassert>
#include <cinttypes>
#include <iostream>

#include "../interface/game_state.hpp"
#include "../utils/union_find_set.hpp"

// N*N的六边形棋
template<int _N>
class HexState : public NPlayerGameStateBase<int, 2>{
private:
    
    static constexpr int N = _N;

    // 红方先手
    static constexpr int8_t R = 0b01, B = 0b10;

    int steps;
    std::array<int8_t, N*N> board;

    // 用于检测红色/蓝色棋块连通性
    mutable UnionFindSet r_detector, b_detector;

    bool r_win() const {
        return r_detector.find(N*N) == r_detector.find(N*N+1);  //只要红色最上面和最下面连通，红赢
    }

    bool b_win() const {
        return b_detector.find(N*N) == b_detector.find(N*N+1);  //只要蓝色最左边与最右边连通，蓝赢
    }

public:
    
    HexState() : steps(0), board{0}, r_detector(N*N+2), b_detector(N*N+2) {
        
        // N*N:   up, red      /  left, blue
        // N*N+1: bottom, red  /  right, blue
        for (int i = 0; i < N; ++ i){
            r_detector.join(N*N, i);  //红色最上面 0-10 与121合并
            r_detector.join(N*N+1, N*N-1-i);  //红色最下面 122与110-120合并
            b_detector.join(N*N, N*i);  //蓝色最左边121 与0、11、22...110合并
            b_detector.join(N*N+1, N*(i+1)-1);  //蓝色最右边 122与10、21、32...120合并
        } //确实是11*11，最终判定条件是最上一排和最下一排连通 上面11个位置必须有一个红子才能与下面几排相连
    }

    bool done() const override {
        return r_win() or b_win();
    }

    int active_player() const override {
        return steps & 1;
    }

    std::vector<double> rewards() const override {
        static const std::vector<double> score_r_win {1, -1},
            score_b_win {-1, 1},
            score_tie {0, 0};
        
        return r_win() ? score_r_win : (b_win() ? score_b_win : score_tie); 
    }

    std::vector<double> cumulative_rewards() const override {
        return rewards();
    }

    inline int n_actions() const override {
        return N*N - steps;
    }

    std::vector<int> action_space() const override {
        std::vector<int> actions;
        for (int i = 0; i < N*N; ++ i){
            if (board[i] == 0){  // board是由N*N个数组成的array 只要还是0就可以落子，落子后会变成01红 or 10蓝
                actions.push_back(i);
            }
        }
        return actions;
    }

    const HexState& next(const int& action) const override {  // action是一个整数 0-120
        static HexState next_state;
        
        assert(board[action] == 0);
        next_state = *this;
        
        next_state.board[action] = active_player() == 0 ? R : B;  //下一步 若这一步是偶数 落红 step++
        
        std::vector<int> neighbors {
            action-N, action+1-N, 
            action+1, action+N, 
            action-1+N, action-1
        };  //只有6个，上、右上、右、下、左下、左

        bool not_top = action >= N, 
            not_bottom = action < N*N-N,
            not_left = action % N != 0,
            not_right = action % N != N-1;

        std::vector<int8_t> conditions {
            not_top, not_top and not_right,
            not_right, not_bottom,
            not_bottom and not_left, not_left 
        };  // 6个元素的vector 上、右上、右、下、左下、左

        UnionFindSet& detector = active_player() == 0 ? next_state.r_detector : next_state.b_detector;  //step是偶数（从0开始） 是红走；否则蓝走

        for (int i = 0; i < conditions.size(); ++ i){ // 最上、最右上、最右、最下、最左下、最左也都可以落子，只不过join neighbors效率不高，一次落子把同色的邻居都加入该子的叶节点
            if (conditions[i] and 
                next_state.board[neighbors[i]] == next_state.board[action]){
                detector.join(neighbors[i], action);
            }
        }

        ++ next_state.steps;
        return next_state;
    }

    void show() const override {
        const static char pieces[] = "_XO";
        for (int i = 0; i < N; ++ i){
            for (int j = 0; j < i; ++ j){
                std::cout << ' ';
            }
            for (int j = 0; j < N; ++ j){
                std::cout << pieces[board[i*N+j]] << ' ';
            }
            std::cout << '\n';
        }
    }

    friend struct std::hash<HexState>;
    friend bool operator== (const HexState& s1, const HexState& s2){
        return s1.board == s2.board;
    }
};

template<int N>
struct std::hash<HexState<N> >{
    size_t operator() (const HexState<N>& s) const {
        size_t code = s.steps;
        for (int i = 0; i < s.board.size(); ++ i){
            code ^= size_t(s.board[i]) << (i & ((sizeof(size_t) << 3) - 1));  //很复杂，感觉在对棋盘状态编码
        }
        return code;
    }
};
//size_t类型是一个类型定义，通常将一些无符号的整形定义为size_t，在使用 size_t 类型时，编译器会根据不同系统来替换标准类型。
//每一个标准C实现应该选择足够大的无符号整形来代表该平台上最大可能出现的对象大小。
//在使用 size_t 类型时，编译器会根据不同系统来替换标准类型。
//i&((sizeof(size_t)<<3)-1) 后面在我的电脑上得到63 相当于imod63 0b01<<i 0b10<<i 是2的倍数 0^数字是数字本身，1^数字为取反