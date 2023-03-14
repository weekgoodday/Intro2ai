#include <ctime>
#include <iostream>
#include <vector>
#include "tictactoe.hpp"

using namespace std;

class TicTacToePolicyBase{ //TicTacToe类为游戏环境类，其中提供表示动作的Action类，表示状态的State类
    public:
        virtual TicTacToe::Action operator()(const TicTacToe::State& state) const = 0;
};

// TicTacToePolicyRandom的策略： randomly select a valid action for the step.
class TicTacToePolicyRandom : public TicTacToePolicyBase{
    public:
        TicTacToe::Action operator()(const TicTacToe::State& state) const {
            vector<TicTacToe::Action> actions = state.action_space();
            int n_action = actions.size();
            int action_id = rand() % n_action;  //随机执行一个action
            if (state.turn == TicTacToe::PLAYER_X){
                return actions[action_id];
            } else {
                return actions[action_id];
            }
        }
        TicTacToePolicyRandom(){
            srand(time(nullptr)); //构造函数 随机发生器的初始化函数
        }
};

#define N 999 //小数精度为3
#define SIZE 300000 //足够装18位二进制board
//改成值函数的策略 select the first valid action.
class TicTacToePolicyDefault : public TicTacToePolicyBase{
    public:
        double eps;
        float statetable[SIZE]; //核心 状态估值表 turn不重要board包含了turn的信息，board用不超过18位的2进制整数表示，因此只需要用数组即可
    public:
        TicTacToe::Action operator()(const TicTacToe::State& state) const {
            vector<TicTacToe::Action> actions = state.action_space();
            if (state.turn == TicTacToe::PLAYER_X){
                // TODO
                float random = rand()%(N+1)/(float)(N+1); //三位精度0-1小数
                int action_length=actions.size();
                if (action_length==1) return actions[0];
                if (random<eps*action_length/(action_length-1)) //减去后面rand出最大的影响
                {
                    int action_id=rand()%action_length; //产生随机当前可选的action
                    cout<<"This time I choose to explore!"<<endl; //其实也不一定explore，有可能正好选到那个最大的，不过前面已经*n/(n-1)，就概率来说已经扣除影响
                    //cout<<action_id<<endl;
                    return actions[action_id];
                }
                else 
                {
                    //找一个最大value的action
                    TicTacToe::State statetemp=state;
                    statetemp.put(actions[0]);
                    float max=statetable[statetemp.board];
                    int maxi=0;
                    for(int i=1;i<action_length;i++)
                    {
                        TicTacToe::State statetemp=state;
                        statetemp.put(actions[i]);
                        if(statetable[statetemp.board]>max)
                        {
                            max=statetable[statetemp.board];
                            maxi=i;
                        }
                    }
                    return actions[maxi];
                }
            } 
            else { //对手是固定策略，选第一个可用的
                return actions[0];
            }
        }
        TicTacToePolicyDefault() //构造函数
        {
            eps=0.1; //超参数可改，探索比例，有多大的概率探索新的
            //memset(statetable,0,SIZE*sizeof(float));
            fill(statetable,statetable+SIZE,0.5); //初始化全部都是0.5

        }
};


#include <chrono>
#include <thread>
int main(){
    
    // set verbose true
    TicTacToe env(false);
    int iter =50000; //超参数可改，实测改成1000 10000会有不同的结果哟~
    float alpha=0.1; //超参数可改，更新比例（value表的数值更新）
    TicTacToePolicyDefault policy;
    int count_win=0;
    int count_total;
    float win_rate;
    cout<<"Before starting officially, I want to train "<<iter<<" times."<<endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    for(int g=0;g<iter;g++)
    {
        bool done = false;
        // TicTacToePolicyRandom policy; //采用两边都随机的policy
        while (!done){
            TicTacToe::State state = env.get_state();
            TicTacToe::Action action = policy(state);
            env.step(action);
            done = env.done(); //一定要能判断结束 每次结束一更新
            // env.step_back(); 
            // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
        //update statetable
        float value_new; //存这次新的V(St+1)
        int board=env.get_state().board;
        if(env.winner()==env.PLAYER_X)
        {
            count_win+=1;
            value_new=1;
            policy.statetable[board]=1; //最终的胜利状态改成1 前面的状态回溯 下同
        }
        else if(env.winner()==env.PLAYER_O)
        {
            value_new=0;
            policy.statetable[board]=0; 
        }
        else
        {
            value_new=0.5;
            policy.statetable[board]=0.5;
        }
        //cout<<board<<" ";
        while(env.step_back()) //每次回溯到头，甚至下一次不需要reset
        {   
            board=env.get_state().board;
            //cout<<board<<" ";
            policy.statetable[board]+=alpha*(value_new-policy.statetable[board]);
        }
        //cout<<endl;
        win_rate=count_win/float(g);
        cout<<"We've played "<<g<<" times. My wining rate is "<<win_rate<<endl;
    }
    
    cout<<"Training over! Let's begin!"<<endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    env.verbose=true;
    policy.eps=0;
    bool done=false;
    while(!done)
    {
        TicTacToe::State state=env.get_state();
        TicTacToe::Action action= policy(state);
        env.step(action);
        done=env.done();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    //int winner = env.winner(); //winner == 0b01 或者说 PLATER_X; winner == 0b10 或者说PLAYER_O
    return 0;
};