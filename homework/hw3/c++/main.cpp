#include <cfloat>
#include <cmath>
#include <cinttypes>
#include <cassert>
#include <ctime>
#include <iostream>
#include <vector>
#include <type_traits>
#include <unordered_map>
#include <array>
#include <random>

#include "jsoncpp/json.h" // Botzone上C++编译时默认包含此库，无需提交此库内容

// 提供随机变量的工具类
class RandomVariables{
private:

    // 固定随机种子
    RandomVariables() = default;
    
    // 非固定随机种子
    // RandomVariables() : random_engine(time(nullptr)) {}
    
    ~RandomVariables() = default;

    std::default_random_engine random_engine;
    std::uniform_real_distribution<double> uniform_dist;
    std::uniform_int_distribution<int> uniform_int_dist;

    static RandomVariables rv;

public:

    // 均匀分布的正整数
    static int uniform_int(){
        return rv.uniform_int_dist(rv.random_engine);
    }

    // [0,1)均匀分布的实数
    static double uniform_real(){
        return rv.uniform_dist(rv.random_engine);
    }

    // 等概率分布的{0,1,2,n-1}排列
    static std::vector<int> uniform_permutation(int n){
        std::vector<int> permutation(n);
        for (int i = 0; i < n; ++ i){
            permutation[i] = i;
        }

        for (int i = 0, j; i < n; ++ i){
            j = uniform_int() % (n - i) + i;
            std::swap(permutation[i], permutation[j]);
        }

        return permutation;
    }
};

RandomVariables RandomVariables::rv;

// 游戏状态接口
template<typename ActionType>
class GameStateBase{
public:
    
    GameStateBase() = default;
    virtual ~GameStateBase() = default;

    using ActionBaseType = ActionType;

    // 玩家人数
    virtual int n_players() const = 0;
    
    // 当前决策玩家
    virtual int active_player() const = 0;

    // 动作空间大小
    virtual int n_actions() const = 0;

    // 当前状态下的动作空间
    virtual std::vector<ActionType> action_space() const = 0;
    
    // 各个玩家在之前的游戏过程中累计的回报
    virtual std::vector<double> cumulative_rewards() const = 0;
    
    // 各个玩家在转移到当前状态后获得的即时回报
    virtual std::vector<double> rewards() const = 0;

    // 游戏是否结束
    virtual bool done() const = 0;

    // 展示游戏状态
    virtual void show() const = 0;

    // 当前玩家选择action后转移到的新状态
    virtual const GameStateBase& next(const ActionType&) const = 0;

    // 状态哈希
    friend struct std::hash<GameStateBase>;

    // 子类应重载operator==
    friend bool operator== (const GameStateBase& s1, const GameStateBase& s2){
        return s1.cumulative_rewards()[s1.active_player()] == s2.cumulative_rewards()[s2.active_player()];
    }
};

// n人游戏状态接口
template<typename ActionType, int n>
class NPlayerGameStateBase : public GameStateBase<ActionType>{
protected:
    static constexpr int _n_players = n;
public:
    NPlayerGameStateBase() = default;
    virtual ~NPlayerGameStateBase() = default;

    int n_players() const override {return _n_players;}    
};


// 并查集算法
class UnionFindSet{
private:
    
    // 保存元素所属的类别
    std::vector<int> color;

public:

    UnionFindSet(int n){
        for (int i = 0; i < n; ++ i){
            color.push_back(i);
        }
    }

    UnionFindSet() = default;

    // 查找一个元素所属的类别
    int find(int x){
        return x == color[x] ? x : (color[x] = find(color[x]));
    }

    // 合并两个元素所属的类别
    void join(int x, int y){
        int cx = find(x), cy = find(y);
        if (cx != cy){
            color[cx] = cy;
        }
    }
};

// N*N的六边形棋状态类
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
        return r_detector.find(N*N) == r_detector.find(N*N+1);
    }

    bool b_win() const {
        return b_detector.find(N*N) == b_detector.find(N*N+1);
    }

public:
    
    HexState() : steps(0), board{0}, r_detector(N*N+2), b_detector(N*N+2) {
        
        // 各边界单独用一个元素表示
        // N*N:   up, red      /  left, blue
        // N*N+1: bottom, red  /  right, blue
        for (int i = 0; i < N; ++ i){
            r_detector.join(N*N, i);
            r_detector.join(N*N+1, N*N-1-i);
            b_detector.join(N*N, N*i);
            b_detector.join(N*N+1, N*(i+1)-1);
        }
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
            if (board[i] == 0){
                actions.push_back(i);
            }
        }
        return actions;
    }

    const HexState& next(const int& action) const override {
        static HexState next_state;
        
        assert(board[action] == 0);
        next_state = *this;
        
        next_state.board[action] = active_player() == 0 ? R : B;
        
        std::vector<int> neighbors {
            action-N, action+1-N, 
            action+1, action+N, 
            action-1+N, action-1
        };

        bool not_top = action >= N, 
            not_bottom = action < N*N-N,
            not_left = action % N != 0,
            not_right = action % N != N-1;

        std::vector<int8_t> conditions {
            not_top, not_top and not_right,
            not_right, not_bottom,
            not_bottom and not_left, not_left 
        };

        UnionFindSet& detector = active_player() == 0 ? next_state.r_detector : next_state.b_detector;

        for (int i = 0; i < conditions.size(); ++ i){
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
            code ^= size_t(s.board[i]) << (i & ((sizeof(size_t) << 3) - 1));
        }
        return code;
    }
};

// 搜索树结点类
class SearchTreeNode{
private:
    
    // 为树中的结点建立唯一编号
    int _index;
    
    // 该结点有多少个子结点
    int _n_children;

    // 该结点的父结点
    SearchTreeNode* _parent;

    // 该结点的子结点列表
    std::vector<SearchTreeNode*> _children;

public:

    SearchTreeNode() = default;

    // 构造一个编号为index的结点，该结点没有父节点和子结点
    SearchTreeNode(int index) : _index(index), _parent(nullptr), _n_children(0) {}

    // 将child添加为当前结点的子结点
    void add_child(SearchTreeNode* child){
        ++ _n_children;
        _children.push_back(child);
        child->_parent = this;
    }

    // 返回结点编号
    int index() const {return _index;}

    // 当前结点有多少子结点
    int n_children() const {return _n_children;}

    // 当前结点的父节点
    SearchTreeNode* parent() const {return _parent;}

    // 当前结点的第child_index个子结点
    SearchTreeNode* child(int child_index) const {return _children[child_index];}

    // 当前结点的子节点列表
    const std::vector<SearchTreeNode*>& children() const {return _children;}

    // 比较结点相同只需要比较编号
    friend bool operator== (const SearchTreeNode& n1, const SearchTreeNode& n2){
        return n1._index == n2._index;
    }

    friend struct std::hash<SearchTreeNode>;
};

template<>
struct std::hash<SearchTreeNode>{
    size_t operator() (const SearchTreeNode& n) const {
        return n._index;
    }
};

// 搜索树类
class SearchTree{
private:
    
    // 单调递增的结点标识符
    int _unique_identifier;

    // 树当前的大小
    int _n_nodes;

    // 搜索树的根
    SearchTreeNode* _root;

    // 将结点的编号映射到结点指针
    std::unordered_map<int, SearchTreeNode*> _node_of;
    
public:

    // 构造一棵搜索树，树根为root，仅包含1个根结点
    SearchTree() : _root(new SearchTreeNode(0)), _unique_identifier(1), _n_nodes(1) {
        _node_of[0] = _root;
    }

    // 返回结点编号对应的结点指针
    SearchTreeNode* node_of(int index) const {
        return _node_of.at(index);
    }

    // 删除掉某棵子树之外的部分，更新树根和大小信息
    void destroy_except_subtree(SearchTreeNode* subtree_root){
        SearchTreeNode* node;
        std::stack<std::pair<SearchTreeNode*, int> > node_stack;
        node_stack.push(std::make_pair(_root, 0));
        std::pair<SearchTreeNode*, int> node_searched;
        int searched, deleted = 0;

        // 深度优先删除结点
        while (not node_stack.empty()){
            node_searched = node_stack.top();
            node_stack.pop();
            node = node_searched.first;
            searched = node_searched.second;

            // 如果当前探索到了不应当删除的子树的根，则不进行删除操作
            if (node == subtree_root){
                continue;
            }

            // 如果所有子结点已经删除，则删除该结点自身
            if (searched >= node->n_children()){
                ++ deleted;
                _node_of.erase(node->index());
                delete node;
            
            } else {
                node_stack.push(std::make_pair(node, searched+1));
                node_stack.push(std::make_pair(node->child(searched), 0));
            }
        }

        _n_nodes -= deleted;
        _root = subtree_root;
    }

    // 返回树根
    SearchTreeNode* root() const {return _root;}

    // 生成一个新的结点，这个结点和搜索树中已有的结点都不相同
    SearchTreeNode* create_node(){
        auto new_node = new SearchTreeNode(_unique_identifier);
        _node_of[_unique_identifier] = new_node;
        ++ _unique_identifier;
        return new_node;
    }

    // 将create_node生成的结点添加为parent的子结点，更新树的大小
    void add_as_child(SearchTreeNode* parent, SearchTreeNode* child){
        parent->add_child(child);
        ++ _n_nodes;
    }

    // 返回树当前的大小
    int n_nodes() const {return _n_nodes;}

    // 销毁整棵树（应当保证每个节点都是调用create_node生成的）
    ~SearchTree() {
        destroy_except_subtree(nullptr);
    }
};


// 选择算法接口，当done()为true时selected_index()即为选择的值序号
class SelectionBase{
public:

    SelectionBase() = default;
    virtual ~SelectionBase() = default;

    virtual void initialize(int items, double initial_value) = 0;
    virtual void submit(double value) = 0;
    virtual bool done() const = 0;
    virtual int selected_index() const = 0;
};

// 最大选择算法，选择所有值中最大的
class MaxSelection : public SelectionBase{
private:

    double max_value;
    int index, total_items, submitted_items;

public:

    MaxSelection() = default;

    // items:备选集合大小，initial_value:与算法无关
    void initialize(int items, double initial_value) override {
        index = 0;
        total_items = items;
        submitted_items = 0;
    }

    void submit(double value) override {
        if (submitted_items == 0 or value > max_value){
            max_value = value;
            index = submitted_items;
        }
        ++ submitted_items;
    }

    bool done() const override {
        return submitted_items >= total_items;
    }

    int selected_index() const override {
        return index;
    }
};

std::vector<double> operator+ (const std::vector<double>& a, const std::vector<double>& b){
    std::vector<double> result(a);
    for (int i = 0; i < result.size(); ++ i){
        result[i] += b[i];
    }
    return result;
}

std::vector<double>& operator+= (std::vector<double>& a, const std::vector<double>& b){
    a = a + b;
    return a;
}

template<typename GameState>
class MonteCarloTreeSearch{
private:

    using ActionType = typename GameState::ActionBaseType;

    static_assert(std::is_base_of<GameStateBase<ActionType>, GameState>::value, "GameState not derived from GameStateBase.");

    // 搜索树
    SearchTree tree;

    // 把状态映射到结点编号
    std::unordered_map<GameState, int> state_to_index;

    // 把结点编号映射为状态
    std::unordered_map<int, GameState> index_to_state;

    // 把结点的index映射到访问次数
    std::unordered_map<int, int> visit_count_of;

    // 把结点的index映射到访问该节点各玩家所获得的总价值
    std::unordered_map<int, std::vector<double> > value_sums_of;
    
    // 完全随机模拟
    std::vector<double> simulate_from(GameState state) const {
        int action_id;
        while (not state.done()){
            action_id = RandomVariables::uniform_int() % state.n_actions();
            state = state.next(state.action_space()[action_id]);
        }
        return state.cumulative_rewards();
    }

    // 采样一条路径，更新途径结点访问计数和总价值
    std::vector<double> sample_path(const GameState& state, double exploration){
        
        // 当前状态在搜索树中的编号
        int index = state_to_index[state];

        // 当前状态对应的搜索树结点
        SearchTreeNode* node = tree.node_of(index);
        
        SearchTreeNode* child;
        
        GameState next_state;
        
        std::vector<double> values;

        // 访问到的结点计数增加
        ++ visit_count_of[index];

        // 如果未完全扩展当前结点，选择一个没有做过的动作来尝试，扩展后模拟
        if (node->n_children() < state.n_actions()){
            
            // 扩展的结点对应的状态
            next_state = state.next(state.action_space()[node->n_children()]);

            // 在搜索树上添加子结点
            child = tree.create_node();
            tree.add_as_child(node, child);
            
            // 维护搜索树上结点编号与状态之间的对应关系
            state_to_index[next_state] = child->index();
            index_to_state[child->index()] = next_state;

            // 子结点初始访问计数为1
            visit_count_of[child->index()] = 1;

            // 子结点初始累计收益为模拟得到的值
            values = simulate_from(next_state);
            value_sums_of[child->index()] = values;

        // 如果当前结点已经完全扩展，那么按照UCT算法选择其中一个子结点继续
        } else if (node->n_children() > 0){

            MaxSelection selection;
            selection.initialize(node->n_children(), -DBL_MAX);

            for (int i = 0, child; i < node->n_children(); ++ i){
                
                // child是当前选择的子结点在树中的编号
                child = node->child(i)->index();

                // 选择UCT值最大的子结点继续探索
                selection.submit(value_sums_of[child][state.active_player()] / visit_count_of[child]
                    + exploration * sqrt(log(visit_count_of[index]) / visit_count_of[child])
                );
            }

            next_state = state.next(state.action_space()[selection.selected_index()]);
            values = sample_path(next_state, exploration);
        } else {
            values = state.cumulative_rewards();
        }

        value_sums_of[index] += values;
        return values;
    }

public:

    MonteCarloTreeSearch(const GameState& root_state){

        // _root_state状态对应树根，树根在搜索树中编号为0
        state_to_index[root_state] = 0;
        index_to_state[0] = root_state;

        // 初始时树根访问计数为0
        visit_count_of[0] = 0;

        // 初始时树根没有累计收益
        value_sums_of[0] = std::vector<double>(root_state.n_players(), 0);
    }
    
    ActionType select_action(int iterations, double exploration){

        GameState root_state = index_to_state[0];

        for (int i = 0; i < iterations; ++ i){

            // 从树根对应的状态开始，每次采样出一条路径，更新途经状态的访问计数和总价值
            sample_path(root_state, exploration);
        }
        
        SearchTreeNode* root = tree.root();
        
        MaxSelection selection;
        selection.initialize(root->n_children(), -DBL_MAX);
        
        // 依次考虑树根扩展出来的所有子结点，从中选择一个平均价值最高的
        for (int i = 0, child; i < root->n_children(); ++ i){

            // child是当前选择的子结点在树中的编号
            child = root->child(i)->index();

            // 按平均价值贪心选择
            selection.submit(value_sums_of[child][root_state.active_player()] / visit_count_of[child]);
        }

        // 也可以按照访问次数贪心选择
        
        return root_state.action_space()[selection.selected_index()];
    }
};

//////////////////////////////////////////////////
/**********************Main**********************/

using namespace std;

// 棋盘大小
const int N = 11;

// UCT探索项系数
const double exploration = 0.2;

// 迭代次数
const int iterations = 5200;

// 搜索生成state下的动作
Json::Value get_next_action(const HexState<N>& state, bool forced_flag){
    MonteCarloTreeSearch<HexState<N> > mcts(state);
    int action = mcts.select_action(iterations, exploration);

    if (forced_flag){
        action = 1*11+2;
    }

	Json::Value action_json;
	action_json["x"] = action / N;
	action_json["y"] = action % N;
	return action_json;
}

int main(){
       
    HexState<N> state;
	
    // 读入JSON
	string str;
	getline(cin, str);
	Json::Reader reader;
	Json::Value input;
	reader.parse(str, input); 
	
    // 分析自己收到的输入和自己过往的输出，并恢复状态
	int turn_id = input["responses"].size();
    int x, y;
    bool forced_flag;
	for (int i = 0; i < turn_id; i++) {
		x = input["requests"][i]["x"].asInt();
        y = input["requests"][i]["y"].asInt();
        if (x >= 0 and y >= 0){
            state = state.next(x * N + y);
        }
        x = input["responses"][i]["x"].asInt();
        y = input["responses"][i]["y"].asInt();
        state = state.next(x * N + y);
	}
    x = input["requests"][turn_id]["x"].asInt();
    y = input["requests"][turn_id]["y"].asInt();

    if (x >= 0 and y >= 0){
        state = state.next(x * N + y);
        forced_flag = false;
    } else if (input["requests"][0].isMember("forced_x")){
        forced_flag = true;
    } else {
        forced_flag = false;
    }

    // 做出决策存为action 
	// 输出决策JSON
	Json::Value result;
	
    result["response"] = get_next_action(state, forced_flag);

	Json::FastWriter writer;
	
    cout << writer.write(result) << endl;
	
    return 0;
}