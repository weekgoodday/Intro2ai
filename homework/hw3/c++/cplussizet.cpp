#include <array>
#include <cassert>
#include <cinttypes>
#include <iostream>
using namespace std;

int main()
{
    size_t code = 10;
    int8_t s=0b01;
    code ^= size_t(s) << (12 & ((sizeof(size_t) << 3) - 1));  //很复杂，感觉在对棋盘状态编码
    cout<<(size_t(0b01)<<12)<<endl;
    cout<<(size_t(s)<<12)<<endl;
    cout<<(12&63)<<endl;
    cout<<(sizeof(size_t) << 3)<<endl;
    cout<<code<<endl;
    int a = sizeof(size_t);
    cout<<a<<endl;
    return 0;
}