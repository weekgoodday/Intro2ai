#include<iostream>
#include<fstream>
#include<assert.h>
//#include "json.h"
#include "jsoncpp/json.h" // C++编译时默认包含此库
using namespace std;

int main()
{
	Json::Reader reader;
	Json::Value root;

	ifstream is;  

	is.open("trijson.json", ios::binary);

	if (reader.parse(is, root))
	{
		Json::StyledWriter sw;     //缩进输出
		cout << "缩进输出" << endl;
		cout << sw.write(root) << endl << endl;  //输出到命令行
	}
	return 0;
}
