#include <bits/stdc++.h>
using namespace std;

int main()
{
    vector <int> a[2];
    int n;
    cin >> n;
    for(int i = 1 ; i <= n ; i++){
        int number;
        cin >> number;
        if(number % 2 == 0){
            a[0].push_back(i);
        }else{
            a[1].push_back(i);
        }
    }
    int even = a[0].size();
    int odd = a[1].size();

    if(even == 1){
        cout << a[0][0] << endl;;
    }if(odd == 1){
        cout << a[1][0] << endl;
    }
    return 0;
}
