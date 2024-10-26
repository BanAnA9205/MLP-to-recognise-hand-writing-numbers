#include "libs.h"

using namespace std;

/*void randPopulate(vector<vector<vector<double>>> &v, mt19937& rng){  
    for (int i = 0; i < v.size(); i++)
        for (int j = 0; j < v[0].size(); j++)
            for (int k = 0; k < v[0][0].size(); k++)
                v[i][j][k] = double(uniform_int_distribution(0, 10000)(rng))/10000;
}*/

void randPopulate(vector<vector<double>> &v, mt19937& rng){  
    for (int i = 0; i < v.size(); i++){
        for (int j = 0; j < v[0].size(); j++)
            v[i][j] = uniform_real_distribution(-1.0, 1.0)(rng);
    }
}

void randPopulate(vector<double> &v, mt19937& rng){
    for (int i = 0; i < v.size(); i++)
        v[i] = uniform_real_distribution(-1.0, 1.0)(rng);
}

double sigmoid(double x){
    return 1/(1 + exp(-x));
}

double sigmoidDeriv(double x){
    return sigmoid(x)*(1 - sigmoid(x));
}

double ReLuderiv(double x){
    return (x > 0) ? 1 : 0;
}

vector<double> softmax(vector<double> &v){
    double sum = 0;
    vector<double> ret(v.size());
    for (int i = 0; i < v.size(); i++) {ret[i] = exp(v[i]); sum += ret[i];}
    for (int i = 0; i < v.size(); i++) ret[i] /= sum;
    return ret;
}

void gradDescent(vector<vector<double>> &W, vector<vector<double>> &WDeriv, double learningRate, int size){
    for (int i = 0; i < W.size(); i++)
        for (int j = 0; j < W[0].size(); j++)
            W[i][j] -= learningRate*WDeriv[i][j]/size;
}

void gradDescent(vector<double> &W, vector<double> &WDeriv, double learningRate, int size){
    for (int i = 0; i < W.size(); i++)
        W[i] -= learningRate*WDeriv[i]/size;
}

void normalize(vector<double> &v){
    double sum = 0;
    for (int i = 0; i < v.size(); i++) sum += v[i];
    for (int i = 0; i < v.size(); i++) v[i] /= sum;
};

string getCurrentTime(){
    time_t now = time(0);
    tm *ltm = localtime(&now);
    return to_string(1900 + ltm->tm_year) + "-" + to_string(1 + ltm->tm_mon) + "-" + to_string(ltm->tm_mday) + " " + 
           to_string(ltm->tm_hour) + "_" + to_string(ltm->tm_min) + "_" + to_string(ltm->tm_sec);
}