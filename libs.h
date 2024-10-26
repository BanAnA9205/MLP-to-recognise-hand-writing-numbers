#include "bits/stdc++.h"


using namespace std;

// funtion declaration
//void randPopulate(vector<vector<vector<double>>> &v, mt19937& rng);
void randPopulate(vector<vector<double>> &v, mt19937& rng);
void randPopulate(vector<double> &v, mt19937& rng);
double sigmoid(double x);
double sigmoidDeriv(double x);
double ReLuderiv(double x);
vector<double> softmax(vector<double> &v);
void gradDescent(vector<vector<double>> &W, vector<vector<double>> &WDeriv, double learningRate, int size);
void gradDescent(vector<double> &W, vector<double> &WDeriv, double learningRate, int size);

void normalize(vector<double> &v);

string getCurrentTime();