#include <bits/stdc++.h>
#include "libs.h"

using namespace std;

// A SIMPLE NEURAL NETWORK FROM A FIRST-TIMER
// Author: Nguyen Van An 
// (not really since i just do the thing that's been done millions of times 
//  so, code-wise? yes, i'm the author. idea-wise? nah)
// This is a simple neural network that is trained on the MNIST dataset of handwritten digits. 

// The network has 784 input neurons, 2 hidden layers with 16 neurons each, and 10 output neurons.
// Dataset can be found at: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
// The dataset is a CSV file, with 60000 rows and 785 columns. 
// The first column is the label, and the remaining 784 columns are the pixel values of the image.
// The test dataset is a CSV file with 10000 rows and 785 columns having the same format as the training dataset.

// ========================================================================= //
// ============================== DATA IMPORT ============================== //
// ========================================================================= //
vector<int> trainAns(60000);
vector<vector<int>> trainData(60000, vector<int>(784));
vector<int> testAns(10000);
vector<vector<int>> testData(10000, vector<int>(784));

void init_train(int sz, int width){
    ifstream file("MNIST_dataset/mnist_train_noheader.csv");
    if (!file.is_open()) throw runtime_error("Could not open file");
    // Copied from here: https://www.gormanalysis.com/blog/reading-and-writing-csv-files-with-cpp/

    string line;
    for (int i = 0; i < sz; i++){
        getline(file, line); stringstream ss(line);
        string token; getline(ss, token, ',');

        trainAns[i] = stoi(token);
        for (int j = 0; j < width; j++){
            getline(ss, token, ',');
            trainData[i][j] = stoi(token);
        }
    }
}

void init_test(){
    ifstream file("MNIST_dataset/mnist_test_noheader.csv");
    if (!file.is_open()) throw runtime_error("Could not open file");

    string line;
    for (int i = 0; i < 10000; i++){
        getline(file, line); stringstream ss(line);
        string token; getline(ss, token, ',');

        testAns[i] = stoi(token);
        for (int j = 0; j < 784; j++){
            getline(ss, token, ',');
            testData[i][j] = stoi(token);
        }
    }
}

string importModel(vector<vector<double>> &Win, vector<vector<vector<double>>> &W, vector<vector<double>> &Wout,
                   vector<vector<double>> &bias, vector<double> &biasOut,
                   int &trainSize, int &width, int layers = 3){
    cout << "Select a model to import: ";
    string s; getline(cin, s);
    s = "weights_records/" + s + ".txt";
    cout << "Opening " << s << "...\n";
    ifstream file(s);
    if (!file.is_open()) throw runtime_error("Could not open model to import");

    file >> trainSize >> width >> layers;
    Win.resize(width, vector<double>(16));
    W.resize(layers - 1, vector<vector<double>>(16, vector<double>(16)));
    Wout.resize(16, vector<double>(10));
    bias.resize(layers, vector<double>(16));
    biasOut.resize(10);

    for (int i = 0; i < width; i++)
        for (int j = 0; j < 16; j++) file >> Win[i][j];
    
    for (int l = 0; l < layers - 1; l++)
        for (int i = 0; i < 16; i++)
            for (int j = 0; j < 16; j++) file >> W[l][i][j];
    
    for (int i = 0; i < 16; i++)
        for (int j = 0; j < 10; j++) file >> Wout[i][j];
    
    for (int l = 0; l < layers; l++)
        for (int i = 0; i < 16; i++) file >> bias[l][i];
    
    for (int i = 0; i < 10; i++) file >> biasOut[i];

    return s;
}
// ======================================================================== //
// ======================================================================== //
// ======================================================================== //



// =============================================================================== //
// ============================== TRAINING FUNCTION ============================== //
// =============================================================================== //
vector<double> forwardProp(vector<vector<double>> &Win, vector<vector<vector<double>>> &W, vector<vector<double>> &Wout,
                           vector<vector<double>> &bias, vector<double> &biasOut,
                           vector<int> &data){
    vector<double> localData(784);
    int layers = W.size() + 1;
    for (int i = 0; i < 784; ++i) localData[i] = double(data[i])/255;
    vector<vector<double>> z(layers, vector<double>(16)), a(layers, vector<double>(16));
    for (int col = 0; col < 16; ++col){
        for (int row = 0; row < 784; ++row)
            z[0][col] += localData[row]*Win[row][col];

        z[0][col] += bias[0][col];
        //a[0][col] = max(0.0, z[0][col]); // ReLU
        a[0][col] = sigmoid(z[0][col]); // Sigmoid
    }

    for (int l = 1; l < layers; l++){
        for (int col = 0; col < 16; ++col){
            for (int row = 0; row < 16; ++row)
                z[l][col] += a[l - 1][row]*W[l - 1][row][col];
            
            z[l][col] += bias[l][col];
            //a[l][col] = max(0.0, z[l][col]); // ReLU
            a[l][col] = sigmoid(z[l][col]); // Sigmoid
        }
    }

    vector<double> zOut(10);
    for (int col = 0; col < 10; ++col){
        for (int row = 0; row < 16; ++row)
            zOut[col] += a[layers - 1][row]*Wout[row][col];
        
        zOut[col] += biasOut[col];
        //forOut[col] = max(0.0, zOut[col]); // ReLU
        //forOut[col] = sigmoid(zOut[col]); // Sigmoid
    }

    vector<double> aOut = softmax(zOut);
    return aOut;
}

double loss(vector<vector<double>> &Win, vector<vector<vector<double>>> &W, vector<vector<double>> &Wout,
            vector<vector<double>> &bias, vector<double> &biasOut,
            vector<int> &perm, int trainSize, int width){
    double ret = 0;
    for (int i = 0; i < trainSize; ++i){
        vector<double> aOut = forwardProp(Win, W, Wout, bias, biasOut, trainData[perm[i]]);
        for (int j = 0; j < 10; ++j){
            ret += (aOut[j] - (trainAns[perm[i]] == j))*(aOut[j] - (trainAns[perm[i]] == j));
        }
    }
    return ret/trainSize;
}

void trainModel(vector<vector<double>> &Win, vector<vector<vector<double>>> &W, vector<vector<double>> &Wout,
                vector<vector<double>> &bias, vector<double> &biasOut,
                int trainSize, int width, double learningRate = 0.9){
    // Creating random vectors
    // Read https://codeforces.com/blog/entry/61587
    // mt19937 rng(69420); // for pre-determined seed
    mt19937 rng1(chrono::steady_clock::now().time_since_epoch().count()); // for random seed
    //mt19937 rng2(chrono::steady_clock::now().time_since_epoch().count());
    //mt19937 rng3(chrono::steady_clock::now().time_since_epoch().count());
    int layers = W.size() + 1;  // Win -> W -> Wout
    randPopulate(Win, rng1); randPopulate(Wout, rng1); 
    for (int i = 0; i < layers - 1; ++i) randPopulate(W[i], rng1);

    randPopulate(biasOut, rng1); // bias -> biasOut
    for (int i = 0; i < layers; ++i) randPopulate(bias[i], rng1);

    // Training the model

    int num_epochs = 20;
    for (int epoch = 0; epoch < num_epochs; ++epoch){
        double oldLearningRate = learningRate;
        vector<int> perm(trainSize);
        for (int i = 0; i < trainSize; ++i) perm[i] = i;
        std::shuffle(perm.begin(), perm.end(), rng1);

        int batch_size = 64, cnt = 0;
        int batch_num = (trainSize + batch_size - 1)/batch_size;

        while (batch_num--){
            int iter_bound = min(batch_size, trainSize - cnt);
            int iter_countdown = iter_bound;
            vector<vector<double>> WinDeriv(width, vector<double>(16));
            vector<vector<vector<double>>> WDeriv(layers-1, vector<vector<double>>(16, vector<double>(16)));
            vector<vector<double>> WoutDeriv(16, vector<double>(10));
            vector<double> biasOutDeriv(10);
            vector<vector<double>> biasDeriv(layers, vector<double>(16));

            while (iter_countdown--){
                vector<double> localData(width);
                for (int i = 0; i < width; ++i) 
                    localData[i] = double(trainData[perm[cnt]][i])/255; // A local copy's better, i guess?
                // Forward propagation (using Sigmoid)

                vector<vector<double>> z(layers, vector<double>(16)), a(layers, vector<double>(16));

                for (int col = 0; col < 16; ++col){
                    for (int row = 0; row < width; ++row)
                        z[0][col] += localData[row]*Win[row][col];

                    z[0][col] += bias[0][col];
                    //a[0][col] = max(0.0, z[0][col]); // ReLU
                    a[0][col] = sigmoid(z[0][col]); // Sigmoid
                }

                for (int l = 1; l < layers; l++){
                    for (int col = 0; col < 16; ++col){
                        for (int row = 0; row < 16; ++row)
                            z[l][col] += a[l - 1][row]*W[l - 1][row][col];
                        
                        z[l][col] += bias[l][col];
                        //a[l][col] = max(0.0, z[l][col]); // ReLU
                        a[l][col] = sigmoid(z[l][col]); // Sigmoid
                    }
                }

                vector<double> zOut(10);
                for (int col = 0; col < 10; ++col){
                    for (int row = 0; row < 16; ++row)
                        zOut[col] += a[layers - 1][row]*Wout[row][col];
                    
                    zOut[col] += biasOut[col];
                    //forOut[col] = max(0.0, zOut[col]); // ReLU
                    //forOut[col] = sigmoid(zOut[col]); // Sigmoid
                }

                vector<double> aOut = softmax(zOut);

                // Backward propagation (go watch 3Blue1Brown)
                vector<double> outDeriv(10);
                for (int col = 0; col < 10; ++col)
                    outDeriv[col] = aOut[col] - (trainAns[perm[cnt]] == col);

                vector<vector<double>> aDeriv(layers, vector<double>(16));

                for (int col = 0; col < 10; ++col){
                    double dAdZ = outDeriv[col]*aOut[col]*(1 - aOut[col]); // softmax derivative
                    for (int row = 0; row < 16; ++row){
                        WoutDeriv[row][col] += a[layers - 1][row]*dAdZ;
                        aDeriv[layers - 1][row] += Wout[row][col]*dAdZ;
                    }
                    biasOutDeriv[col] += dAdZ;
                }

                for (int l = layers - 1; l > 0; l--){
                    for (int col = 0; col < 16; ++col){
                        double dAdZ = aDeriv[l][col]*sigmoidDeriv(z[l][col]);
                        for (int row = 0; row < 16; ++row){
                            WDeriv[l - 1][row][col] += a[l - 1][row]*dAdZ;
                            aDeriv[l - 1][row] += W[l - 1][row][col]*dAdZ;
                        }
                        biasDeriv[l][col] += dAdZ;
                    }
                }

                for (int col = 0; col < 16; ++col){
                    double dAdZ = aDeriv[0][col]*sigmoidDeriv(z[0][col]);
                    for (int row = 0; row < width; ++row)
                        WinDeriv[row][col] += localData[row]*dAdZ;
                    
                    biasDeriv[0][col] += dAdZ;
                }

                ++cnt;  // Don't forget to increase the counter 
                        // (like, seriously, how many times have you forgotten this?)
            }

            gradDescent(Win, WinDeriv, learningRate, iter_bound);
            for (int i = 0; i < layers - 1; ++i) gradDescent(W[i], WDeriv[i], learningRate, iter_bound);
            gradDescent(Wout, WoutDeriv, learningRate, iter_bound);
            gradDescent(biasOut, biasOutDeriv, learningRate, iter_bound);
            for (int i = 0; i < layers; ++i) gradDescent(bias[i], biasDeriv[i], learningRate, iter_bound);
        }
        
        cout << "Epoch " << epoch + 1 << " out of " << num_epochs << " completed\n";
        double L = loss(Win, W, Wout, bias, biasOut, perm, trainSize, width);
        cout << "Loss: " << L << '\n';
        if (epoch > 10 && epoch < 15)
            learningRate *= 1/double(1 + exp(epoch/L - num_epochs/pow(L, 0.8)));
        cout << "learning rate: " << oldLearningRate << " -> " << learningRate << '\n';
    }
}

int predict(vector<vector<double>> &Win, vector<vector<vector<double>>> &W, vector<vector<double>> &Wout,
            vector<vector<double>> &bias, vector<double> &biasOut, vector<int> &data){
    vector<double> aOut = forwardProp(Win, W, Wout, bias, biasOut, data);
    int ret = 0;
    for (int i = 1; i < 10; ++i) if (aOut[i] > aOut[ret]) ret = i;

    return ret;
}
// =============================================================================== //
// =============================================================================== //
// =============================================================================== //


// =================================================================== //
// ============================== MAIN =============================== //
// =================================================================== //
int main(int argc, char const *argv[]){
    // Input "new_train" to reinitialize the training data or "reuse_train" 
    // to use the existing training data. 
    // By default, the training data is reused whenever possible.
    // If "new_train" is input, the training size and width can also be specified.
    // (why?) Because by doing so we will be able to experiment with different 
    // training sizes and widths. 
        // For example, how will the neural network behaves when being limited 
        // to only the first 400 pixels of the image?
    // The default training size is 60000 and the default width is 784.
    // The next argument will be the number of test cases to run, which is 10000 by default.
    // The final argument is the learning rate, which is 0.1 by default.
        // Example: ./main new_train 1000 100 8000 0.01
        //          ./main old_train 0 0 0 (all 0s for default values)
        //          ./main new_train (this is the same as the previous one)

    ////////////////////////////////////////////////
    // The model is saved in the following format //
    ////////////////////////////////////////////////

    int layers = 3;
    vector<vector<double>> Win(784, vector<double>(16));
    vector<vector<vector<double>>> W(layers - 1, vector<vector<double>>(16, vector<double>(16)));
    vector<vector<double>> Wout(16, vector<double>(10));
    vector<vector<double>> bias(layers, vector<double>(16));
    vector<double> biasOut(10);
    int trainSize = 60000, width = 784;
    string record;

    if (argc == 1 || (argc > 1 && string(argv[1]) == "new_train")) {
        if (argc > 2) trainSize = (stoi(argv[2]) == 0) ? 60000 : stoi(argv[2]);
        if (argc > 3) width = (stoi(argv[3]) == 0) ? 784 : stoi(argv[3]);
        init_train(trainSize, width); 

        cout << "Training model with " << trainSize << " cases and width " << width << "...\n";
        cout << "Using " << layers << " layers\n";

        double learningRate = 0.9;
        if (argc > 5) learningRate = (stod(argv[5]) == 0) ? 0.9 : stod(argv[5]);

        trainModel(Win, W, Wout, bias, biasOut, trainSize, width, learningRate);

        // Save the model
        record = "weights_records/[" + getCurrentTime() + "]_model.txt";
        ofstream file(record);
        if (!file.is_open()) throw runtime_error("Could not open file to save");

        file << trainSize << " " << width << " " << layers << "\n";
        for (int i = 0; i < width; i++){
            for (int j = 0; j < 16; j++) file << fixed << setprecision(6) << Win[i][j] << " "; 
            file << "\n";
        }

        for (int l = 0; l < layers - 1; l++)
            for (int i = 0; i < 16; i++){
                for (int j = 0; j < 16; j++) file << fixed << setprecision(6) << W[l][i][j] << " "; 
                file << "\n";
            }

        for (int i = 0; i < 16; i++){
            for (int j = 0; j < 10; j++) file << fixed << setprecision(6) << Wout[i][j] << " "; 
            file << "\n";
        }

        for (int l = 0; l < layers; l++){
            for (int i = 0; i < 16; i++) file << fixed << setprecision(6) << bias[l][i] << " "; 
            file << "\n";
        }

        cout << "Model saved successfully\n";
    } 
    else if (argc > 1 && string(argv[1]) == "old_train"){
        record = importModel(Win, W, Wout, bias, biasOut, trainSize, width, layers);
        cout << "Model imported successfully\n";
    } 
    else throw runtime_error("Invalid input");

    int testSize = 10000;
    if (argc > 4) testSize = (stoi(argv[4]) == 0) ? 10000 : stoi(argv[4]);
    init_test();
    cout << "Test dataset imported successfully\n";

    // Testing the model
    vector<int> prediction(testSize), testPerm(testSize);

    for (int i = 0; i < testSize; ++i) testPerm[i] = i;
    shuffle(testPerm.begin(), testPerm.end(), mt19937(chrono::steady_clock::now().time_since_epoch().count()));

    cout << testSize << " randomly chosen test cases are being tested...\n";
    for (int i = 0; i < testSize; ++i)
        prediction[i] = predict(Win, W, Wout, bias, biasOut, testData[testPerm[i]]);
    
    int correct = 0;
    for (int i = 0; i < testSize; ++i)
        if (prediction[i] == testAns[testPerm[i]]) ++correct;
    
    cout << "Model: " << record << "\n";
    cout << "Correct predictions: " << correct << " out of " << testSize << " tests\n";
    cout << "Accuracy: " << (double)correct/testSize*100 << "%\n";
    cout << "Saving test results\n";

    record = "test_results/" + record.substr(16, 19) + "_results.txt";
    ofstream file(record);
    if (!file.is_open()) throw runtime_error("Could not open file to save");

    for (int i = 0; i < testSize; ++i){
        file << "====================\n";
        file << "Test number: " << testPerm[i] << "\n";
        file << "prediction: " << prediction[i] << " - answer: " << testAns[testPerm[i]] << "\n";
        file << "Result: " << ((prediction[i] == testAns[testPerm[i]]) ? "Correct\n" : "Incorrect\n");
        file << "====================\n";
    }

    cout << "Test results saved successfully\n";
    return 0;
}
// =================================================================== //
// =================================================================== //
// =================================================================== //