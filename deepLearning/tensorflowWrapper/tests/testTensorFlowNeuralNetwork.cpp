#include <iostream>
#include <vector>
#include <utility>

#include <glog/logging.h>
#include "tensorflowWrapper/TensorFlowNeuralNetwork.hpp"

using std::cout;
using std::endl;
using std::cin;
using RAI::FuncApprox::TensorFlowNeuralNetwork;

template<typename Dtype>
void perform_N_SolverIterations(TensorFlowNeuralNetwork<Dtype> &network,
                                const std::vector<std::pair<std::string, Eigen::Matrix<Dtype, Eigen::Dynamic, Eigen::Dynamic>>> &namedData,
                                Dtype learningRate,
                                int nIterations) {
  std::vector<std::pair<std::string, Eigen::Matrix<Dtype, Eigen::Dynamic, Eigen::Dynamic>>> namedDataAndLearningRate = namedData;
  Eigen::Matrix<Dtype, Eigen::Dynamic, Eigen::Dynamic> learningRateData(1, 1);
  learningRateData << learningRate;

  namedDataAndLearningRate.push_back({"trainUsingTargetOutput/learningRate", learningRateData});
  std::vector<Eigen::Matrix<Dtype, Eigen::Dynamic, Eigen::Dynamic>> outputs;

  for (int i = 0; i < nIterations; ++i) {
    network.run(namedDataAndLearningRate, {}, {"trainUsingTargetOutput/solver"}, outputs);
  }
}

using Dtype = float;
using MatrixXD = Eigen::Matrix<Dtype, Eigen::Dynamic, Eigen::Dynamic>;
using VectorXD = Eigen::Matrix<Dtype, Eigen::Dynamic, 1>;

int main() {

  cout << "Generating all *.pb" << endl;

  int error;
  error = system("cd resources; ./run_python_scripts.sh");

  LOG_IF(FATAL, error) << "There was an error with calling run_python_scripts.sh";

  if (error) {
    cout << "There was an error with calling run_generate_scripts.sh" << endl;
    return -1;
  }
if(false) {
  {
    cout << "Testing simple TensorFlow graph" << endl;

    TensorFlowNeuralNetwork<Dtype> testNet("resources/simple_graph.pb");

    std::vector<std::pair<std::string, MatrixXD> > inputs;
    std::vector<MatrixXD> outputs;

    MatrixXD aData(1, 1);
    aData << 3;

    MatrixXD bData(1, 1);
    bData << 2;

    inputs = {{"a", aData},
              {"b", bData}};

    testNet.forward(inputs, {"c"}, outputs);

    cout << aData << "*" << bData << " = " << endl;
    cout << "Ground truth: " << aData * bData << endl;
    cout << "Output TensorFlowNeuralNetwork: " << outputs[0] << endl;
    cout << "Press Enter to continue" << endl;
    cin.get();
  }

  {
    cout << "Testing extractNumberOfLearnableScalars" << endl;
    TensorFlowNeuralNetwork<Dtype> simpleMLP("resources/simple_mlp.pb");
    cout << "Number of learnable scalars: " << simpleMLP.extractNumberOfLearnableScalars()
         << " (should be 121401 = 1*300 + 300 + 300*400 + 400 + 400*1 + 1)" << endl;
    cout << "Press Enter to continue" << endl;
    cin.get();
  }

  {
    cout << "Testing simple multilayer perceptron" << endl;

    TensorFlowNeuralNetwork<Dtype> simpleMLP("resources/simple_mlp.pb");

    // Trying to learn the following function:
    // sin(3*x)

    int nBatches = 500;
    int batchSize = 1920;
    int nIterationsPerBatch = 5;

    for (int batch = 0; batch < nBatches; ++batch) {
      // Generating data:
      MatrixXD inputData = MatrixXD::Random(batchSize, 1) * 3;
      MatrixXD targetOutputData = MatrixXD(batchSize, 1);

      targetOutputData = (inputData.col(0).array() * 3).sin();

      std::vector<std::pair<std::string, MatrixXD>> namedData = {{"input",        inputData},
                                                                 {"targetOutput", targetOutputData}};
      perform_N_SolverIterations<Dtype>(simpleMLP, namedData, 1e-3, nIterationsPerBatch);

      if (batch % 100 == 0) {
        cout << batch << endl;
      }
    }

    int nTestDataPoints = 1;
    MatrixXD testInputData(nTestDataPoints, 1);
    testInputData << 3.1459 / (4 * 3);

    std::vector<std::pair<std::string, MatrixXD>> namedTestInputData = {{"input", testInputData}};
    std::vector<MatrixXD> outputs;

    simpleMLP.forward(namedTestInputData, {"output"}, outputs);

    MatrixXD testOutputData = outputs[0];

    cout << "Network output: " << testOutputData << endl;
    cout << "Desired output: " << sin(testInputData(0, 0) * 3) << endl;
    cout << "Press Enter to continue" << endl;
    cin.get();
  }

  {
    using Dtype = double;
    using MatrixXD = Eigen::Matrix<Dtype, Eigen::Dynamic, Eigen::Dynamic>;
    cout << "Testing simple multilayer perceptron (double/float64)" << endl;

    TensorFlowNeuralNetwork<Dtype> simpleMLP("resources/simple_mlp_float64.pb");

    // Trying to learn the following function:
    // sin(3*x)

    int nBatches = 500;
    int batchSize = 20;
    int nIterationsPerBatch = 5;

    for (int batch = 0; batch < nBatches; ++batch) {
      // Generating data:
      MatrixXD inputData = MatrixXD::Random(batchSize, 1) * 3;
      MatrixXD targetOutputData = MatrixXD(batchSize, 1);

      targetOutputData = (inputData.col(0).array() * 3).sin();

      std::vector<std::pair<std::string, MatrixXD>> namedData = {{"input",        inputData},
                                                                 {"targetOutput", targetOutputData}};
      perform_N_SolverIterations<Dtype>(simpleMLP, namedData, 1e-3, nIterationsPerBatch);

      if (batch % 100 == 0) {
        cout << batch << endl;
      }
    }

    int nTestDataPoints = 1;
    MatrixXD testInputData(nTestDataPoints, 1);
    testInputData << 3.1459 / (4 * 3);

    std::vector<std::pair<std::string, MatrixXD>> namedTestInputData = {{"input", testInputData}};
    std::vector<MatrixXD> outputs;

    simpleMLP.forward(namedTestInputData, {"output"}, outputs);

    MatrixXD testOutputData = outputs[0];

    cout << "Network output: " << testOutputData << endl;
    cout << "Desired output: " << sin(testInputData(0, 0) * 3) << endl;
    cout << "Press Enter to continue" << endl;
    cin.get();
  }

  {
    cout << "Testing copyWeightsFromOtherNetwork" << endl;

    TensorFlowNeuralNetwork<Dtype> simpleMLPWithTransferableWeights(
        "resources/simple_mlp_with_transferable_weights.pb");
    TensorFlowNeuralNetwork<Dtype> simpleMLPWithTransferableWeights2(
        "resources/simple_mlp_with_transferable_weights2.pb");

    std::vector<tensorflow::Tensor> outputs;
    simpleMLPWithTransferableWeights.session->Run({}, {"hiddenLayer1/b"}, {}, &outputs);
    cout << "Old weight: " << outputs[0].vec<Dtype>()(0) << " (should be 0.1)" << endl;

    simpleMLPWithTransferableWeights.copyWeightsFromOtherNetwork(&simpleMLPWithTransferableWeights2);

    simpleMLPWithTransferableWeights.session->Run({}, {"hiddenLayer1/b"}, {}, &outputs);
    cout << "Weight after copying: " << outputs[0].vec<Dtype>()(0) << " (should be 2.0)" << endl;
  }

  {
    cout << "Testing interpolateWeightsWithWeightsOfOtherNetwork" << endl;

    TensorFlowNeuralNetwork<Dtype> simpleMLPWithTransferableWeights(
        "resources/simple_mlp_with_transferable_weights.pb");
    TensorFlowNeuralNetwork<Dtype> simpleMLPWithTransferableWeights2(
        "resources/simple_mlp_with_transferable_weights2.pb");

    std::vector<tensorflow::Tensor> outputs;
    simpleMLPWithTransferableWeights.session->Run({}, {"hiddenLayer1/b"}, {}, &outputs);
    cout << "Old weight: " << outputs[0].vec<Dtype>()(0) << " (should be 0.1)" << endl;

    Dtype tau = 0.001;

    simpleMLPWithTransferableWeights.interpolateLP(&simpleMLPWithTransferableWeights2,
                                                   tau);

    simpleMLPWithTransferableWeights.session->Run({}, {"hiddenLayer1/b"}, {}, &outputs);
    cout << "Weight after interpolating: " << outputs[0].vec<Dtype>()(0) << " (should be 0.1*(1 - " << tau
         << ") + 2*" << tau << " = " << "0.1019)" << endl;
    cout << "Press Enter to continue" << endl;
    cin.get();
  }

  {
    cout << "Testing getGraphDef and setGraphDef" << endl;

    TensorFlowNeuralNetwork<Dtype> simpleGraph("resources/simple_graph.pb");
    TensorFlowNeuralNetwork<Dtype> simpleMLP("resources/simple_mlp.pb");

    simpleMLP.setGraphDef(simpleGraph.getGraphDef());

    std::vector<std::pair<std::string, MatrixXD> > inputs;
    std::vector<MatrixXD> outputs;

    MatrixXD aData(1, 1);
    aData << 3;

    MatrixXD bData(1, 1);
    bData << 2;

    inputs = {{"a", aData},
              {"b", bData}};

    simpleMLP.forward(inputs, {"c"}, outputs);

    cout << aData << "*" << bData << " = " << endl;
    cout << "Ground truth: " << aData * bData << endl;
    cout << "Output TensorFlowNeuralNetwork: " << outputs[0] << endl;
    cout << "Press Enter to continue" << endl;
    cin.get();
  }

  {
    TensorFlowNeuralNetwork<Dtype> simpleGraph("resources/simple_graph.pb");
    TensorFlowNeuralNetwork<Dtype> simpleGraph2(simpleGraph.getGraphDef());

    std::vector<std::pair<std::string, MatrixXD> > inputs;
    std::vector<MatrixXD> outputs;

    MatrixXD aData(1, 1);
    aData << 3;

    MatrixXD bData(1, 1);
    bData << 2;

    inputs = {{"a", aData},
              {"b", bData}};

    simpleGraph2.forward(inputs, {"c"}, outputs);

    cout << aData << "*" << bData << " = " << endl;
    cout << "Ground truth: " << aData * bData << endl;
    cout << "Output TensorFlowNeuralNetwork: " << outputs[0] << endl;
    cout << "Press Enter to continue" << endl;
    cin.get();
  }
}
  {
    cout << "Testing getTrainableParametersAsOneVector and setTrainableParametersAsOneVector" << endl;

    TensorFlowNeuralNetwork<Dtype> simpleMLP("resources/simple_mlp.pb");

    // Trying to learn the following function:
    // sin(3*x)

    int nBatches = 500;
    int batchSize = 1920;
    int nIterationsPerBatch = 5;

    for (int batch = 0; batch < nBatches; ++batch) {
      // Generating data:
      MatrixXD inputData = MatrixXD::Random(batchSize, 1) * 3;
      MatrixXD targetOutputData = MatrixXD(batchSize, 1);

      targetOutputData = (inputData.col(0).array() * 3).sin();

      std::vector<std::pair<std::string, MatrixXD>> namedData = {{"input",        inputData},
                                                                 {"targetOutput", targetOutputData}};
      perform_N_SolverIterations<Dtype>(simpleMLP, namedData, 1e-3, nIterationsPerBatch);

      if (batch % 100 == 0) {
        cout << batch << endl;
      }
    }

    TensorFlowNeuralNetwork<Dtype> simpleMLP2("resources/simple_mlp.pb");

    VectorXD parameterVector;
    simpleMLP.getLP(parameterVector);
    simpleMLP2.setLP(parameterVector);

    int nTestDataPoints = 1;
    MatrixXD testInputData(nTestDataPoints, 1);
    testInputData << 3.1459 / (4 * 3);

    std::vector<std::pair<std::string, MatrixXD>> namedTestInputData = {{"input", testInputData}};
    std::vector<MatrixXD> outputs;

    simpleMLP2.forward(namedTestInputData, {"output"}, outputs);

    MatrixXD testOutputData = outputs[0];

    cout << "Network output: " << testOutputData << endl;
    cout << "Desired output: " << sin(testInputData(0, 0) * 3) << endl;
    cout << "Press Enter to continue" << endl;
    cin.get();
  }

  cout << "Tests done" << endl;
}
