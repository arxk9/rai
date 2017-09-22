//
// Created by jhwangbo on 20.09.16.
//


using Dtype = float;

// logsNplots

// random sampling
#include "rai/common/math/RandomNumberGenerator.hpp"
#include <rai/RAI_core>

int main(int argc, char* argv[]) {

/////////////////////// testing uniform sampling ////////////////////

  rai::RandomNumberGenerator<float> randNumb;
  Dtype sampleNump[2];
  Dtype X_sphere[1000];
  Dtype Y_sphere[1000];

  for (int i = 0; i < 1000; i++) {
    randNumb.sampleInUnitSphere<2>(sampleNump);
    X_sphere[i] = sampleNump[0];
    Y_sphere[i] = sampleNump[1];
  }

  rai::Utils::Graph::FigProp2D figurePropertiestest;
  figurePropertiestest.title = "state vs value";
  figurePropertiestest.xlabel = "state";
  figurePropertiestest.ylabel = "value";

  rai::Utils::graph->figure(1, figurePropertiestest);
  rai::Utils::graph->appendData(1, X_sphere, Y_sphere, 1000, rai::Utils::Graph::PlotMethods2D::points, "in sphere", "pt 3 ps 2");
  rai::Utils::graph->drawFigure(1);

  std::cout<<"the samples should be confined in a unit circle"<<std::endl;
  rai::Utils::graph->waitForEnter();


  rai::Vector<int> order;
  order.resize(5);
  for(int i=0; i< 5; i++) order[i] = i;
  int sum = 0;

  for(int k=0; k<50000; k++) {
    randNumb.shuffleSTDVector(order);
    sum += order[4];
    if(k<20)
      std::cout<<"random sequence samples "<<order[0]<<order[1]<<order[2]<<order[3]<<order[4]<<std::endl;
  }

  std::cout<<"average should be 2: "<<float(sum)/50000.0f<<std::endl;

  rai::Utils::graph->waitForEnter();

  Eigen::RowVectorXi sample(10), sample2(10), sample3(10);
  sample<<1,2,3,4,5,6,7,8,9,10;
  sample2<<1,2,3,4,5,6,7,8,9,10;
  sample3<<1,2,3,4,5,6,7,8,9,10;

  randNumb.shuffleColumnsOfThreeMatrices(sample, sample2, sample3);
  std::cout<<"matrix order: "<<sample<<std::endl;
  std::cout<<"matrix order: "<<sample2<<std::endl;
  std::cout<<"matrix order: "<<sample3<<std::endl;


}