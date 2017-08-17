//
// Created by jemin on 25.07.16.
//
// Eigen
#include <Eigen/Dense>

// logsNplots
#include "gnuplot_wrapper/gnuplotter.hpp"
#include <unistd.h>

using namespace Eigen;
using namespace RAI::Utils;
using namespace RAI::Utils::Graph;

using RAI::Utils::graph;


int main() {

  // 2D example ////
  {
    MatrixXf x_data(51, 1), y_data(51, 1), z_data(51, 1);
    x_data = VectorXf::LinSpaced(51, -2, 2).transpose();
    y_data = x_data.array().cos();
    z_data = x_data.array().sin();
    MatrixXf std = VectorXf::Ones(51, 1).transpose() * 0.1;

    FigProp2D figure1properties;
    figure1properties.xlabel = "state";
    figure1properties.ylabel = "value";
    figure1properties.title = "v function";
    graph->figure(1, figure1properties);
    graph->appendDataAndStd(1, x_data.data(), y_data.data(), std.data(), 51, "test", "lt 1 lc 10 lw 2");
    graph->appendData(1, x_data.data(),
                                  z_data.data(),
                                  51,
                                  PlotMethods2D::linespoints,
                                  "test",
                                  "lt 3 lc \"black\" lw 2 pt 2 ps 1.0 pi 10");
    graph->drawFigure(1);
    graph->waitForEnter();
  }

  // 3D example////
  {
    MatrixXf x_data(51, 51);
    MatrixXf y_data(51, 51);
    MatrixXf z_data(51, 51);
    MatrixXf w_data(51, 51);

    for (int i = 0; i < 51; i++) {
      x_data.col(i) = VectorXf::LinSpaced(51, -2, 2);
      y_data.row(i) = VectorXf::LinSpaced(51, -2, 2).transpose();
    }

    z_data = (x_data.array().square() + y_data.array().square()).sqrt().sin();
    w_data = (x_data.array().square() + y_data.array().square()).sqrt().cos();

    FigProp3D figure1properties;
    figure1properties.xlabel = "state";
    figure1properties.ylabel = "value";
    figure1properties.title = "v function";
    figure1properties.palette = "rgb 23,28,3";

    figure1properties.displayType = DisplayType3D::points3D;

    graph->figure3D(2, figure1properties);
    graph->append3D_Data(2, x_data.data(), y_data.data(), z_data.data(), 51 * 51, false,
                         PlotMethods3D::points, "test");
    graph->append3D_Data(2, x_data.data(), y_data.data(), w_data.data(), 51 * 51, true,
                         PlotMethods3D::dots, "test");
    graph->drawFigure(2);

    figure1properties.displayType = DisplayType3D::heatMap2D;
    graph->drawHeatMap(3, figure1properties, x_data.data(), y_data.data(), z_data.data(), 51, 51, "");
    graph->drawFigure(3);
    figure1properties.displayType = DisplayType3D::heatMap3D;
    graph->drawHeatMap(4, figure1properties, x_data.data(), y_data.data(), z_data.data(), 51, 51, "test");
    graph->drawFigure(4);
  }

  graph->waitForEnter();


  //// moving figure
  for (int timeID = 0; timeID < 100; timeID++) {

    MatrixXf x_data(51, 1), y_data(51, 1), z_data(51, 1);
    x_data = VectorXf::LinSpaced(51, -2, 2).transpose();
    y_data = (x_data.array() + 0.02 * timeID).cos();

    Graph::FigProp2D figure1properties;
    graph->figure(1, figure1properties);
    graph->appendData(1, x_data.data(),
                      y_data.data(),
                      51,
                      PlotMethods2D::linespoints,
                      "test",
                      "lt 3 lc \"black\" lw 2 pt 2 ps 1.0 pi 10");
    graph->drawFigure(1);

    usleep(50000);
  }

  graph->waitForEnter();

  //// moving figure
  for (int timeID = 0; timeID < 100; timeID++) {

    MatrixXf x_data(51, 1), y_data(51, 1), z_data(51, 1);
    x_data = VectorXf::LinSpaced(51, -2, 2).transpose();
    y_data = (x_data.array() + 0.02 * timeID).cos();
    z_data = (x_data.array() + 0.02 * timeID).sin();

    FigProp2D figure1properties;
    graph->figure(1, figure1properties);
    graph->appendData(1, x_data.data(),
                      y_data.data(),
                      51,
                      PlotMethods2D::linespoints,
                      "test",
                      "lt 3 lc 0 lw 2 pt 2 ps 1.0 pi 10");
    graph->appendData(1, x_data.data(), z_data.data(), 51, PlotMethods2D::lines, "test", "lt 3 lc 1 lw 2");
    graph->drawFigure(1);

    usleep(50000);
  }

}