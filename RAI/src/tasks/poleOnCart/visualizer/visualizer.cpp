/* author: jemin hwangbo*/

#include <iomanip>

#include "tasks/poleOnCart/visualizer/visualizer.hpp"
#include "math.h"
#include "vector"

using namespace std;

Visualizer::Visualizer() {

  // initialize
  std::string programName = "1D Pole Balance";
  char *argv = new char[programName.length() + 1];
  strcpy(argv, programName.c_str());
  int argc = 1;
  glutInit(&argc, &argv);
  delete (argv);

  // request double buffered true color window with Z-buffer
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);

  // create window
  windowWidth = 1200;
  windowHeight = 400;
  windowAspectRatio = (double) windowWidth / (double) windowHeight;

  glutInitWindowPosition(700, 200);
  glutInitWindowSize(windowWidth, windowHeight);
  glutCreateWindow(programName.c_str());

  // enable Z-buffer depth test
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LESS);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glClearColor(0.8f, 0.85f, 1.0f, 1);

  // lights
  glLightfv(GL_LIGHT0, GL_AMBIENT, ambient0);
  glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse0);
  glLightfv(GL_LIGHT0, GL_SPECULAR, specular0);

  glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specular0);
  glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 10.0f);

  glEnable(GL_LIGHTING);
  glEnable(GL_COLOR_MATERIAL);
  glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
  glShadeModel(GL_SMOOTH);
  // Set up the projection parameters
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  updateLightPositions();

  glEnable(GL_LIGHT0);

  // enable transparency
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glEnable(GL_BLEND);

  glEnable(GL_LINE_SMOOTH);
//  glEnable(GL_POLYGON_SMOOTH);
  glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
  glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);
  updateProjection();

  updateView();
}

Visualizer::~Visualizer() {}

void Visualizer::drawWorld(Eigen::Vector2d state, std::string info) {

  double position = state(0);
  double angle = state(1) + M_PI / 2.0;

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

  std::vector<std::vector<float> > vertices;
  vertices.resize(8);
  float topMidX, topMidY, xOff, yOff;
  float poleLength = 0.5;
  topMidX = cos(angle) * poleLength;
  topMidY = sin(angle) * poleLength;
  xOff = sin(angle) * 0.01f;
  yOff = -cos(angle) * 0.01f;

  vertices[0].push_back(topMidX - xOff);
  vertices[0].push_back(topMidY - yOff);

  vertices[1].push_back(topMidX + xOff);
  vertices[1].push_back(topMidY + yOff);

  vertices[2].push_back(xOff);
  vertices[2].push_back(yOff);

  vertices[3].push_back(-xOff);
  vertices[3].push_back(-yOff);

  glBegin(GL_QUADS); // 2x2 pixels
//	// Top face (y = 1.0f)
//	// Define vertices in counter-clockwise (CCW) order with normal pointing out
//
  glColor3f(0.0f, 1.0f, 0.0f);     // Red
  glVertex3f(position + vertices[0][0], vertices[0][1], 0.0f);
  glVertex3f(position + vertices[1][0], vertices[1][1], 0.0f);
  glVertex3f(position + vertices[2][0], vertices[2][1], 0.0f);
  glVertex3f(position + vertices[3][0], vertices[3][1], 0.0f);
  glColor3f(1.0f, 0.0f, 0.0f);     // Red
  glVertex3f(position - 0.15, 0.1, 0.0f);
  glVertex3f(position + 0.15, 0.1, 0.0f);
  glVertex3f(position + 0.15, -0.1, 0.0f);
  glVertex3f(position - 0.15, -0.1, 0.0f);
  glEnd();  // End of drawing color-cube

//  glColor3f(0.0f, 0.0f, 0.0f);     // Red
  DrawCircle(position + topMidX, topMidY, 0.05, 20);

  glColor3f(0.0f, 0.0f, 0.0f);
  glRasterPos3d(0.3, 0.8, 0);
  glutBitmapString(GLUT_BITMAP_9_BY_15, (unsigned char *) info.c_str());

  glFlush();
  glutSwapBuffers();
}

void Visualizer::updateView() {
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  gluLookAt(0.0f, 0.0f, 10.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
  updateLightPositions();
}

void Visualizer::updateLightPositions() {
  glLightfv(GL_LIGHT0, GL_POSITION, position0);
}

void Visualizer::updateProjection() {
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(-3.0, 3.0, -1.0, 1.0, -50.0, 50.0);
}

void Visualizer::DrawCircle(float cx, float cy, float r, int num_segments) {
  glBegin(GL_LINE_LOOP);
  for (int ii = 0; ii < num_segments; ii++) {
    float theta = 2.0f * 3.1415926f * float(ii) / float(num_segments);//get the current angle

    float x = r * cosf(theta);//calculate the x component
    float y = r * sinf(theta);//calculate the y component

    glVertex3f(x + cx, y + cy, 0.0);//output vertex

  }
  glEnd();
}