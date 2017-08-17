/* author: jemin hwangbo*/

#include <iomanip>

#include "tasks/poleBalancing1D/visualizer/visualizer.hpp"
#include "math.h"

using namespace std;

Visualizer::Visualizer() {

	// initialize
	std::string programName = "1D Pole Balance";
	char *argv = new char[programName.length() + 1];
	strcpy(argv, programName.c_str());
  int argc = 1;
	glutInit(&argc, &argv);
	delete argv;

	// request double buffered true color window with Z-buffer
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);

	// create window
	windowWidth = 600;
	windowHeight = 400;
	windowAspectRatio = (double) windowWidth / (double) windowHeight;

	glutInitWindowPosition(0, 0);
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

	updateView();
}

Visualizer::~Visualizer() {}

void Visualizer::drawWorld(float angle, std::string info) {


	updateProjection();

	//  Clear screen and Z-buffer
	updateView();
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
	updateView();
	std::vector<std::vector<float> > vertices;
	vertices.resize(8);
	float topMidY, topMidZ, xOff, yOff, zOff;
	angle = angle * 3.14/2.0;
	topMidY = sin(angle);
	topMidZ = cos(angle);
	yOff = cos(angle) * 0.03f;
	zOff = sin(angle) * 0.03f;
	xOff = 0.1f;

	vertices[0].push_back(0.0f);
	vertices[0].push_back(topMidY - yOff);
	vertices[0].push_back(topMidZ + zOff);

	vertices[1].push_back(0.0f);
	vertices[1].push_back(topMidY + yOff);
	vertices[1].push_back(topMidZ - zOff);

	vertices[2].push_back(-xOff);
	vertices[2].push_back(topMidY + yOff);
	vertices[2].push_back(topMidZ - zOff);

	vertices[3].push_back(-xOff);
	vertices[3].push_back(topMidY - yOff);
	vertices[3].push_back(topMidZ + zOff);

	vertices[4].push_back(0.0f);
	vertices[4].push_back(-yOff);
	vertices[4].push_back(zOff);

	vertices[5].push_back(0.0f);
	vertices[5].push_back(yOff);
	vertices[5].push_back(-zOff);

	vertices[6].push_back(-xOff);
	vertices[6].push_back(yOff);
	vertices[6].push_back(-zOff);

	vertices[7].push_back(-xOff);
	vertices[7].push_back(-yOff);
	vertices[7].push_back(zOff);

	glBegin(GL_QUADS); // 2x2 pixels
	// Top face (y = 1.0f)
	// Define vertices in counter-clockwise (CCW) order with normal pointing out

	// Front face  (z = 1.0f)
	glColor3f(1.0f, 0.0f, 0.0f);     // Red
	glVertex3f( vertices[0][0], vertices[0][1], vertices[0][2]);
	glVertex3f( vertices[1][0], vertices[1][1], vertices[1][2]);
	glVertex3f( vertices[5][0], vertices[5][1], vertices[5][2]);
	glVertex3f( vertices[4][0], vertices[4][1], vertices[4][2]);

	glEnd();  // End of drawing color-cube
	glColor3f(0.0f, 0.0f, 0.0f);
	glRasterPos3d(0.0, -0.2, 1.2);
	glutBitmapString(GLUT_BITMAP_9_BY_15, (unsigned char*) info.c_str());

	glFlush();
	glutSwapBuffers();
}


void Visualizer::updateView() {
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(10.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 3.0f);
	updateLightPositions();
}

void Visualizer::updateLightPositions() {
	glLightfv(GL_LIGHT0, GL_POSITION, position0);
}

void Visualizer::updateProjection() {
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-1.2, 1.2, -0.3, 1.3, -1.0, 50.0);

}