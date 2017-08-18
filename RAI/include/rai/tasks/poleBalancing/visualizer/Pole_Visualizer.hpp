
#include "RAI_graphics.hpp"
#include "obj/Mesh.hpp"
#include "rai/common/math/RAI_math.hpp"
#include "obj/Cylinder.hpp"
#include "obj/Sphere.hpp"

namespace RAI {
namespace Vis {

class Pole_Visualizer {

 public:
  Pole_Visualizer();

  ~Pole_Visualizer();

  void setTerrain(std::string fileName);
  void drawWorld(HomogeneousTransform &bodyPose, double action);
  Graphics::RAI_graphics* getGraphics();


 private:
  Graphics::RAI_graphics graphics;
  Graphics::Obj::Cylinder Pole;
  Graphics::Obj::Sphere Dot;
  Graphics::Obj::Sphere Dot2;
  RAI::Graphics::Obj::Mesh arrow;
  HomogeneousTransform defaultPose_;
};

}
}
